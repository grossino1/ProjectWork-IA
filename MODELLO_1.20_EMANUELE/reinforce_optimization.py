"""
=============================================================================
CASO 2 — RL Fine-Tuning con Reward Shaping sul Corkscrew (v2-Fixed)
=============================================================================

COSA FA:
  Parte dal modello GOLDEN_STABLE 
  e applica RL SOLO per migliorare la zona problematica 2350-2750m.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import csv
import os
import sys
import time
import argparse
from collections import deque

from gym_torcs import TorcsEnv

# ---------------------------------------------------------------------------
# COSTANTI
# ---------------------------------------------------------------------------
INPUT_SIZE   = 30
OUTPUT_SIZE  = 4
GAMMA        = 0.99
TAU          = 0.005

# FIX #1: LR_ACTOR molto più alto dell'originale (era 1e-6!)
LR_ACTOR     = 1e-4
LR_CRITIC    = 2e-4

BATCH_SIZE   = 256
BUFFER_SIZE  = 200_000

# Zona Cork e transizione
CORK_HARD_START  = 2400.0   # inizio zona a controllo MISTO
CORK_HARD_END    = 2750.0
CORK_BLEND_ZONE  = 100.0    # metri di rampa actor/anchor ai bordi
CRITICAL_POINT   = 2477.0   # cambio di pendenza — il punto critico

# Parametri rumore esplorazione
NOISE_SIGMA_INIT  = 0.08
NOISE_SIGMA_MIN   = 0.01
NOISE_DECAY       = 0.995
COOLDOWN_SIGMA    = 0.02    # sigma ridotta dopo crash ripetuti

# Imitation anchor weight (zona Cork)
IMITATION_WEIGHT  = 0.15    # FIX: più basso per dare più spazio all'RL

# Warmup
ACTOR_WARMUP_EPISODES = 3

# Crash detection
CRASH_WINDOW     = 3        # se crash 3 volte consecutive entro ±50m → cooldown
CRASH_DIST_TOL   = 50.0

# ---------------------------------------------------------------------------
# ARCHITETTURA (identica all'originale per compatibilità checkpoint)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.steer_head = nn.Linear(64, 1)
        self.accel_head = nn.Linear(64, 1)
        self.brake_head = nn.Linear(64, 1)
        self.gear_head  = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.base(state)
        steer = torch.tanh(self.steer_head(features))
        accel = torch.sigmoid(self.accel_head(features))
        brake = torch.sigmoid(self.brake_head(features))
        gear  = torch.sigmoid(self.gear_head(features))
        return torch.cat([steer, accel, brake, gear], dim=-1)


class Critic(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size + OUTPUT_SIZE, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=1))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


# ---------------------------------------------------------------------------
# REPLAY BUFFER con prioritizzazione semplificata
# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """
    Versione semplificata di PER: manteniamo un indice di priorità per ogni
    transizione. Le transizioni nella zona Cork hanno priorità base alta,
    le altre 1.0. Dopo ogni update, la priorità aumenta per errori TD grandi.

    Non usa heapq per semplicità — campionamento O(n) ma sufficiente per
    buffer da 200k con batch 256.
    """

    def __init__(self, capacity: int):
        self.buffer    = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity  = capacity
        self.alpha     = 0.6    # esponente priorità
        self.beta      = 0.4    # esponente IS weights (aumenta nel tempo)

    def push(self, state, action, reward, next_state, done, track_idx):
        # Priorità iniziale: Cork zone vale di più
        base_priority = 2.0 if CORK_HARD_START < track_idx < CORK_HARD_END else 1.0
        self.buffer.append((state, action, reward, next_state, done, track_idx))
        self.priorities.append(base_priority)

    def sample(self, batch_size: int):
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size,
                                   replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done, track_idx = zip(*samples)

        # Importance sampling weights
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return (np.stack(state), np.stack(action), np.stack(reward),
                np.stack(next_state), np.stack(done), np.stack(track_idx),
                indices, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Aggiorna priorità in base agli errori TD."""
        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6
            # Mantenere priorità minima Cork
            if CORK_HARD_START < float(self.buffer[idx][5]) < CORK_HARD_END:
                p = max(p, 2.0)
            self.priorities[idx] = p

    def anneal_beta(self, step: int, total_steps: int):
        """Beta aumenta linearmente da 0.4 a 1.0 nel corso del training."""
        self.beta = min(1.0, 0.4 + 0.6 * step / total_steps)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# REWARD SHAPING
# ---------------------------------------------------------------------------
def compute_reward(obs: dict,
                   prev_obs: dict,
                   done: bool,
                   start_dist_raced: float) -> tuple[float, bool]:
    """
    Reward mirato al problema specifico: il Corkscrew con cambio pendenza.

    Returns:
        (reward, force_done)
    """
    speed     = obs.get('speedX', 0.0)
    angle     = obs.get('angle', 0.0)
    track_pos = obs.get('trackPos', 0.0)
    dist      = obs.get('distFromStart', 0.0)
    rpm       = obs.get('rpm', 0.0)
    gear      = obs.get('gear', 1)
    dist_raced = obs.get('distRaced', 0.0) - start_dist_raced

    prev_dist = prev_obs.get('distFromStart', 0.0)
    prev_speed = prev_obs.get('speedX', 0.0)

    force_done = False

    # ---- Reward base: progressione ----
    progress     = speed * np.cos(angle)
    slip_penalty = abs(speed * np.sin(angle))
    track_penalty = abs(track_pos)

    reward = progress - 0.3 * slip_penalty - 0.6 * track_penalty

    # ---- ZONA CRITICA: avvicinamento al cambio pendenza ----
    # Tra 2350m e CRITICAL_POINT l'auto DEVE decelerare
    if 2350 < dist < CRITICAL_POINT:
        target_speed_ms = 40.0    # ~144 km/h, velocità sicura per l'ingresso
        speed_kmh = speed * 3.6
        if speed_kmh > target_speed_ms:
            # Penalizza velocità eccessiva in avvicinamento
            overspeed_penalty = (speed_kmh - target_speed_ms) * 0.05
            reward -= overspeed_penalty

        # Premia la presenza del freno
        brake_val = obs.get('brake', 0.0)
        if brake_val > 0.1:
            reward += 1.5    # bonus frenata attiva

    # ---- ZONA CRITICA: cambio pendenza esatto ----
    if abs(dist - CRITICAL_POINT) < 30.0:
        # Reward extra per velocità controllata
        if speed * 3.6 < 50.0:
            reward += 3.0
        # Penalità forte per velocità alta
        elif speed * 3.6 > 80.0:
            reward -= 5.0
        # Penalità se fuori centro pista
        if abs(track_pos) > 0.5:
            reward -= 3.0

    # ---- Post-Corkscrew: riaccelerazione ----
    if CRITICAL_POINT < dist < CORK_HARD_END:
        # Premia la ripresa progressiva dell'accelerazione
        if speed > prev_speed:
            reward += 0.5

    # ---- Penalità bordi pista ----
    if abs(track_pos) > 1.8:
        reward -= 20.0
    if abs(track_pos) > 2.1:
        reward -= 100.0
        force_done = True

    # ---- Giro completato ----
    if dist_raced > 3610:
        reward += 500.0
        force_done = True

    # ---- Timeout (Corretto) ----
    # Ferma l'episodio se l'auto è palesemente bloccata all'inizio, 
    # ma esclude la zona della griglia di partenza (3500m - 3610m)
    if dist_raced < 10.0 and 100.0 < dist < 3500.0:
        force_done = True   # Auto incastrata nei muri della prima curva
   

    return reward, force_done


# ---------------------------------------------------------------------------
# BLEND ACTOR/ANCHOR — transizione graduale (FIX del cambio brusco)
# ---------------------------------------------------------------------------
def compute_blend_factor(track_idx: float) -> float:
    """
    Restituisce α ∈ [0, 1] che indica quanto usare l'actor (vs anchor).
    
    α = 0  → 100% anchor
    α = 1  → 100% actor
    
    Transizione lineare nelle zone di bordo di ampiezza CORK_BLEND_ZONE.
    """
    if track_idx < CORK_HARD_START - CORK_BLEND_ZONE:
        return 0.0
    if track_idx > CORK_HARD_END + CORK_BLEND_ZONE:
        return 0.0
    # Rampa di ingresso
    if track_idx < CORK_HARD_START:
        return (track_idx - (CORK_HARD_START - CORK_BLEND_ZONE)) / CORK_BLEND_ZONE
    # Zona piena
    if track_idx <= CORK_HARD_END:
        return 1.0
    # Rampa di uscita
    return 1.0 - (track_idx - CORK_HARD_END) / CORK_BLEND_ZONE


# ---------------------------------------------------------------------------
# PREPROCESSING STATO
# ---------------------------------------------------------------------------
def preprocess_state(S: dict) -> np.ndarray:
    def g(k, d=0.0): return float(S.get(k, d))

    angle     = np.array([g('angle')],    dtype=np.float32) / 3.14159
    gear      = np.array([g('gear', 1)],  dtype=np.float32) / 6.0
    rpm       = np.array([g('rpm')],      dtype=np.float32) / 10000.0
    speed     = np.array([g('speedX'), g('speedY'), g('speedZ')],
                         dtype=np.float32) / 200.0
    track     = np.array(S.get('track', [0]*19),
                         dtype=np.float32) / 200.0
    track_pos = np.array([g('trackPos')], dtype=np.float32) / 3.0
    wheel     = np.array(S.get('wheelSpinVel', [0]*4),
                         dtype=np.float32) / 100.0

    return np.concatenate([angle, gear, rpm, speed, track, track_pos, wheel])


# ---------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------------------------
def reinforce(base_model_path: str = None):
    # ---- Selezione modello base ----
    if base_model_path is None:
        # Preferisce il modello IL se disponibile, poi il GOLDEN_STABLE
        for candidate in ["actor_IL_CASO1.pth", "actor_GOLDEN_STABLE.pth"]:
            if os.path.exists(candidate):
                base_model_path = candidate
                break
    if base_model_path is None or not os.path.exists(base_model_path):
        print("ERRORE: Nessun modello base trovato.")
        print("Esegui prima: python caso1_imitation_learning.py")
        sys.exit(1)

    print(f"[Base model] {base_model_path}")

    # ---- Setup ----
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)

    actor  = Actor(INPUT_SIZE)
    actor.load_state_dict(torch.load(base_model_path))

    # Anchor: copia frozen del modello base
    anchor = Actor(INPUT_SIZE)
    anchor.load_state_dict(torch.load(base_model_path))
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad = False

    target_actor  = Actor(INPUT_SIZE)
    target_actor.load_state_dict(actor.state_dict())

    critic        = Critic(INPUT_SIZE)
    target_critic = Critic(INPUT_SIZE)
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # Scheduler: riduce LR dopo stagnazione
    actor_scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        actor_optimizer, patience=15, factor=0.5, min_lr=1e-6)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, patience=15, factor=0.5, min_lr=1e-6)

    memory = PrioritizedReplayBuffer(BUFFER_SIZE)

    # ---- Log ----
    log_path = "rl_optimization_results_caso2.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "steps", "reward", "last_lap_time",
        "dist_raced", "max_dist_from_start",
        "cork_crashes", "noise_sigma", "lr_actor"
    ])

    # ---- Stato training ----
    noise_sigma      = NOISE_SIGMA_INIT
    crash_history    = deque(maxlen=CRASH_WINDOW)   # dist. di ogni crash
    best_dist_raced  = 0.0
    best_lap_time    = float("inf")
    total_updates    = 0

    print("\n" + "=" * 60)
    print("CASO 2 — RL Fine-tuning Corkscrew")
    print("=" * 60)

    for episode in range(1000):
        # ---- Connessione TORCS con retry ----
        connected = False
        relaunch  = (episode == 0)
        while not connected:
            try:
                env.reset(relaunch=relaunch)
                connected = True
            except Exception as e:
                print(f"[TORCS] Retry connessione: {e}")
                relaunch = True
                time.sleep(5.0)

        obs              = env.client.S.d
        prev_obs         = obs.copy()
        state            = preprocess_state(obs)
        start_dist_raced = obs.get('distRaced', 0.0)
        episode_reward   = 0.0
        done             = False
        steps            = 0
        max_dist_from_start = 0.0
        cork_crash_this_ep  = False

        # Cooldown automatico: se crash concentrati nella stessa zona
        recent_crashes = list(crash_history)
        if (len(recent_crashes) >= CRASH_WINDOW and
                all(abs(d - CRITICAL_POINT) < CRASH_DIST_TOL
                    for d in recent_crashes)):
            effective_sigma = COOLDOWN_SIGMA
            print(f"  [Cooldown] Crash ripetuti a ~{CRITICAL_POINT}m "
                  f"→ σ={COOLDOWN_SIGMA}")
        else:
            effective_sigma = noise_sigma

        actor.eval()

        while not done:
            state_t     = torch.FloatTensor(state).unsqueeze(0)
            track_idx   = float(obs.get('distFromStart', 0.0))
            alpha        = compute_blend_factor(track_idx)

            max_dist_from_start = max(max_dist_from_start, track_idx)

            with torch.no_grad():
                actor_action  = actor(state_t).numpy()[0]
                anchor_action = anchor(state_t).numpy()[0]

            # Blend continuo actor/anchor
            raw_action = (alpha * actor_action
                          + (1.0 - alpha) * anchor_action)

            # FIX #3: esplorazione su TUTTI e 4 gli output nella zona Cork
            if alpha > 0.5 and episode >= ACTOR_WARMUP_EPISODES:
                noise_vec = np.random.normal(
                    0, effective_sigma, size=OUTPUT_SIZE
                ).astype(np.float32)
                # Rumore maggiore su steer e brake (i più importanti al Cork)
                noise_vec[0] *= 1.5    # steer
                noise_vec[2] *= 2.0    # brake
                raw_action = raw_action + noise_vec

            # Clip e conversione gear
            env_action = raw_action.copy()
            env_action[0] = np.clip(env_action[0], -1.0,  1.0)  # steer
            env_action[1] = np.clip(env_action[1],  0.0,  1.0)  # accel
            env_action[2] = np.clip(env_action[2],  0.0,  1.0)  # brake
            gear = int(round(np.clip(raw_action[3], -1.0, 1.0) * 5.0 + 1.0))
            env_action[3] = float(max(1, min(6, gear)))

            try:
                _, _, env_done, _ = env.step(env_action)
                if env_done:
                    done = True
            except Exception as e:
                print(f"[TORCS] Errore step: {e}")
                done = True
                break

            obs = env.client.S.d
            if not obs:
                break

            custom_reward, force_done = compute_reward(
                obs, prev_obs, done, start_dist_raced
            )
            if force_done:
                done = True

            dist_raced = obs.get('distRaced', 0.0) - start_dist_raced

            # Registro crash nella zona Cork
            if done and alpha > 0.3:
                crash_history.append(track_idx)
                cork_crash_this_ep = True

            episode_reward += custom_reward
            memory.push(
                state, raw_action, custom_reward,
                preprocess_state(obs), done, track_idx
            )
            state     = preprocess_state(obs)
            prev_obs  = obs.copy()
            steps    += 1

            if steps > 12000:
                done = True

        # ---- Update RL ----
        if len(memory) > BATCH_SIZE and episode >= ACTOR_WARMUP_EPISODES:
            num_updates = min(100, steps // 4)
            actor.train()

            for upd_step in range(num_updates):
                total_updates += 1
                memory.anneal_beta(total_updates, total_steps=500_000)

                (b_s, b_a, b_r, b_ns, b_d,
                 b_ti, b_idx, b_iw) = memory.sample(BATCH_SIZE)

                b_s   = torch.FloatTensor(b_s)
                b_a   = torch.FloatTensor(b_a)
                b_r   = torch.FloatTensor(b_r).unsqueeze(1)
                b_ns  = torch.FloatTensor(b_ns)
                b_d   = torch.FloatTensor(b_d).unsqueeze(1)
                b_iw  = torch.FloatTensor(b_iw).unsqueeze(1)   # IS weights

                # ---- Critic update ----
                with torch.no_grad():
                    q_next   = target_critic(b_ns, target_actor(b_ns))
                    q_target = b_r + (1 - b_d) * GAMMA * q_next

                q_pred      = critic(b_s, b_a)
                td_errors   = (q_pred - q_target).detach().cpu().numpy().squeeze()
                critic_loss = (b_iw * F.mse_loss(q_pred, q_target,
                                                  reduction='none')).mean()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optimizer.step()

                # Aggiorna priorità PER
                memory.update_priorities(b_idx, td_errors)

                # ---- Actor update (solo campioni nella zona Cork) ----
                cork_mask = torch.FloatTensor(
                    [1.0 if CORK_HARD_START < t < CORK_HARD_END else 0.0
                     for t in b_ti]
                ).unsqueeze(1)

                if cork_mask.sum() > 0:
                    pred_a    = actor(b_s)
                    rl_loss   = -critic(b_s, pred_a)

                    with torch.no_grad():
                        anch_a = anchor(b_s)
                    imit_loss = F.mse_loss(pred_a, anch_a,
                                           reduction='none').mean(dim=1,
                                                                   keepdim=True)

                    # FIX: blend RL/imitation con peso variabile
                    total_loss = (
                        cork_mask * (
                            (1.0 - IMITATION_WEIGHT) * rl_loss
                            + IMITATION_WEIGHT * imit_loss
                        )
                    ).sum() / cork_mask.sum()

                    actor_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    actor_optimizer.step()

                soft_update(target_actor,  actor,  TAU)
                soft_update(target_critic, critic, TAU)

        # ---- Decay del rumore ----
        noise_sigma = max(NOISE_SIGMA_MIN, noise_sigma * NOISE_DECAY)

        # ---- Scheduler RL ----
        if episode >= ACTOR_WARMUP_EPISODES:
            actor_scheduler.step(-episode_reward)
            critic_scheduler.step(-episode_reward)

        # ---- Salvataggio best model ----
        lap_time  = obs.get('lastLapTime', 0.0) if obs else 0.0
        if dist_raced > best_dist_raced:
            best_dist_raced = dist_raced
            torch.save(actor.state_dict(), "actor_CASO2_best_dist.pth")
            print(f"  ★ New best distance: {dist_raced:.1f}m")

        if 0 < lap_time < best_lap_time:
            best_lap_time = lap_time
            torch.save(actor.state_dict(), "actor_CASO2_best_laptime.pth")
            print(f"  ★ New best lap time: {lap_time:.2f}s")

        # Checkpoint periodico
        if episode % 50 == 0:
            torch.save(actor.state_dict(),
                       f"actor_CASO2_ep{episode}.pth")

        lr_now = actor_optimizer.param_groups[0]['lr']
        log_writer.writerow([
            episode, steps, f"{episode_reward:.2f}", f"{lap_time:.2f}",
            f"{dist_raced:.1f}", f"{max_dist_from_start:.1f}",
            int(cork_crash_this_ep), f"{noise_sigma:.4f}", f"{lr_now:.2e}"
        ])
        log_file.flush()

        print(f"Ep {episode:4d} | "
              f"dist={dist_raced:6.1f}m "
              f"maxPos={max_dist_from_start:6.1f}m "
              f"R={episode_reward:8.1f} "
              f"σ={noise_sigma:.3f} "
              f"lr={lr_now:.1e}")

    log_file.close()
    env.end()
    print("\nTraining completato.")
    print(f"Best distance: {best_dist_raced:.1f}m")
    print(f"Best lap time: {best_lap_time:.2f}s")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CASO 2 — RL Fine-tuning Corkscrew TORCS"
    )
    parser.add_argument(
        "--base", type=str, default=None,
        help="Percorso al modello base .pth "
             "(default: cerca actor_IL_CASO1.pth o actor_GOLDEN_STABLE.pth)"
    )
    args = parser.parse_args()
    reinforce(base_model_path=args.base)