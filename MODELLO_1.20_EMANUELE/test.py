"""=============================================================================
TEST — Golden Stable + Caso2 al Corkscrew

=============================================================================

LOGICA:
  - Fuori dalla zona Corkscrew (< 2300m e > 2850m)  → 100% GOLDEN STABLE
  - Zona di transizione (2300-2400m e 2750-2850m)    → blend lineare
  - Dentro il Corkscrew (2400-2750m)                 → 100% CASO2 BEST

RESET MANUALE
  -"Continue" -> "New Practice" in TORCS -> INVIO nel prompt

=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import os

from gym_torcs import TorcsEnv

# ---------------------------------------------------------------------------
# COSTANTI (identiche al training Caso2)
# ---------------------------------------------------------------------------
INPUT_SIZE  = 30
OUTPUT_SIZE = 4

CORK_HARD_START = 2400.0
CORK_HARD_END   = 2750.0
CORK_BLEND_ZONE = 100.0     # rampa di transizione in metri
CRITICAL_POINT  = 2477.0    # cambio di pendenza

# ---------------------------------------------------------------------------
# ARCHITETTURA (identica al training — NON modificare)
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


# ---------------------------------------------------------------------------
# PREPROCESSING STATO (identico al training)
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
# BLEND ACTOR/ANCHOR (identico al training Caso2)
# ---------------------------------------------------------------------------
def compute_blend_factor(track_idx: float) -> float:
    """
    Restituisce α ∈ [0, 1]:
      α = 0  → 100% golden stable (anchor)
      α = 1  → 100% cork actor (caso2)
    """
    if track_idx < CORK_HARD_START - CORK_BLEND_ZONE:
        return 0.0
    if track_idx > CORK_HARD_END + 200:
        return 0.0
    if track_idx < CORK_HARD_START:
        return (track_idx - (CORK_HARD_START - CORK_BLEND_ZONE)) / CORK_BLEND_ZONE
    if track_idx <= CORK_HARD_END:
        return 1.0
    return 1.0 - (track_idx - CORK_HARD_END) / CORK_BLEND_ZONE


# ---------------------------------------------------------------------------
# CARICAMENTO MODELLI con verifica
# ---------------------------------------------------------------------------
def load_model(path: str, label: str) -> Actor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modello '{label}' non trovato: {path}\n"
            f"Assicurati che il file esista nella directory corrente."
        )
    model = Actor(INPUT_SIZE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"  [OK] {label}: {path}")
    return model


# ---------------------------------------------------------------------------
# TEST PRINCIPALE
# ---------------------------------------------------------------------------
def test_hybrid(anchor_path: str, cork_path: str, n_episodes: int):
    print("\n" + "=" * 65)
    print("TEST IBRIDO: Golden Stable + Caso2 al Corkscrew")
    print("(RESET MANUALE DA TORCS)")
    print("=" * 65)
    print(f"  Modello FUORI Cork: {anchor_path}")
    print(f"  Modello DENTRO Cork: {cork_path}")
    print(f"  Zona blend:  {CORK_HARD_START - CORK_BLEND_ZONE:.0f}m "
          f"→ {CORK_HARD_END + CORK_BLEND_ZONE:.0f}m")
    print(f"  Episodi: {n_episodes}")
    print("=" * 65)

    # ---- Caricamento ----
    print("\nCaricamento modelli...")
    anchor     = load_model(anchor_path, "Golden Stable (anchor)")
    cork_actor = load_model(cork_path,   "Cork Actor (caso2)")

    env = TorcsEnv(vision=False, throttle=True, gear_change=True)

    results = []

    for episode in range(1, n_episodes + 1):
        print(f"\n{'─' * 65}")
        print(f"EPISODIO {episode}/{n_episodes}")
        print(f"{'─' * 65}")

        # ★ RESET MANUALE: Aspetta istruzioni dall'utente ★
        if episode > 1:
            print("\n EPISODIO TERMINATO")
            print("\n COSA FARE:")
            print("  'Continue'->'New Practice'->INVIO")
            print("\n" + "─" * 65)
            input("▶ Premi INVIO quando sei pronto in TORCS >>> ")
            print("▶ Connessione in corso...\n")

        # ---- Connessione TORCS (con retry) ----
        connected = False
        attempt = 0
        while not connected:
            attempt += 1
            try:
                # Primo episodio: relaunch=True
                # Episodi successivi: relaunch=False 
                env.reset(relaunch=(episode == 1))
                connected = True
                print(f"  ✓ Connesso a TORCS (tentativo {attempt})")
            except Exception as e:
                print(f"  ✗ Errore connessione: {e}")
                print(f"  ⏳ Riprovo tra 3 secondi...")
                print(f"     (Verifica che TORCS sia in 'New Practice')")
                time.sleep(3.0)

        obs              = env.client.S.d
        state            = preprocess_state(obs)
        start_dist_raced = obs.get('distRaced', 0.0)
        done             = False
        steps            = 0
        max_dist         = 0.0
        dist_raced       = 0.0
        lap_completed    = False
        crash_point      = None

        print(f"  ✓ Episodio {episode} avviato\n")

        # ---- Esecuzione episodio ----
        while not done:
            state_t   = torch.FloatTensor(state).unsqueeze(0)
            track_idx = float(obs.get('distFromStart', 0.0))
            alpha     = compute_blend_factor(track_idx)

            max_dist = max(max_dist, track_idx)

            with torch.no_grad():
                anchor_action = anchor(state_t).numpy()[0]
                cork_action   = cork_actor(state_t).numpy()[0]

            # Blend deterministico (solo test)
            blended = (1.0 - alpha) * anchor_action + alpha * cork_action

            # Costruzione azione per l'ambiente
            env_action    = blended.copy()
            env_action[0] = np.clip(blended[0], -1.0,  1.0)   # steer
            env_action[1] = np.clip(blended[1],  0.0,  1.0)   # accel
            env_action[2] = np.clip(blended[2],  0.0,  1.0)   # brake
            gear          = int(round(np.clip(blended[3], 0.0, 1.0) * 5.0 + 1.0))
            env_action[3] = float(max(1, min(6, gear)))

            # ── Step nell'ambiente ──────────────────────────────────────────
            try:
                _, _, env_done, _ = env.step(env_action)
                if env_done:
                    done = True
            except Exception as e:
                print(f"  [TORCS] Errore step: {e}")
                done = True
                break

            obs = env.client.S.d
            if not obs:
                break

            dist_raced = obs.get('distRaced', 0.0) - start_dist_raced
            track_pos  = obs.get('trackPos', 0.0)

            # ── Condizioni di terminazione ──────────────────────────────────
            if dist_raced > 3610:
                print(f"\n  ✓ GIRO COMPLETATO!")
                lap_completed = True
                done = True

            elif abs(track_pos) > 2.1:
                crash_point = track_idx
                print(f"\n  ✗ SCHIANTO al metro {track_idx:.1f}m "
                      f"(trackPos={track_pos:.2f})")
                done = True

            elif steps > 12000:
                print(f"\n  ✗ TIMEOUT ({steps} step senza completare il giro)")
                done = True

            state  = preprocess_state(obs)
            steps += 1

        # ── Fine episodio ───────────────────────────────────────────────────
        results.append({
            'episode':   episode,
            'max_dist':  max_dist,
            'dist_raced': dist_raced,
            'completed': lap_completed,
            'crash':     crash_point,
            'steps':     steps,
        })

        summary = "✓ COMPLETO" if lap_completed else f"✗ crash a {crash_point:.1f}m" if crash_point else "✗ timeout"
        print(f"\n  Ep {episode}: {summary} | max_dist={max_dist:.1f}m | "
              f"steps={steps}")

    env.end()

    # ── Riepilogo finale ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RIEPILOGO FINALE")
    print("=" * 65)

    for r in results:
        if r['completed']:
            status = "✓ COMPLETO"
        elif r['crash']:
            status = f"✗ crash a {r['crash']:.1f}m"
        else:
            status = f"✗ timeout a {r['max_dist']:.1f}m"
        print(f"  Ep {r['episode']:2d}: {status}")

    completed = [r for r in results if r['completed']]
    crashes_before_cork = [r for r in results
                           if r['crash'] and r['crash'] < CORK_HARD_START]
    crashes_at_cork     = [r for r in results
                           if r['crash'] and CORK_HARD_START <= r['crash'] <= CORK_HARD_END]

    print(f"\nGiri completati:      {len(completed)}/{len(results)}")
    print(f"Crash prima del Cork: {len(crashes_before_cork)}/{len(results)}")
    print(f"Crash al Cork:        {len(crashes_at_cork)}/{len(results)}")

    avg_max = np.mean([r['max_dist'] for r in results])
    print(f"Distanza media max:   {avg_max:.1f}m")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ibrido: Golden Stable + Caso2 al Corkscrew"
    )
    parser.add_argument(
        "--anchor", type=str, default="actor_GOLDEN_STABLE.pth",
        help="Modello base usato fuori dal Corkscrew (default: actor_GOLDEN_STABLE.pth)"
    )
    parser.add_argument(
        "--cork", type=str, default="actor_CASO2_best_dist.pth",
        help="Modello usato nella zona Corkscrew (default: actor_CASO2_best_dist.pth)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Numero di episodi da eseguire (default: 10)"
    )
    args = parser.parse_args()
    test_hybrid(args.anchor, args.cork, args.episodes)