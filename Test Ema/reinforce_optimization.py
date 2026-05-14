import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
import csv
import os
import sys
from collections import deque
from gym_torcs import TorcsEnv
from train_imitation import ExpertModel, INPUT_SIZE, OUTPUT_SIZE

# --- HYPERPARAMETERS - FIDELITY FOCUS (v20) ---
INPUT_SIZE = 30 
OUTPUT_SIZE = 4 
GAMMA = 0.99
TAU = 0.005 
LR_ACTOR = 5e-7   # Ridotto ancora per preservare la traiettoria da 1:13
LR_CRITIC = 5e-6  
BATCH_SIZE = 128 
BUFFER_SIZE = 100000 
TRAIN_EVERY = 1   
GRADIENT_CLIP = 0.5 
IMITATION_WEIGHT_START = 1.0 # 100% Imitazione per i primi episodi
IMITATION_WEIGHT_END = 0.95  
IMITATION_DECAY_EPISODES = 200 
ACTOR_WARMUP_EPISODES = 10    # Nessun training RL per i primi 10 ep, solo raccolta dati

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super(Actor, self).__init__()
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
        self.gear_head = nn.Linear(64, 1)

    def forward(self, state):
        features = self.base(state)
        steer = torch.tanh(self.steer_head(features))
        accel = torch.sigmoid(self.accel_head(features))
        brake = torch.sigmoid(self.brake_head(features))
        gear = torch.sigmoid(self.gear_head(features)) 
        return torch.cat([steer, accel, brake, gear], dim=-1)

class Critic(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size + OUTPUT_SIZE, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

def update_targets(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def reinforce():
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    actor = Actor(INPUT_SIZE)
    
    if os.path.exists("optimized_actor.pth"):
        actor.load_state_dict(torch.load("optimized_actor.pth"))
        print(">>> Ripristinato modello OTTIMIZZATO")
    else:
        try:
            actor.load_state_dict(torch.load("expert_model.pth"))
            print(">>> Pesi iniziali caricati da expert_model.pth (Traiettoria 1:13)")
        except:
            print("ERRORE: expert_model.pth mancante.")
            return

    target_actor = Actor(INPUT_SIZE)
    target_actor.load_state_dict(actor.state_dict())
    critic = Critic(INPUT_SIZE)
    target_critic = Critic(INPUT_SIZE)
    target_critic.load_state_dict(critic.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    # --- ANCHORING FIX ---
    # Usiamo SEMPRE expert_model.pth come ancora di verità assoluta
    # Invece di usare il modello ottimizzato corrente che potrebbe aver "imparato" vizi
    expert_anchor = Actor(INPUT_SIZE)
    try:
        expert_anchor.load_state_dict(torch.load("expert_model.pth"))
        print(">>> Expert Anchor fissato su expert_model.pth (Truth Reference)")
    except:
        expert_anchor.load_state_dict(actor.state_dict())
        print(">>> ATTENZIONE: expert_model.pth non trovato, uso il modello corrente come ancora")
    expert_anchor.eval() 
    
    memory = ReplayBuffer(BUFFER_SIZE)
    
    log_exists = os.path.exists("rl_optimization_results.csv")
    log_file = open("rl_optimization_results.csv", "a", newline='')
    log_writer = csv.writer(log_file)
    
    best_dist = 2483.9 # FORZATO per proteggere il record
    if not log_exists or os.path.getsize("rl_optimization_results.csv") == 0:
        log_writer.writerow(["episode", "steps", "total_reward", "last_lap", "max_dist"])
    else:
        # Leggi la migliore distanza storica per evitare di sovrascrivere actor_best con robaccia
        try:
            with open("rl_optimization_results.csv", "r") as f:
                reader = csv.DictReader(f)
                dists = [float(row['max_dist']) for row in reader if row['max_dist']]
                if dists: best_dist = max(max(dists), best_dist)
                print(f">>> Record storico caricato: {best_dist:.1f}m")
        except:
            pass

    for episode in range(1000):
        # --- ROBUST AUTONOMOUS RESET ---
        # Rilancio TORCS ogni 20 episodi per pulire la memoria e prevenire freeze
        relaunch = (episode % 20 == 0)
        try:
            env.reset(relaunch=relaunch)
        except Exception as e:
            print(f"!!! Errore critico durante il reset (Ep {episode}): {e}. Tento Hard Relaunch...")
            env.reset(relaunch=True)

        obs = env.client.S.d 
        state = preprocess_state(obs)
        episode_reward = 0
        done = False
        steps = 0
        prev_damage = obs.get('damage', 0)
        max_dist_reached = 0.0
        start_dist_raced = obs.get('distRaced', 0)
        prev_lap_time = 0.0

        print(f"\n>>> INIZIO EPISODIO {episode} (Relaunch={relaunch})")

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): action_t = actor(state_t)
            raw_action = action_t.numpy()[0]
            
            # --- CHIRURGIA DEL NOISE (Solo Corkscrew) ---
            dist_raced_absolute = obs.get('distRaced', 0)
            track_index = dist_raced_absolute % 3602
            
            if episode < 2:
                noise_scale = 0.0 # Stabilizzazione assoluta
            elif 2350 < track_index < 2550:
                noise_scale = 0.05 # Alta esplorazione nel punto critico
            else:
                noise_scale = 0.005 # Stabilità nel resto del circuito
            
            env_action = raw_action.copy()
            env_action[0] = np.clip(env_action[0] + np.random.normal(0, noise_scale), -1.0, 1.0)
            
            # Gear scaling 1-6
            env_action[3] = int(round(env_action[3] * 5.0 + 1.0))
            
            # Step
            try:
                _, _, _, _ = env.step(env_action)
            except Exception as e:
                print(f"!!! Timeout/Errore durante lo step: {e}")
                done = True
                break

            obs = env.client.S.d
            speed = obs['speedX']
            angle = obs['angle']
            track_pos = obs['trackPos']
            curr_lap_time = obs.get('curLapTime', 0.0)
            current_dist_raced = obs.get('distRaced', 0) - start_dist_raced
            
            # --- REWARD FUNCTION (Aligned with GEMINI.md) ---
            # 1. Progresso Longitudinale: Vx * cos(theta)
            progress = speed * np.cos(angle)
            
            # 2. Penalità Drift: Vx * sin(theta)
            drift_penalty = abs(speed * np.sin(angle))
            
            # 4. Penalità Fuori Asse: Vx * trackPos
            off_center_penalty = 0.0
            if abs(track_pos) > 0.95:
                off_center_penalty = speed * (abs(track_pos) - 0.95) * 5.0

            # --- CHIRURGIA DEL REWARD (Corkscrew 2400m - 2550m) ---
            vertical_penalty = 0.0
            speed_limit_penalty = 0.0
            
            if 2400 < track_index < 2550:
                # Penalità per caduta libera (speedZ negativa)
                if obs['speedZ'] < -0.5:
                    vertical_penalty = abs(obs['speedZ']) * 50.0
                
                # Forza la frenata: se arrivi sopra gli 80 km/h al drop, punizione
                if track_index < 2460 and speed > 80:
                    speed_limit_penalty = (speed - 80) * 10.0

            # 6. Penalità Danni (Estrema)
            damage_delta = obs['damage'] - prev_damage
            damage_penalty = 0.0
            if damage_delta > 0:
                damage_penalty = -20000.0
                done = True

            custom_reward = progress - (0.5 * drift_penalty) - (0.1 * off_center_penalty) - vertical_penalty - speed_limit_penalty + damage_penalty

            
            # --- CONDIZIONI DI USCITA ---
            if curr_lap_time < prev_lap_time and current_dist_raced > 3000:
                print(f"!!! GIRO COMPLETATO !!! Tempo: {obs.get('lastLapTime', 0):.2f}s")
                torch.save(actor.state_dict(), "actor_lap_complete.pth")
                done = True
            
            if abs(track_pos) > 1.8 or np.cos(angle) < -0.1:
                done = True
            
            if steps > 30000: 
                done = True

            # Memoria e Training
            episode_reward += custom_reward
            memory.push(state, raw_action, custom_reward, preprocess_state(obs), done)
            state = preprocess_state(obs)
            prev_damage = obs['damage']
            prev_lap_time = curr_lap_time
            max_dist_reached = max(max_dist_reached, current_dist_raced)
            steps += 1

            if len(memory) > BATCH_SIZE and episode >= ACTOR_WARMUP_EPISODES:
                b_state, b_action, b_reward, b_next_state, b_done = memory.sample(BATCH_SIZE)
                b_state, b_action, b_reward = torch.FloatTensor(b_state), torch.FloatTensor(b_action), torch.FloatTensor(b_reward).unsqueeze(1)
                b_next_state, b_done = torch.FloatTensor(b_next_state), torch.FloatTensor(b_done).unsqueeze(1)

                with torch.no_grad():
                    q_next = target_critic(b_next_state, target_actor(b_next_state))
                    q_target = b_reward + (1 - b_done) * GAMMA * q_next
                
                critic_loss = F.mse_loss(critic(b_state, b_action), q_target)
                critic_optimizer.zero_grad(); critic_loss.backward(); critic_optimizer.step()

                pred_a = actor(b_state)
                rl_loss = -critic(b_state, pred_a).mean()
                with torch.no_grad(): exp_a = expert_anchor(b_state)
                imit_loss = F.mse_loss(pred_a, exp_a)
                
                w = max(IMITATION_WEIGHT_END, IMITATION_WEIGHT_START - (episode/IMITATION_DECAY_EPISODES))
                total_loss = (1-w)*rl_loss + w*imit_loss
                actor_optimizer.zero_grad(); total_loss.backward(); actor_optimizer.step()
                update_targets(target_actor, actor, TAU)
                update_targets(target_critic, critic, TAU)

        if max_dist_reached > best_dist:
            best_dist = max_dist_reached
            torch.save(actor.state_dict(), "actor_best.pth")
            print(f"--- NUOVO RECORD DISTANZA: {best_dist:.1f}m ---")

        if episode > 0 and episode % 10 == 0:
            version_path = f"brains/actor_ep{episode}.pth"
            torch.save(actor.state_dict(), version_path)
            torch.save(actor.state_dict(), "optimized_actor.pth")
            print(f">>> AUTO-SAVE: {version_path} salvato.")

        log_writer.writerow([episode, steps, episode_reward, obs.get('lastLapTime', 0), max_dist_reached])
        log_file.flush()
        print(f"Fine Ep {episode} | Step: {steps} | Dist: {max_dist_reached:.1f}m | Rew: {episode_reward:.1f}")
    
    log_file.close()
    env.end()

def preprocess_state(S):
    angle = np.array([S['angle']], dtype=np.float32) / 3.14159
    gear = np.array([S['gear']], dtype=np.float32) / 6.0
    rpm = np.array([S['rpm']], dtype=np.float32) / 10000.0
    speed = np.array([S['speedX'], S['speedY'], S['speedZ']], dtype=np.float32) / 200.0
    track = np.array(S['track'], dtype=np.float32) / 200.0
    track_pos = np.array([S['trackPos']], dtype=np.float32) / 3.0
    wheel_spin = np.array(S['wheelSpinVel'], dtype=np.float32) / 100.0
    return np.concatenate([angle, gear, rpm, speed, track, track_pos, wheel_spin])

if __name__ == "__main__":
    reinforce()
