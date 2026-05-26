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
import time
from collections import deque
from gym_torcs import TorcsEnv
from train_imitation import ExpertModel, INPUT_SIZE, OUTPUT_SIZE

# --- HYPERPARAMETERS - SAFE RECOVERY (v33 - Sync Mode) ---
INPUT_SIZE = 30 
OUTPUT_SIZE = 4 
GAMMA = 0.99
TAU = 0.005 
LR_ACTOR = 1e-6    # Learning rate molto basso per non rovinare la stabilità subito
LR_CRITIC = 1e-4   
BATCH_SIZE = 128 
BUFFER_SIZE = 100000 
IMITATION_WEIGHT_CORK = 0.95 
ACTOR_WARMUP_EPISODES = 2 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done, track_idx):
        self.buffer.append((state, action, reward, next_state, done, track_idx))
    def sample(self, batch_size):
        cork_samples = [s for s in self.buffer if 2400 < s[5] < 2700]
        other_samples = [s for s in self.buffer if not (2400 < s[5] < 2700)]
        batch = []
        if len(cork_samples) > batch_size * 0.6:
            batch.extend(random.sample(cork_samples, int(batch_size * 0.6)))
            batch.extend(random.sample(other_samples, batch_size - int(batch_size * 0.6)))
        elif len(cork_samples) > 0:
            batch.extend(cork_samples)
            batch.extend(random.sample(other_samples, batch_size - len(cork_samples)))
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, track_idx = zip(*batch)
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done), np.stack(track_idx)
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super(Actor, self).__init__()
        # Architettura originale 128 compatibile con GOLDEN_STABLE
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
    
    # FORZIAMO L'USO DEL GOLDEN STABLE (L'unico di cui siamo certi al 100%)
    model_path = "actor_GOLDEN_STABLE.pth"
    if os.path.exists(model_path):
        actor.load_state_dict(torch.load(model_path))
        print(f">>> RECOVERY: Caricato {model_path} per garantire stabilità.")
    else:
        print("ERRORE: actor_GOLDEN_STABLE.pth non trovato!")
        return

    expert_anchor = Actor(INPUT_SIZE)
    expert_anchor.load_state_dict(torch.load(model_path))
    
    target_actor = Actor(INPUT_SIZE)
    target_actor.load_state_dict(actor.state_dict())
    critic = Critic(INPUT_SIZE)
    target_critic = Critic(INPUT_SIZE)
    target_critic.load_state_dict(critic.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    expert_anchor.eval() 
    actor.train() 
    memory = ReplayBuffer(BUFFER_SIZE)
    
    log_file = open("rl_optimization_results.csv", "a", newline='')
    log_writer = csv.writer(log_file)
    
    for episode in range(1000):
        relaunch = False 
        connected = False
        while not connected:
            try:
                env.reset(relaunch=relaunch)
                connected = True
            except:
                relaunch = True
                time.sleep(5.0)

        obs = env.client.S.d 
        state = preprocess_state(obs)
        episode_reward = 0
        done = False
        steps = 0
        start_dist_raced = obs.get('distRaced', 0)

        print(f"\n>>> RECOVERY EPISODIO {episode}")
        actor.eval()

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            track_index = obs.get('distFromStart', 0)
            
            # ZONA ACTOR BLINDATA: 2400-2700 (Solo Corkscrew)
            is_actor_zone = (2400 < track_index < 2700)

            with torch.no_grad():
                if is_actor_zone:
                    action_t = actor(state_t)
                else:
                    action_t = expert_anchor(state_t)
            
            raw_action = action_t.numpy()[0]
            
            if is_actor_zone and episode >= ACTOR_WARMUP_EPISODES:
                noise = np.random.normal(0, 0.03, size=4)
                raw_action = np.clip(raw_action + noise, -1.0, 1.0)
            
            env_action = raw_action.copy()
            env_action[3] = int(round(env_action[3] * 5.0 + 1.0))

            try:
                _, _, env_done, _ = env.step(env_action)
                if env_done: done = True
            except:
                done = True
                break

            obs = env.client.S.d
            if not obs: break

            speed = obs.get('speedX', 0.0)
            angle = obs.get('angle', 0.0)
            track_pos = obs.get('trackPos', 0.0)
            current_dist_raced = obs.get('distRaced', 0) - start_dist_raced

            # Reward Pulita (Progresso puro nel Corkscrew)
            custom_reward = 0.0
            if 2400 < track_index < 2700:
                progress = speed * np.cos(angle)
                custom_reward = progress - (abs(speed * np.sin(angle)) * 0.5)

            if current_dist_raced > 3610:
                print(f"--- GIRO COMPLETATO! ---")
                done = True
            
            if abs(track_pos) > 2.1 or steps > 10000:
                done = True

            episode_reward += custom_reward
            memory.push(state, raw_action, custom_reward, preprocess_state(obs), done, track_index)
            state = preprocess_state(obs)
            steps += 1

        if len(memory) > BATCH_SIZE and episode >= ACTOR_WARMUP_EPISODES:
            updates = 50 
            actor.train()
            for _ in range(updates):
                b_state, b_action, b_reward, b_next_state, b_done, b_track_idx = memory.sample(batch_size=BATCH_SIZE)
                b_state, b_action, b_reward = torch.FloatTensor(b_state), torch.FloatTensor(b_action), torch.FloatTensor(b_reward).unsqueeze(1)
                b_next_state, b_done = torch.FloatTensor(b_next_state), torch.FloatTensor(b_done).unsqueeze(1)

                with torch.no_grad():
                    q_next = target_critic(b_next_state, target_actor(b_next_state))
                    q_target = b_reward + (1 - b_done) * GAMMA * q_next
                
                critic_loss = F.mse_loss(critic(b_state, b_action), q_target)
                critic_optimizer.zero_grad(); critic_loss.backward(); critic_optimizer.step()

                pred_a = actor(b_state)
                rl_losses = -critic(b_state, pred_a) * 0.01 
                with torch.no_grad(): exp_a = expert_anchor(b_state)
                imit_losses = F.mse_loss(pred_a, exp_a, reduction='none').mean(dim=1, keepdim=True)
                
                actor_mask = torch.FloatTensor([1.0 if 2400 < t < 2700 else 0.0 for t in b_track_idx]).to(pred_a.device).unsqueeze(1)
                b_weights = torch.FloatTensor([IMITATION_WEIGHT_CORK if 2400 < t < 2700 else 1.0 for t in b_track_idx]).to(pred_a.device).unsqueeze(1)
                
                if actor_mask.sum() > 0:
                    total_loss = (actor_mask * ((1.0 - b_weights) * rl_losses + b_weights * imit_losses)).sum() / actor_mask.sum()
                    actor_optimizer.zero_grad(); total_loss.backward(); actor_optimizer.step()
                    update_targets(target_actor, actor, TAU)
                update_targets(target_critic, critic, TAU)

        log_writer.writerow([episode, steps, episode_reward, obs.get('lastLapTime', 0), current_dist_raced])
        log_file.flush()
        print(f"Ep {episode} | Dist: {current_dist_raced:.1f}m")
    
    log_file.close()
    env.end()

def preprocess_state(S):
    angle = np.array([S.get('angle', 0)], dtype=np.float32) / 3.14159
    gear = np.array([S.get('gear', 1)], dtype=np.float32) / 6.0
    rpm = np.array([S.get('rpm', 0)], dtype=np.float32) / 10000.0
    speed = np.array([S.get('speedX', 0), S.get('speedY', 0), S.get('speedZ', 0)], dtype=np.float32) / 200.0
    track = np.array(S.get('track', [0]*19), dtype=np.float32) / 200.0
    track_pos = np.array([S.get('trackPos', 0)], dtype=np.float32) / 3.0
    wheel_spin = np.array(S.get('wheelSpinVel', [0]*4), dtype=np.float32) / 100.0
    return np.concatenate([angle, gear, rpm, speed, track, track_pos, wheel_spin])

if __name__ == "__main__":
    reinforce()
