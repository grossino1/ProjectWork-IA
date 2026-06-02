import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- Configurazione ULTRA VELOCE per Patching ---
INPUT_SIZE = 30
OUTPUT_SIZE = 4
BATCH_SIZE = 128
EPOCHS = 10 # Ridotto per non far scadere il tempo e fare solo un "refresh"
LEARNING_RATE = 0.0005

class ExpertModel(nn.Module):
    def __init__(self, input_size):
        super(ExpertModel, self).__init__()
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

    def forward(self, x):
        features = self.base(x)
        steer = torch.tanh(self.steer_head(features))
        accel = torch.sigmoid(self.accel_head(features))
        brake = torch.sigmoid(self.brake_head(features))
        gear = torch.sigmoid(self.gear_head(features))
        return torch.cat([steer, accel, brake, gear], dim=-1)

def train():
    dataset_path = "manualtot.csv"
    print(f"Loading dataset: {dataset_path}")
    data = pd.read_csv(dataset_path)
    
    # Pulizia Elite come discusso
    data = data.dropna()
    data = data[data['trackPos'].abs() <= 1.0]
    
    # Preprocessing (stessa logica del reinforce)
    states = []
    actions = []
    
    for _, row in data.iterrows():
        # Stato
        angle = row['angle'] / 3.14159
        gear = row['target_gear'] / 6.0
        rpm = row['rpm'] / 10000.0
        speed = [row['speedX']/200.0, row['speedY']/200.0, row['speedZ']/200.0]
        track = [row[f'track_{i}']/200.0 for i in range(19)]
        track_pos = row['trackPos'] / 3.0
        wheel_spin = [row[f'wheelSpinVel_{i}']/100.0 for i in range(4)]
        
        state = [angle, gear, rpm] + speed + track + [track_pos] + wheel_spin
        states.append(state)
        
        # Azioni
        action = [row['target_steer'], row['target_accel'], row['target_brake'], row['target_gear']/6.0]
        actions.append(action)

    X = torch.FloatTensor(np.array(states))
    y = torch.FloatTensor(np.array(actions))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ExpertModel(INPUT_SIZE)
    
    # CARICHIAMO IL MODELLO GOLDEN PER NON PARTIRE DA ZERO (EVITA REGRESSIONE)
    if os.path.exists("actor_GOLDEN_STABLE.pth"):
        print("Pre-loading actor_GOLDEN_STABLE.pth to avoid regression...")
        model.load_state_dict(torch.load("actor_GOLDEN_STABLE.pth"))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"Starting Fast Patch Training ({EPOCHS} epochs)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Peso maggiorato sulla frenata e sullo sterzo per i settori critici
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "actor_GOLDEN_STABLE_PATCHED.pth")
    print("Modello salvato: actor_GOLDEN_STABLE_PATCHED.pth")

if __name__ == "__main__":
    train()
