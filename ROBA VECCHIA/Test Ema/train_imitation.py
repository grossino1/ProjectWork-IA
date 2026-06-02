import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- Configurazione Avanzata (v2 - High Fidelity) ---
INPUT_SIZE = 30  
OUTPUT_SIZE = 4 
BATCH_SIZE = 64
EPOCHS = 200 # Aumentato per una convergenza più profonda
LEARNING_RATE = 0.0005 # Ridotto per una maggiore precisione chirurgica

class ExpertModel(nn.Module):
    def __init__(self, input_size):
        super(ExpertModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 256), # Aumentata capacità
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "manualtot.csv")
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Errore: {e}")
        return

    # y: steer, accel, brake, gear
    y = data.iloc[:, 1:5].values.astype(np.float32) 
    # X: angle, gear, rpm, speedX,Y,Z, track0-18, trackPos, wheelSpin0-3
    X = data.iloc[:, 5:].values.astype(np.float32) 

    # Normalizzazione Rigorosa (Sincronizzata con reinforce_optimization.py)
    y[:, 3] = (y[:, 3] - 1.0) / 5.0 # Gear target 1-6 -> 0-1

    X[:, 0] /= 3.14159  # angle
    X[:, 1] /= 6.0      # gear
    X[:, 2] /= 10000.0  # rpm
    X[:, 3] /= 200.0    # speedX
    X[:, 4] /= 200.0    # speedY
    X[:, 5] /= 200.0    # speedZ
    X[:, 6:25] /= 200.0 # track sensors
    X[:, 25] /= 3.0     # trackPos
    X[:, 26:30] /= 100.0 # wheelSpinVel

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ExpertModel(INPUT_SIZE)
    # Torniamo a MSE standard per la massima fluidità di guida
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Inizio addestramento High-Fidelity su {len(X)} campioni...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.6f}")

    # Salvataggio con il nome richiesto
    save_path = os.path.join(base_path, "Test20-05_imitation114.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modello High-Fidelity salvato: {save_path}")

if __name__ == "__main__":
    train()
