import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- Configurazione ---
INPUT_SIZE = 30  # angle(1) + gear(1) + rpm(1) + speedX,Y,Z(3) + track(19) + trackPos(1) + wheelSpinVel(4)
OUTPUT_SIZE = 4 # steer, accel, brake, gear
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001

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
        gear = torch.sigmoid(self.gear_head(features)) # Scalato 0-1 per il training
        return torch.cat([steer, accel, brake, gear], dim=-1)

def train():
    # 1. Caricamento dati
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "manualtot.csv")
    print(f"Loading dataset from: {dataset_path}")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Errore: {dataset_path} non trovato.")
        return

    # Separazione feature e target
    y = data.iloc[:, 1:5].values.astype(np.float32) 
    X = data.iloc[:, 5:].values.astype(np.float32) 

    # Normalizzazione Target: Gear da [1, 6] a [0, 1]
    y[:, 3] = (y[:, 3] - 1.0) / 5.0

    input_dim = X.shape[1]
    print(f"Dimensioni rilevate: Input={input_dim}, Target={y.shape[1]}")

    # Normalizzazione Feature (Invariata)
    X[:, 0] /= 3.14159  # angle
    X[:, 1] /= 6.0      # gear
    X[:, 2] /= 10000.0  # rpm
    X[:, 3] /= 200.0    # speedX
    X[:, 4] /= 200.0    # speedY
    X[:, 5] /= 200.0    # speedZ
    X[:, 6:25] /= 200.0 # track sensors
    X[:, 25] /= 3.0     # trackPos (Allineato a reinforce_optimization.py)
    X[:, 26:30] /= 100.0 # wheelSpinVel

    # Conversione in Tensor
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Setup Modello
    model = ExpertModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop con Weighted Loss per la Frenata
    print(f"Inizio addestramento su {len(X)} campioni (Brake-Weighted)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calcolo pesi: diamo più importanza ai campioni dove si frena (target_brake > 0)
            # batch_y[:, 2] è la colonna del freno
            weights = torch.ones_like(batch_y)
            weights[:, 2] += (batch_y[:, 2] > 0.1).float() * 5.0 # Peso 6x sulla frenata
            weights[:, 0] += (torch.abs(batch_y[:, 0]) > 0.1).float() * 2.0 # Peso 3x sulla sterzata
            
            loss = torch.mean(weights * (outputs - batch_y)**2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.6f}")

    # 4. Salvataggio
    torch.save(model.state_dict(), "expert_model.pth")
    print("Modello salvato con successo: expert_model.pth")

if __name__ == "__main__":
    train()
