# =============================================================================
# train_imitation.py — FASE 1: Imitation Learning (Behavioral Cloning)
# =============================================================================
# SCOPO DEL FILE:
#   Questo script addestra la rete neurale "Golden Stable" usando il metodo
#   del Behavioral Cloning (BC). Legge il dataset di guida umana (manualtot.csv),
#   e insegna alla rete a replicare i comandi del pilota dati i sensori della pista.
#   L'output è il file "actor_GOLDEN_STABLE.pth" che contiene i pesi della rete
#   addestrata, usato poi da test_hybrid_v2.py e reinforce_optimization.py.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURAZIONE GLOBALE ---
# INPUT_SIZE=30: dimensione del vettore di stato (vedi preprocessing più sotto)
# OUTPUT_SIZE=4: steer, accel, brake, gear
# BATCH_SIZE=64: numero di campioni processati insieme ad ogni iterazione
# EPOCHS=150: numero di passaggi completi sul dataset durante il training
# LEARNING_RATE=0.001: velocità di aggiornamento dei pesi della rete neurale
INPUT_SIZE = 30
OUTPUT_SIZE = 4
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001


# =============================================================================
# ARCHITETTURA DELLA RETE NEURALE — ExpertModel (Multi-Head)
# =============================================================================
# La rete usa un'architettura "Multi-Head" (a teste separate):
#   - Una base comune (Shared Base) che estrae feature dai 30 sensori
#   - 4 rami indipendenti (teste) che calcolano separatamente steer/accel/brake/gear
#
# PERCHÉ MULTI-HEAD?
#   Se steer e brake condividessero lo stesso neurone di uscita, i loro gradienti
#   si disturberebbero a vicenda durante la backpropagation. Le teste separate
#   permettono a ogni output di ottimizzarsi indipendentemente.
#
# STRUTTURA DELLA BASE COMUNE:
#   Linear(30→128) + LayerNorm + ReLU
#   Linear(128→128) + LayerNorm + ReLU
#   Linear(128→64)  + LayerNorm + ReLU
#
# PERCHÉ LayerNorm E NON BatchNorm?
#   LayerNorm normalizza ogni singolo campione indipendentemente, funziona
#   anche con batch piccoli e non richiede statistiche di popolazione.
# =============================================================================
class ExpertModel(nn.Module):
    def __init__(self, input_size):
        super(ExpertModel, self).__init__()

        # Base comune: tre layer lineari densi con normalizzazione e attivazione ReLU
        self.base = nn.Sequential(
            nn.Linear(input_size, 128),  # 30 sensori → 128 neuroni
            nn.LayerNorm(128),           # normalizza per stabilizzare i gradienti
            nn.ReLU(),                   # attivazione non-lineare
            nn.Linear(128, 128),         # 128 → 128 (approfondisce la rappresentazione)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),          # 128 → 64 (comprime le feature più rilevanti)
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Teste separate: ogni testa produce UN solo valore di output
        self.steer_head = nn.Linear(64, 1)  # sterzo
        self.accel_head = nn.Linear(64, 1)  # acceleratore
        self.brake_head = nn.Linear(64, 1)  # freno
        self.gear_head  = nn.Linear(64, 1)  # marcia

    def forward(self, x):
        # Propagazione attraverso la base comune
        features = self.base(x)

        # Ogni testa produce il proprio output con la propria funzione di attivazione:
        # - tanh  per steer: range [-1.0, +1.0] (destra/sinistra)
        # - sigmoid per accel/brake: range [0.0, 1.0] (rilasciato/pieno)
        # - sigmoid per gear: range [0.0, 1.0] (normalizzato, sarà de-normalizzato a runtime)
        steer = torch.tanh(self.steer_head(features))
        accel = torch.sigmoid(self.accel_head(features))
        brake = torch.sigmoid(self.brake_head(features))
        gear  = torch.sigmoid(self.gear_head(features))  # scalato 0-1, poi convertito a marcia intera

        # Concatena i 4 output in un unico vettore [steer, accel, brake, gear]
        return torch.cat([steer, accel, brake, gear], dim=-1)


# =============================================================================
# FUNZIONE DI TRAINING PRINCIPALE
# =============================================================================
def train():

    # -------------------------------------------------------------------------
    # FASE 1: CARICAMENTO E PARSING DEL DATASET
    # -------------------------------------------------------------------------
    # Il dataset manualtot.csv contiene la telemetria registrata durante la guida
    # manuale con controller DS4. Ogni riga è uno "snapshot" dello stato della
    # macchina + i comandi impartiti dal pilota in quel momento.
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "manualtot.csv")
    print(f"Loading dataset from: {dataset_path}")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Errore: {dataset_path} non trovato.")
        return

    # STRUTTURA DEL CSV (colonne):
    #   col 0:      timestamp
    #   col 1-4:    target_steer, target_accel, target_brake, target_gear  ← OUTPUT (Y)
    #   col 5-34:   sensori TORCS (angle, gear, rpm, speedX/Y/Z, track×19,
    #               trackPos, wheelSpinVel×4)                              ← INPUT (X)

    # Separa input (X) e output (Y) in due matrici NumPy a precisione singola
    y = data.iloc[:, 1:5].values.astype(np.float32)   # comandi del pilota
    X = data.iloc[:, 5:].values.astype(np.float32)    # sensori del simulatore

    # NORMALIZZAZIONE TARGET — Marcia da [1,6] a [0,1]:
    # La testa gear usa sigmoid che produce [0,1], quindi la marcia reale
    # va normalizzata nello stesso range durante il training.
    # Formula: gear_norm = (gear_raw - 1) / 5
    # Es: marcia 1 → 0.0, marcia 6 → 1.0
    y[:, 3] = (y[:, 3] - 1.0) / 5.0

    input_dim = X.shape[1]  # sarà 30 se il CSV è corretto
    print(f"Dimensioni rilevate: Input={input_dim}, Target={y.shape[1]}")

    # -------------------------------------------------------------------------
    # FASE 2: NORMALIZZAZIONE DELLE FEATURE DI INPUT
    # -------------------------------------------------------------------------
    # PERCHÉ NORMALIZZARE?
    #   I sensori hanno scale molto diverse: rpm arriva a 10.000, track arriva a 200,
    #   angle è in [-π, +π]. Se passati grezzi alla rete, i sensori con valori grandi
    #   (es. rpm=8000) dominerebbero i gradienti rispetto a sensori piccoli (angle=0.1),
    #   causando oscillazioni nell'addestramento e convergenza lenta o instabile.
    #   La normalizzazione porta tutto in un range omogeneo ~[0,1] o [-1,1].
    X[:, 0]     /= 3.14159   # angle:        [-π, π]     → [-1, 1]
    X[:, 1]     /= 6.0       # gear sensore: [1, 6]      → [0.17, 1.0]
    X[:, 2]     /= 10000.0   # rpm:          [0, 10000]  → [0, 1]
    X[:, 3]     /= 200.0     # speedX:       [0, ~200]   → [0, ~1]
    X[:, 4]     /= 200.0     # speedY:       [-200, 200] → [-1, 1]
    X[:, 5]     /= 200.0     # speedZ:       [-200, 200] → [-1, 1]
    X[:, 6:25]  /= 200.0     # track[0..18]: [0, 200]    → [0, 1]  (19 sensori laser)
    X[:, 25]    /= 3.0       # trackPos:     [-1, 1] già, diviso 3 → [-0.33, 0.33]
    X[:, 26:30] /= 100.0     # wheelSpinVel: [0, ~100]   → [0, ~1] (4 ruote)

    # Conversione in tensori PyTorch e creazione del DataLoader
    # Questa operazione è il "ponte" fondamentale che trasforma i vostri dati grezzi (estratti dal file CSV tramite Pandas e NumPy) 
    # in un formato speciale che PyTorch è in grado di comprendere, manipolare e usare per addestrare la rete neurale.
    # Infatti solo tramiti i Tensori PyTorch è possibile registrare le operazioni per poter calcolare automaticamente la retropropagazione dell'errore.
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset  = TensorDataset(X_tensor, y_tensor)

    # shuffle=True: mescola i campioni ad ogni epoca per evitare che la rete
    # impari l'ordine temporale invece del comportamento di guida
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------------------------------------------------------
    # FASE 3: SETUP MODELLO E OTTIMIZZATORE
    # -------------------------------------------------------------------------
    model     = ExpertModel(input_dim)   # rete neurale Multi-Head
    criterion = nn.MSELoss()             # usato solo come riferimento, vedi sotto
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Con queste tre righe avete materialmente messo in pista la macchina (model), stabilito che l'obiettivo è azzerare lo scarto quadratico rispetto all'uomo 
    # (criterion) e ingaggiato l'algoritmo matematico più efficiente sul mercato per correggere i neuroni a ogni errore commesso (optimizer).

    # -------------------------------------------------------------------------
    # FASE 4: TRAINING LOOP CON WEIGHTED MSE LOSS (LOSS ASIMMETRICA)
    # -------------------------------------------------------------------------
    # PROBLEMA: il dataset è sbilanciato. Il 90% dei campioni ha brake=0 e steer≈0
    # (rettilinei). Se usassimo MSE standard, la rete imparerebbe a stare ferma
    # sui pedali e a sterzare poco, perché così minimizza la loss media.
    # I pochi campioni con frenata forte (staccate) verrebbero ignorati.
    #
    # SOLUZIONE: Weighted MSE — assegna un peso maggiore ai campioni critici.
    # Formula: L = (1/N) * Σ_i Σ_j w_{i,j} * (ŷ_{i,j} - y_{i,j})²
    #
    # Pesi assegnati:
    #   - Frenata (col 2): peso 6x se target_brake > 0.1
    #   - Sterzata (col 0): peso 3x se |target_steer| > 0.1
    #   - Accel/gear: peso 1x (normale)
    print(f"Inizio addestramento su {len(X)} campioni (Brake-Weighted)...")

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_X, batch_y in loader:
            # Ogni volta che la rete studia un batch, calcola degli errori chiamati "gradienti". Prima di iniziare un nuovo batch, 
            # questo comando cancella i gradienti del calcolo precedente. Se non lo facessi, l'ottimizzatore sommerebbe gli errori del 
            # passato a quelli presenti, mandando la matematica della rete totalmente in confusione.
            optimizer.zero_grad()  

            # I sensori del batch corrente (batch_X) entrano nella rete neurale. 
            outputs = model(batch_X)  

            # La rete esegue i calcoli attraverso la Shared Base e le 4 teste, sputando fuori la sua ipotesi di guida (outputs). In pratica, la rete dice: 
            # "Secondo me, con questi sensori, dovrei sterzare di X, accelerare di Y, frenare di Z e mettere la marcia K".

            # La rete crea una matrice di pesi che parte da 1.0 per tutti i comandi. Poi guarda cosa ha fatto l'uomo in quel batch:
            # Se l'uomo stava frenando (target_brake > 0.1), aggiunge $+5.0$ alla colonna del freno, che diventa 6.0 (Peso 6x).
            # Se l'uomo stava sterzando decisamente (|target_steer| > 0.1), aggiunge $+2.0$ alla colonna dello sterzo, che diventa 3.0 (Peso 3x).
            weights = torch.ones_like(batch_y)  # parte tutto a peso 1
            weights[:, 2] += (batch_y[:, 2] > 0.1).float() * 5.0
            weights[:, 0] += (torch.abs(batch_y[:, 0]) > 0.1).float() * 2.0

            # Loss pesata: Viene calcolato l'errore quadratico medio moltiplicato per i pesi appena decisi.
            loss = torch.mean(weights * (outputs - batch_y)**2)

            # Backpropagation: 
            # loss.backward(): È la retropropagazione: L'errore viaggia all'indietro lungo tutta l'architettura Multi-Head fino alla base comune, 
            # calcolando esattamente la "responsabilità" (gradiente) di ogni singolo neurone nell'aver commesso quell'errore.
            # optimizer.step(): L'ottimizzatore Adam prende i gradienti appena calcolati e modifica leggermente i pesi sinaptici dei 128, 128 e 64 neuroni. 
            # Se un neurone ha causato una sbandata o una mancata frenata, i suoi collegamenti vengono indeboliti o corretti.
            # Infine, total_loss += loss.item() accumula il punteggio di errore di tutta l'epoca, permettendovi di stampare a schermo il progresso ogni 10 epoche 
            # e verificare visivamente che la rete stia effettivamente imparando (la loss deve scendere progressivamente verso lo zero).
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Stampa il progresso ogni 10 epoche
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.6f}")

    # -------------------------------------------------------------------------
    # FASE 5: SALVATAGGIO DEL MODELLO
    # -------------------------------------------------------------------------
    # state_dict() salva solo i pesi (non l'architettura).
    # Il file .pth verrà caricato da test_hybrid_v2.py e reinforce_optimization.py.
    # IMPORTANTE: l'architettura ExpertModel deve essere identica in tutti gli script
    # che caricano questo file, altrimenti il load_state_dict() fallisce.
    torch.save(model.state_dict(), "actor_GOLDEN_STABLE.pth")
    print("Modello salvato con successo: actor_GOLDEN_STABLE.pth")


# Punto di ingresso: esegui train() solo se lo script è lanciato direttamente
if __name__ == "__main__":
    train()