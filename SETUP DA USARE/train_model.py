"""
train_model.py
==============
Addestramento supervisionato (Behavioural Cloning) per TORCS.
Versione Ottimizzata: Rete snella, AMP, Esportazione End-to-End JIT.
"""

import os, sys, json, glob, getopt, random, csv, time, pickle
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    print("[ERRORE] PyTorch non trovato. Installa con: pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    print("[ERRORE] scikit-learn non trovato. Installa con: pip install scikit-learn")
    sys.exit(1)

RAW_SCALE = np.array(
    [3.14159, 300.0, 100.0, 1.0, 10000.0] + [200.0] * 19,
    dtype=np.float32,
)

N_RAW_FEATURES = 24


def _iter_records(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "steps" in data:
            for step in data["steps"]:
                sensors = step.get("state", step.get("sensors", {}))
                actions = step.get("actions", step.get("action", {}))
                if isinstance(sensors, dict) and isinstance(actions, dict):
                    yield sensors, actions
            return
    except (json.JSONDecodeError, UnicodeDecodeError, MemoryError):
        pass

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            if not isinstance(rec, dict): continue
            sensors = rec.get("sensors", {})
            actions = rec.get("actions", {})
            if isinstance(sensors, dict) and isinstance(actions, dict):
                yield sensors, actions

def load_logs(logs_dir: str, min_speed: float = 5.0):
    files = sorted(glob.glob(os.path.join(logs_dir, "*.jsonl")) + glob.glob(os.path.join(logs_dir, "*.json")))
    if not files:
        print(f"[ERRORE] Nessun log in '{logs_dir}'")
        sys.exit(1)

    X_list, y_list, skipped, total = [], [], 0, 0

    for filepath in files:
        for sensors, actions in _iter_records(filepath):
            speed = sensors.get("speedX", 0.0)
            if abs(speed) < min_speed:
                skipped += 1
                continue

            track = sensors.get("track", [])
            if len(track) != 19:
                skipped += 1
                continue

            x_raw = np.array([
                sensors.get("angle", 0.0), sensors.get("speedX", 0.0),
                sensors.get("speedY", 0.0), sensors.get("trackPos", 0.0),
                sensors.get("rpm", 0.0),
            ] + [float(v) for v in track], dtype=np.float32)

            x_pre = np.clip(x_raw / RAW_SCALE, -3.0, 3.0)
            gear_norm = float(actions.get("gear", 1)) / 6.0
            y_raw = np.array([
                float(actions.get("steer", 0.0)), float(actions.get("accel", 0.0)),
                float(actions.get("brake", 0.0)), gear_norm,
            ], dtype=np.float32)

            X_list.append(x_pre)
            y_list.append(y_raw)
            total += 1

    print(f"[DATI] Step validi: {total:,} | Scartati: {skipped:,}")
    if total == 0: sys.exit(1)
    return np.stack(X_list), np.stack(y_list)

def fit_scaler_pca(X_raw: np.ndarray, pca_components, model_dir: str):
    print("\n[PRE-PROC] Fitting StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"[PRE-PROC] Fitting PCA ({pca_components}) ...")
    pca = PCA(n_components=pca_components, random_state=0)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "pca.pkl"), "wb") as f: pickle.dump(pca, f)
    
    return X_pca, pca.n_components_, scaler, pca

class TorcsDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class TorcsDriverNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            ) for _ in range(1)
        ])
        self.res_act = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )
        self.head_steer = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.head_accel = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_brake = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.head_gear  = nn.Sequential(nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.encoder(x)
        for block in self.res_blocks: h = self.res_act(h + block(h))
        h = self.bottleneck(h)
        return torch.cat([self.head_steer(h), self.head_accel(h), self.head_brake(h), self.head_gear(h)], dim=1)

class TorcsEndToEndNet(nn.Module):
    def __init__(self, net, scaler, pca):
        super().__init__()
        self.net = net
        self.register_buffer("raw_scale", torch.tensor(RAW_SCALE, dtype=torch.float32))
        self.register_buffer("scaler_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("pca_comps", torch.tensor(pca.components_.T, dtype=torch.float32))

    def forward(self, x_raw):
        x = torch.clamp(x_raw / self.raw_scale, -3.0, 3.0)
        x = (x - self.scaler_mean) / self.scaler_scale
        x = torch.matmul(x - self.pca_mean, self.pca_comps)
        return self.net(x)

class WeightedMSELoss(nn.Module):
    WEIGHTS = torch.tensor([2.0, 1.0, 1.5, 0.5])
    def forward(self, pred, target):
        return ((pred - target) ** 2 * self.WEIGHTS.to(pred.device)).mean()

def train(X_pca, y, input_dim, model_dir, epochs, batch_size, lr, val_split, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TorcsDataset(X_pca, y)
    n_val = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    model = TorcsDriverNet(input_dim=input_dim).to(device)
    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler_amp = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f"\n[TRAIN] Device: {device} | Parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(1, epochs + 1):
        t0, tr_loss = time.time(), 0.0
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item() * len(xb) for xb, yb in val_loader) / n_val
        scheduler.step()
        print(f"Ep {epoch:>3d} | Tr Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | Time: {time.time()-t0:.1f}s")

    return model.cpu().eval()

if __name__ == "__main__":
    logs_dir = "logs"
    model_dir = "models"
    epochs = 50
    batch = 256
    lr = 1e-3
    val_split = 0.15
    min_speed = 5.0
    pca_components = 0.95
    seed = 42
    
    # Ora legge correttamente TUTTI i parametri che passi da terminale
    opts, _ = getopt.getopt(sys.argv[1:], "", ["logs-dir=", "model-dir=", "epochs=", "batch=", "lr=", "val-split=", "min-speed=", "pca-components=", "seed="])
    for opt, val in opts:
        if opt == "--logs-dir": logs_dir = val
        elif opt == "--model-dir": model_dir = val
        elif opt == "--epochs": epochs = int(val)
        elif opt == "--batch": batch = int(val)
        elif opt == "--lr": lr = float(val)
        elif opt == "--val-split": val_split = float(val)
        elif opt == "--min-speed": min_speed = float(val)
        elif opt == "--pca-components":
            v = float(val)
            pca_components = int(v) if v >= 1.0 else v
        elif opt == "--seed": seed = int(val)

    # Calcola il percorso esatto della cartella basandosi su dove si trova lo script
    base = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(logs_dir): logs_dir = os.path.join(base, logs_dir)
    if not os.path.isabs(model_dir): model_dir = os.path.join(base, model_dir)
    
    print(f"\n[INFO] Cerco i log nella cartella: {logs_dir}")
    
    X_raw, y = load_logs(logs_dir, min_speed)
    X_pca, input_dim, scaler_obj, pca_obj = fit_scaler_pca(X_raw, pca_components, model_dir)
    
    model = train(X_pca, y, input_dim, model_dir, epochs, batch, lr, val_split, seed)
    
    print("\n[EXPORT] Compilazione JIT del modello End-to-End...")
    end_to_end_model = TorcsEndToEndNet(model, scaler_obj, pca_obj)
    dummy_input = torch.zeros(1, N_RAW_FEATURES, dtype=torch.float32)
    
    traced_model = torch.jit.trace(end_to_end_model, dummy_input)
    traced_path = os.path.join(model_dir, "torcs_driver_jit.pt")
    traced_model.save(traced_path)
    
    print(f"[EXPORT] Modello JIT ultrarapido pronto e salvato in: {traced_path}")