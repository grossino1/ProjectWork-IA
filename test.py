# =============================================================================
# test.py —  Test del Sistema Ibrido Finale
# =============================================================================
# SCOPO DEL FILE:
#   Questo script esegue il test del sistema ibrido completo.
#   Carica i due modelli addestrati (actor_GOLDEN_Stable.pth + actor_CASO2_best_dist.pth) e li combina
#   in tempo reale durante la guida, usando il Blend Factor spaziale per
#   decidere quale modello comanda l'auto in base alla posizione sulla pista.
#   NON esegue nessun aggiornamento dei pesi — è pura inferenza.
#
# FLUSSO:
#   1. Carica actor_GOLDEN_STABLE.pth  (Imitation Learning — guida base)
#   2. Carica actor_CASO2_best_dist.pth (DDPG RL — specialista Corkscrew)
#   3. Per ogni step: calcola alpha(distFromStart) e somma linearmente i due output
#   4. Registra i risultati su test_hybrid_results.csv
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import csv
import os
from gym_torcs import TorcsEnv  # wrapper OpenAI Gym per TORCS

# ====================================================================================================
# COSTANTI — Devono essere IDENTICHE a quelle usate in reinforce_optimization.py e train_imitation.py
# ====================================================================================================
# Se questi valori differissero dal training, il blend si attiverebbe in zone
# diverse rispetto a quelle su cui i modelli sono stati addestrati, causando
# comportamenti imprevedibili.
INPUT_SIZE  = 30    # dimensione vettore di stato (30 sensori normalizzati)
OUTPUT_SIZE = 4     # steer, accel, brake, gear

CORK_HARD_START = 2400.0   # distanza (m) inizio zona Corkscrew — controllo 100% RL
CORK_HARD_END   = 2750.0   # distanza (m) fine zona Corkscrew
CORK_BLEND_ZONE = 100.0    # ampiezza (m) delle rampe di transizione lineare
CRITICAL_POINT  = 2477.0   # metro esatto del cambio di pendenza (svalicamento cieco)

# ======================================================================================================================
# ARCHITETTURA DELLA RETE NEURALE — Actor (identica allla classe Actor del renforce e alla classe ExpertModel del train)
# ======================================================================================================================
# Questa classe deve essere identica a quella usata durante il training, perché se cambia anche un solo layer, il load_state_dict(),
# che carica i pesi dei modelli addestrati, fallisce perché le forme dei tensori non corrispondono.
# 
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
class Actor(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()

        ## Base comune: tre layer lineari densi con normalizzazione per stabilizzare i gradienti 
        # e funzione di attivazione non lineare ReLU (Se il segnale è negativo lo azzera (interruttore spento), 
        # se è positivo lo fa passare invariato).
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

        # Teste separate: ognuna calcola un singolo valore output continuo 
        self.steer_head = nn.Linear(64, 1)  # sterzo
        self.accel_head = nn.Linear(64, 1)  # acceleratore
        self.brake_head = nn.Linear(64, 1)  # freno
        self.gear_head  = nn.Linear(64, 1)  # marcia 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Propagazione attraverso la base comune
        features = self.base(state)

        # Ogni testa produce il proprio output con la propria funzione di attivazione:
        # - tanh  per steer: range [-1.0, +1.0] (destra/sinistra)
        # - sigmoid per accel/brake: range [0.0, 1.0] (rilasciato/pieno)
        # - sigmoid per gear: range [0.0, 1.0] (normalizzato, sarà de-normalizzato a runtime)
        steer = torch.tanh(self.steer_head(features))    # [-1.0, +1.0]
        accel = torch.sigmoid(self.accel_head(features)) # [0.0, 1.0]
        brake = torch.sigmoid(self.brake_head(features)) # [0.0, 1.0]
        gear  = torch.sigmoid(self.gear_head(features))  # [0.0, 1.0]   

        # Concatena i 4 output in un unico vettore [steer, accel, brake, gear]
        return torch.cat([steer, accel, brake, gear], dim=-1)

# ======================================================================================
# PREPROCESSING DELLO STATO — identico a reinforce_optimization.py e train_imitation.py
# ======================================================================================
# Converte il dizionario di sensori grezzi di TORCS in un vettore NumPy
# normalizzato di 30 elementi, pronto per essere passato alla rete.
#
# ORDINE DEI 30 ELEMENTI:
#   [0]     angle       / pi          → [-1, 1]
#   [1]     gear        / 6.0         → [0, 1]
#   [2]     rpm         / 10000       → [0, 1]
#   [3-5]   speedX/Y/Z  / 200.0       → [-1, 1]
#   [6-24]  track[0..18]/ 200.0       → [0, 1]  (19 sensori laser)
#   [25]    trackPos    / 3.0         → [-0.33, 0.33]
#   [26-29] wheelSpinVel/ 100.0       → [0, 1]
#
# PERCHÉ QUESTO ORDINE SPECIFICO?
#   Deve essere identico all'ordine usato in train_imitation.py e
#   reinforce_optimization.py. Se l'ordine cambia, la rete riceve
#   le feature nei posti sbagliati e produce output insensati.
# =============================================================================
def preprocess_state(S: dict) -> np.ndarray:
    # Funzione helper: legge un valore dal dizionario, ritorna default=0.0 se assente
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

    # np.concatenate unisce tutti i vettori in un unico array monodimensionale X ∈ R^30
    return np.concatenate([angle, gear, rpm, speed, track, track_pos, wheel])

# =============================================================================
# BLEND FACTOR — funzione di miscelazione spaziale α(s)
# =============================================================================
# Calcola il coefficiente alpha che determina quanto "peso" dare al Cork Actor
# rispetto al Golden Stable in base alla posizione sulla pista (in metri).
#
# COMPORTAMENTO:
#   α = 0.0 → 100% Golden Stable (Imitation Learning)   [fuori dal Corkscrew]
#   α = 1.0 → 100% Cork Actor (Reinforcement Learning)  [dentro il Corkscrew]
#   0 < α < 1 → blend lineare (zona di transizione, 100m alle soglie)
#
# FORMULA NELLE ZONE DI TRANSIZIONE:
#   Ingresso: α = (s - 2300) / 100  (da 2300m a 2400m)
#   Uscita:   α = 1 - (s - 2750) / 100  (da 2750m a 2850m)
#
# PERCHÉ LE RAMPE LINEARI?
#   Senza rampe, il passaggio istantaneo da un modello all'altro a 150 km/h
#   produrrebbe un gradino nel comando di sterzo o freno, destabilizzando l'auto.
# =============================================================================
def compute_blend_factor(track_idx: float) -> float:
    # Prima della zona blend in ingresso → tutto Golden Stable
    if track_idx < CORK_HARD_START - CORK_BLEND_ZONE:
        return 0.0
    # Dopo la zona blend in uscita → tutto Golden Stable
    if track_idx > CORK_HARD_END + 200:
        return 0.0
    # Rampa lineare di ingresso (2300m → 2400m): alpha sale da 0 a 1
    if track_idx < CORK_HARD_START:
        return (track_idx - (CORK_HARD_START - CORK_BLEND_ZONE)) / CORK_BLEND_ZONE
    # Zona piena Corkscrew (2400m → 2750m): tutto Cork Actor
    if track_idx <= CORK_HARD_END:
        return 1.0
    # Rampa lineare di uscita (2750m → 2850m): alpha scende da 1 a 0
    return 1.0 - (track_idx - CORK_HARD_END) / CORK_BLEND_ZONE

# =============================================================================
# CARICAMENTO MODELLI
# =============================================================================
def load_model(path: str, label: str) -> Actor:
    # Verifica che il file esista prima di tentare il caricamento
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modello '{label}' non trovato: {path}\n"
            f"Assicurati che il file esista nella directory corrente."
        )
    # Dichiarazione Rete Neurale
    model = Actor(INPUT_SIZE)
    # map_location="cpu": carica il modello sulla CPU anche se era stato salvato su GPU.
    # model.load_state_dict(...): Prende il dizionario di numeri appena estratto e lo "inietta" 
    # dentro i layer della tua rete neurale. 
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()  # disabilita dropout e BatchNorm in modalità training
    print(f"  [OK] {label}: {path}")
    return model

# =============================================================================
# FUNZIONE DI TEST PRINCIPALE
# =============================================================================
def test_hybrid(anchor_path: str, cork_path: str, n_episodes: int):
    print("\n" + "=" * 65)
    print("TEST IBRIDO: Golden Stable + Caso2 al Corkscrew")
    print("=" * 65)
    print(f"  Modello FUORI Cork: {anchor_path}")
    print(f"  Modello DENTRO Cork: {cork_path}")
    print(f"  Zona blend:  {CORK_HARD_START - CORK_BLEND_ZONE:.0f}m "
          f"→ {CORK_HARD_END + CORK_BLEND_ZONE:.0f}m")
    print(f"  Episodi: {n_episodes}")
    print("=" * 65)

    # Carica entrambi i modelli in memoria
    print("\nCaricamento modelli...")
    anchor     = load_model(anchor_path, "Golden Stable (anchor)")  # modello IL
    cork_actor = load_model(cork_path,   "Cork Actor (caso2)")      # modello RL

    # Inizializza l'ambiente TORCS:
    # vision=False: niente camera, usa solo sensori numerici
    # throttle=True: controllo separato accel e brake (non throttle unificato)
    # gear_change=True: la rete può cambiare marcia
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)

    # Apre il file CSV per registrare i risultati di ogni episodio
    log_path   = "test_hybrid_results.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "max_dist_from_start", "dist_raced",
        "lap_time", "completed", "crash_point", "steps"
    ])

    results = []  # lista dei risultati per il riepilogo finale

    # =========================================================================
    # LOOP DEGLI EPISODI
    # =========================================================================
    for episode in range(1, n_episodes + 1):
        print(f"\n{'─' * 65}")
        print(f"EPISODIO {episode}/{n_episodes}")
        print(f"{'─' * 65}")

        # Connessione a TORCS con retry automatico in caso di errore
        # relaunch=True al primo episodio: avvia TORCS fresh
        # relaunch=False negli episodi successivi: usa il Soft Reset (4.5s vs 22s)
        connected = False
        relaunch  = (episode == 1)
        while not connected:
            try:
                env.reset(relaunch=relaunch)
                connected = True
            except Exception as e:
                print(f"  [TORCS] Retry connessione: {e}")
                relaunch = True
                time.sleep(5.0)

        # =======================================
        # STATO INIZIALE DEL SIMULATORE 
        # =======================================
        # Estrazione del dizionario dei sensori grezzi direttamente dal server TORCS.
        # Contiene dati come velocità, angolo, distanze dai bordi (track), ecc.
        obs              = env.client.S.d       

        # Trasformazione dei dati grezzi in un vettore NumPy normalizzato pronto per la rete neurale.
        # Questa funzione rimuove i dati inutili e scala gli altri per facilitare la convergenza.
        state            = preprocess_state(obs) 

        # Memorizzazione della distanza totale percorsa dall'auto dall'inizio della sua "vita" nel simulatore.
        # Serve come punto di riferimento per calcolare quanti metri effettivi percorre in questo giro.
        start_dist_raced = obs.get('distRaced', 0.0) 

        # Flag booleano di controllo del loop: finché è False, l'auto continua a guidare.
        done             = False

        # Contatore dei fotogrammi (passi): serve per il limite di sicurezza (es. max 12.000 passi)
        # per evitare che l'auto giri all'infinito se rimane incastrata a bassa velocità.
        steps            = 0

        # Distanza massima raggiunta (distFromStart)
        max_dist         = 0.0    

        # Distanza percorsa nell'episodio  
        dist_raced       = 0.0    

        # Flag booleano che indica se il giro è stato completato
        lap_completed    = False

        # Salva eventualmente il metro dove è avvenuto il crash
        crash_point      = None  

        # =====================================================================
        # LOOP DI CONTROLLO — un'iterazione = un time step del simulatore
        # =====================================================================
        while not done:
            # Converte lo stato in tensore PyTorch [1, 30] (batch size 1)
            state_t   = torch.FloatTensor(state).unsqueeze(0)

            # Legge la posizione sulla pista  
            track_idx = float(obs.get('distFromStart', 0.0))

            # Calcola il coefficiente per il blend factor per capire quale modello utilizzare
            alpha     = compute_blend_factor(track_idx)

            # Aggiorna la distanza massima raggiunta
            max_dist = max(max_dist, track_idx)

            # ---- INFERENZA PARALLELA DEI DUE MODELLI ----
            with torch.no_grad():  # disabilita il calcolo dei gradienti (ovvero non permette l'addestramento)
                anchor_action = anchor(state_t).numpy()[0]      # output del modello IL
                cork_action   = cork_actor(state_t).numpy()[0]  # output del modello RL

            # ---- BLEND DETERMINISTICO ----
            # Formula: u_finale = (1-α)*u_IL + α*u_RL
            # COMPORTAMENTO:
            #   α = 0.0 → 100% Golden Stable (Imitation Learning)   [fuori dal Corkscrew]
            #   α = 1.0 → 100% Cork Actor (Reinforcement Learning)  [dentro il Corkscrew]
            #   0 < α < 1 → blend lineare (zona di transizione, 100m alle soglie)
            blended = (1.0 - alpha) * anchor_action + alpha * cork_action

            # ---- COSTRUZIONE AZIONE FINALE ----
            env_action    = blended.copy()
            env_action[0] = np.clip(blended[0], -1.0,  1.0)   # steer: vincola in [-1, 1]
            env_action[1] = np.clip(blended[1],  0.0,  1.0)   # accel: vincola in [0, 1]
            env_action[2] = np.clip(blended[2],  0.0,  1.0)   # brake: vincola in [0, 1]

            # De-normalizza la marcia: da [0,1] a intero [1,6]
            # Formula inversa: gear_int = round(gear_norm * 5 + 1)
            gear          = int(round(np.clip(blended[3], 0.0, 1.0) * 5.0 + 1.0))
            env_action[3] = float(max(1, min(6, gear)))  # clamp a [1, 6]

            # ---- STEP NEL SIMULATORE ----
            # Invia i comandi a TORCS e riceve il nuovo stato
            try:
                _, _, env_done, _ = env.step(env_action)
                if env_done:  # TORCS ha segnalato la fine dell'episodio
                    done = True
            except Exception as e:
                print(f"  [TORCS] Errore step: {e}")
                done = True
                break

            # Aggiorna lo stato per il prossimo step
            obs = env.client.S.d
            if not obs:
                break

            # Calcola la distanza percorsa dall'inizio dell'episodio
            dist_raced = obs.get('distRaced', 0.0) - start_dist_raced
            track_pos  = obs.get('trackPos', 0.0)
            
            # =====================================================================
            # ---- CONDIZIONI DI TERMINAZIONE DELL'EPISODIO ----
            # =====================================================================

            # 1. GIRO COMPLETATO: dist_raced > 3610m (lunghezza pista Corkscrew)
            if dist_raced > 3610:
                lap_time = obs.get('lastLapTime', 0.0)
                print(f"\n  ✓ GIRO COMPLETATO! Lap time: {lap_time:.2f}s")
                lap_completed = True
                done = True

            # 2. USCITA DI PISTA: |trackPos| > 2.1 (oltre i cordoli)
            elif abs(track_pos) > 2.1:
                crash_point = track_idx
                print(f"\n  ✗ SCHIANTO al metro {track_idx:.1f}m "
                      f"(trackPos={track_pos:.2f})")
                done = True

            # 3. TIMEOUT: troppi step senza completare il giro
            elif steps > 12000:
                print(f"\n  ✗ TIMEOUT ({steps} step senza completare il giro)")
                done = True

            # Preprocessing per il prossimo step
            state  = preprocess_state(obs)
            steps += 1

        # =====================================================================
        # FINE EPISODIO — registra risultati
        # =====================================================================
        lap_time = obs.get('lastLapTime', 0.0) if obs else 0.0
        results.append({
            'episode':    episode,
            'max_dist':   max_dist,
            'dist_raced': dist_raced,
            'lap_time':   lap_time,
            'completed':  lap_completed,
            'crash':      crash_point,
            'steps':      steps,
        })

        # Scrive la riga sul CSV e svuota il buffer (flush) per sicurezza
        log_writer.writerow([
            episode, f"{max_dist:.1f}", f"{dist_raced:.1f}",
            f"{lap_time:.2f}", lap_completed,
            f"{crash_point:.1f}" if crash_point else "",
            steps
        ])
        log_file.flush()

        summary = ("✓ COMPLETO" if lap_completed
                   else f"✗ crash a {crash_point:.1f}m" if crash_point
                   else "✗ timeout")
        print(f"\n  Ep {episode}: {summary} | max_dist={max_dist:.1f}m | "
              f"lap={lap_time:.2f}s | steps={steps}")

    log_file.close()
    env.end()  # chiude TORCS

    # =========================================================================
    # RIEPILOGO FINALE — statistiche aggregate su tutti gli episodi
    # =========================================================================
    print("\n" + "=" * 65)
    print("RIEPILOGO FINALE")
    print("=" * 65)

    # Per ogni episodio stampa la sua terminazione
    for r in results:
        if r['completed']:
            status = f"✓ COMPLETO  lap={r['lap_time']:.2f}s"
        elif r['crash']:
            status = f"✗ crash a {r['crash']:.1f}m"
        else:
            status = f"✗ timeout a {r['max_dist']:.1f}m"
        print(f"  Ep {r['episode']:2d}: {status}")

    # Stampa le statistiche di media tra tutti gli episodi completati
    completed = [r for r in results if r['completed']]
    crashes_before_cork = [r for r in results
                           if r['crash'] and r['crash'] < CORK_HARD_START]
    crashes_at_cork     = [r for r in results
                           if r['crash'] and CORK_HARD_START <= r['crash'] <= CORK_HARD_END]

    print(f"\nGiri completati:      {len(completed)}/{len(results)}")
    print(f"Crash prima del Cork: {len(crashes_before_cork)}/{len(results)}")
    print(f"Crash al Cork:        {len(crashes_at_cork)}/{len(results)}")

    if completed:
        best_lt = min(r['lap_time'] for r in completed)
        avg_lt  = np.mean([r['lap_time'] for r in completed])
        print(f"Miglior lap time:     {best_lt:.2f}s")
        print(f"Lap time medio:       {avg_lt:.2f}s")

    avg_max = np.mean([r['max_dist'] for r in results])
    print(f"Distanza media max:   {avg_max:.1f}m")
    print(f"\nLog salvato in: {log_path}")

# =============================================================================
# ENTRY POINT — argomenti da linea di comando
# =============================================================================
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