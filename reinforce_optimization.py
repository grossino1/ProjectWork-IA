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

# ===========================================================================
# 1. DIMENSIONI DEI VETTORI NEURALI
# ===========================================================================
# Dimensione dello Stato (Input): i 30 sensori normalizzati letti da TORCS 
# (es. track, trackPos, speedX, angle, wheelSpinVel, ecc.)
INPUT_SIZE   = 30

# Dimensione dell'Azione (Output): i 4 comandi fisici continui inviati alla vettura
# [0]: Sterzo (-1 a +1), [1]: Acceleratore (0 a 1), [2]: Freno (0 a 1), [3]: Marcia (-1 a +1 normalizzato)
OUTPUT_SIZE  = 4

# ===========================================================================
# 2. PARAMETRI DEL RINFORZO (DDPG)
# ===========================================================================
# Fattore di sconto temporale (Gamma): impostato a 0.99 (molto alto). 
# Dice al Critic che il futuro conta quasi quanto il presente, costringendo l'auto 
# a essere lungimirante e a sacrificare velocità immediata (frenando) pur di finire il giro.
# Questo permette all'auto di frenare preventivamete quando arriva all'apice del cavatappi prima della curva cieca.
GAMMA        = 0.99

# Coefficiente di Soft Update (Tau): controlla la velocità delle reti ombra (Target).
# Impostato a 0.005, significa che ad ogni passo le reti Target, cioè il modello che impara lentamente dal modello online,
# assimilano solo lo 0.5% dei pesi sinaptici delle reti Online, garantendo stabilità matematica.
TAU          = 0.005

# Learning Rate dell'Actor (Il Pilota): impostato a 1e-4 (0.0001).
LR_ACTOR     = 1e-4

# Learning Rate del Critic (L'Ingegnere): impostato a 2e-4 (0.0002).
# Viene mantenuto volutamente più alto di quello dell'Actor. Questo perché:
# il giudice (Critic) deve imparare a dare i voti leggermente più velocemente di quanto 
# il pilota (Actor) ci metta ad apprendere le manovre.
LR_CRITIC    = 2e-4

# ===========================================================================
# 3. GESTIONE DELLA MEMORIA E DEI BATCH DI STUDIO
# ===========================================================================
# Quanti fotogrammi di passata esperienza vengono estratti insieme dal buffer ad ogni ciclo di studio.
BATCH_SIZE   = 256

# Capienza massima del Prioritized Replay Buffer (Il Diario dei Ricordi).
# Può contenere fino a 200.000 transizioni singole prima di sovrascrivere le più vecchie.
BUFFER_SIZE  = 200_000

# ===========================================================================
# 4. GEOMETRIA E MASCHERAMENTO SPAZIALE (I Confini della Pista)
# ===========================================================================
# Il metro esatto della pista in cui si attiva chirurgicamente la logica ibrida 
# e in cui la cork_mask inizia a considerare i campioni validi per l'addestramento.
CORK_HARD_START  = 2400.0

# Il metro esatto in cui finisce la zona calda e il controllo totale torna all'Imitation Learning.
CORK_HARD_END    = 2750.0

# Ampiezza della rampa di transizione (100 metri). 
CORK_BLEND_ZONE  = 100.0

# Il culmine dello scollinamento cieco (La cresta alpina del Corkscrew).
CRITICAL_POINT   = 2477.0

# ===========================================================================
# 5. PARAMETRI DI ESPLORAZIONE (IL RUMORE GAUSSIANO)
# ===========================================================================
# Deviazione standard iniziale del rumore (Sigma): l'auto parte oscillando dell'8% 
# sui pedali per mappare e scoprire nuove reazioni fisiche nel vuoto.
NOISE_SIGMA_INIT  = 0.08

# Il limite minimo sotto il quale il rumore non deve scendere, garantendo sempre 
# un 1% di micro-esplorazione residua di sicurezza anche ad addestramento avanzato.
NOISE_SIGMA_MIN   = 0.01

# Coefficiente di decadimento per episodio: ad ogni fine giro il rumore si contrae dello 0.5%,
# stabilizzando la guida man mano che l'Actor diventa più sicuro delle sue staccate.
NOISE_DECAY       = 0.995

# Sigma di emergenza (Cooldown): se l'auto entra in un loop di fallimenti, il rumore 
# viene abbassato d'impatto al 2% per costringerla a rimettersi sui binari sicuri dell'uomo.
COOLDOWN_SIGMA    = 0.02

# ===========================================================================
# 6. VINCOLO GEOMETRICO E AMMORTIZZATORE DI STILE
# ===========================================================================
# Il peso dell'Imitation Learning nella Loss Ibrida (15%).
# Dice all'Actor: "Cerca all'85% di ottimizzare i freni per non morire, ma rimani vincolato 
# al 15% allo stile fluido dell'Anchor umana per non innescare lo zig-zagging".
IMITATION_WEIGHT  = 0.15

# ===========================================================================
# 7. CRITERI DI SICUREZZA E WARMUP
# ===========================================================================
# Nei primi 3 episodi l'Actor guida senza studiare, girando a vuoto per accumulare 
# le prime pagine di ricordi nel Replay Buffer prima di iniziare a calcolare i gradienti.
ACTOR_WARMUP_EPISODES = 3

# Finestra storica degli schianti: quanti crash consecutivi monitorare per attivare il Cooldown.
CRASH_WINDOW     = 3

# Tolleranza spaziale (±50 metri dal punto critico). Se l'auto si stampa per 3 volte di fila 
# in questa esatta sotto-porzione di pista, scatta il Cooldown automatico per l'instabilità.
CRASH_DIST_TOL   = 50.0

# ---------------------------------------------------------------------------
# ARCHITETTURA 
# ---------------------------------------------------------------------------
# La classe Actor è la rete neurale che rappresenta la Policy (la politica di azione). È l'unica entità che decide materialmente come muovere 
# i comandi dell'auto (sterzo, acceleratore, freno, marcia) partendo dai sensori della pista.

# La sua architettura interna (i tre hidden layers da 128, 128 e 64 neuroni e le 4 teste)  è volutamente identica a quella del modello ad imitazione (ExpertModel). 
# Questo trucco serve a far sì che lo script possa prendere il file actor_GOLDEN_STABLE.pth (il diploma dell'imitazione) e caricarlo direttamente dentro l'Actor del rinforzo.

# Il suo ruolo qui: L'Actor parte con le conoscenze dell'essere umano, ma in questo script viene attivamente modificato dal Reinforcement Learning 
# per scoprire nuove staccate millimetriche dentro il Corkscrew.
class Actor(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):  #nel caso quando creiamo l'oggetto non specifichiamo nulla di default utilizza il valore della costante INPUT_SIZE
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

# La classe Critic è la rete neurale che approssima la Funzione di Valore Q(s,a). Il Critic non guida l'auto e non sa cosa sia un pedale o un volante.

# Cosa fa concretamente: Guarda l'input del simulatore (state) E l'azione appena scelta dall'Actor (action) fondendoli insieme nel primo livello 
# (nn.Linear(input_size + OUTPUT_SIZE, 256)). Il suo unico output finale è un singolo numero (il valore Q), ovvero il voto a lungo termine che dà a quella mossa.

# Il suo ruolo qui: Funge da insegnante privato dell'Actor. Dice all'Actor se la traiettoria scelta nel Corkscrew porterà a un punteggio alto o a una penalità.
class Critic(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size + OUTPUT_SIZE, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLUqu(),
            nn.Linear(256, 1)   #viene compresso tutto in un neurone finale: prende i 256 valori dello stato precendente, moltiplica ognuno di essi per un
                                #peso differente, somma tutto insieme e aggiunge un valore chiaamato  bias 

            #il singolo neurone rappresenta il voto finala (il valore Q), che riassume tutte le definizione complesse dei 256 neuroni
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:    #il Critic riceve in input due dati lo state, la situazione attuale, l'action, la mossa che ha appena scelto l'Actor
        return self.network(torch.cat([state, action], dim=1))   #una volta ottenuto il vettore tramite torch.cat di 34 elementi, lo facciamo passare nella rete neurale che ci restituirà il valore Q


def soft_update(target: nn.Module, source: nn.Module, tau: float):   #funzione utilizzata sia sul critic che sul actor, serve a far si che i modelli target seguona i progressi del modello online (source)
    for tp, sp in zip(target.parameters(), source.parameters()):    #fa scorrere le liste dei neuroni target e online (source) e gli associa alias tp e sp
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)    #usa la funzione Media Mobile Esponenziale per calcolare il nuovo valore del modello target, partendo dal vecchio valore di target e dal valore del modello online(source)

# ---------------------------------------------------------------------------
# REPLAY BUFFER con prioritizzazione semplificata
# Mentre la macchina gira nel simulatore TORCS, vive delle esperienze. Invece di dimenticarle subito, le scrive in questo diario. 
# Quando poi deve mettersi a studiare (fase di update), apre il diario e rilegge i vecchi ricordi per capire dove ha sbagliato.
# La parola "Prioritized" (Prioritario) significa che la rete non rilegge i ricordi a caso (uniformemente). 
# L'auto preferisce rileggere più spesso le pagine del diario in cui ha fatto gli errori più gravi o le pagine che parlano del tratto di pista più difficile (il Corkscrew).
# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer    = deque(maxlen=capacity) # È lo scompartimento in cui vengono salvati i ricordi veri e propri (Stato, Azione, Ricompensa, Stato Successivo).
        self.priorities = deque(maxlen=capacity) # È un secondo scompartimento che contiene un semplice numero per ogni ricordo: la sua importanza. C'è una corrispondenza 1:1 tra un ricordo nel buffer e la sua priorità.
        self.capacity  = capacity
        self.alpha     = 0.6    # È un esponente che regola quanto vogliamo essere severi con la prioritizzazione. Se valesse 0, torneremmo a scegliere i ricordi in modo totalmente casuale; impostato a 0.6, crea un ottimo bilanciamento tra il rileggere i ricordi importanti e il non dimenticare del tutto quelli normali.
        self.beta      = 0.4    # esponente IS weights (aumenta nel tempo)

    # Ogni volta che la macchina compie un'azione in pista, invoca questo metodo per salvare l'esperienza:
    def push(self, state, action, reward, next_state, done, track_idx):
        # Priorità iniziale: Cork zone vale di più
        base_priority = 2.0 if CORK_HARD_START < track_idx < CORK_HARD_END else 1.0   #base_priority assegna l'importnaza del ricordo: se l'auto si trova tra l'inzio e la fine del cavatappi, l'importanza e pari a 2 altrimenti e pari ad 1
        self.buffer.append((state, action, reward, next_state, done, track_idx))    #salvataggio del ricordo nel buffer
        self.priorities.append(base_priority)     #nella lista parallela salviamo la priorità del ricordo. Corrispondenza: primo ricordo nel buffer avrà la prima priorità nella lista

    # Quando l'ottimizzatore deve aggiornare i neuroni, chiede al buffer un blocchetto di ricordi
    def sample(self, batch_size: int):
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha  #crea un array con le priorità, con esponente alpha in maniera tale da dare una probabilità di essere scelti anche i ricordi meno importanti
        probs /= probs.sum()   #trasformiamo le priorità in probabilità vere e proprie

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)   # np.random.choice pesca dalla lunghezza del buffer in maniera casuale, un numero di indici, associati ai ricordi, pari a size, senza reinserimento (una volta estratto un ricordo non posso ripescarlo), grazie a p=probs pescherà più spesso i ricordi del cavatappi
        samples = [self.buffer[i] for i in indices]   #estrae i dati dal buffer per tutti i 256 (BATCH_SIZE) indici scelti
        state, action, reward, next_state, done, track_idx = zip(*samples)   #zip trasforma la lista di pacchetti samples in righe separate

        # Importance sampling weights
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return (np.stack(state), np.stack(action), np.stack(reward),
                np.stack(next_state), np.stack(done), np.stack(track_idx),
                indices, weights.astype(np.float32))

    # Dopo che l'Actor ha studiato un ricordo, il Critic calcola il TD-error (Temporal Difference error), ovvero lo scarto tra ciò che la rete si 
    # aspettava che succedesse e ciò che è successo davvero (la misura della "sorpresa" o dell'errore di valutazione).
    # Questo metodo prende quell'errore assoluto (|err|) e lo imposta come nuova priorità di quel ricordo. Un ricordo in cui 
    # la rete ha sbagliato di molto diventa istantaneamente prioritario per i cicli di studio successivi.
    def update_priorities(self, indices, td_errors: np.ndarray):
        """Aggiorna priorità in base agli errori TD."""
        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6
            # Mantenere priorità minima Cork
            if CORK_HARD_START < float(self.buffer[idx][5]) < CORK_HARD_END:
                p = max(p, 2.0)
            self.priorities[idx] = p

    # Man mano che i passi aumentano (step), beta cresce linearmente fino a raggiungere il valore massimo di 1.0.
    def anneal_beta(self, step: int, total_steps: int):
        """Beta aumenta linearmente da 0.4 a 1.0 nel corso del training."""
        self.beta = min(1.0, 0.4 + 0.6 * step / total_steps)

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# REWARD SHAPING
# L'agente a rinforzo (DDPG) non sa cosa sia una curva o un limite di velocità; il suo unico obiettivo matematico è massimizzare la somma dei punti che riceve. 
# Tramite questo blocco di codice, abbiamo  riprogrammato l'istinto dell'auto per insegnarle come sopravvivere alla sella cieca del Corkscrew.
# ---------------------------------------------------------------------------
def compute_reward(obs: dict,
                   prev_obs: dict,
                   done: bool,
                   start_dist_raced: float) -> tuple[float, bool]:
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

    # ---- Reward base: progressione: ----
    # Prima di gestire il Corkscrew, l'auto deve comunque saper avanzare lungo il tracciato. Ad ogni frame di gioco, riceve un punteggio calcolato su tre vettori fisici:
    progress     = speed * np.cos(angle) # Più l'auto va veloce in avanti, più punti accumula
    slip_penalty = abs(speed * np.sin(angle)) # Se l'auto derapa o si mette di traverso, subisce una penalità proporzionale.
    track_penalty = abs(track_pos) # Più l'auto si allontana dalla linea ideale, più punti perde.

    reward = progress - 0.3 * slip_penalty - 0.6 * track_penalty

    # Più si ci avvicina alla zona del Corkscrew, più i reward diventano rigidi, essendo questo lo scopo di ottimizzazione del RL
    # ---- ZONA CRITICA: avvicinamento al cambio pendenza ----
    
    # Tra 2350m e CRITICAL_POINT l'auto DEVE decelerare, L'agente capisce istantaneamente che correre troppo in quel tratto è "doloroso".
    if 2350 < dist < CRITICAL_POINT:
        target_speed_ms = 40.0    # ~144 km/h
        speed_kmh = speed * 3.6
        if speed_kmh > target_speed_ms:
            # Penalizza velocità eccessiva in avvicinamento
            overspeed_penalty = (speed_kmh - target_speed_ms) * 0.05
            reward -= overspeed_penalty

        # Premia la presenza del freno
        brake_val = obs.get('brake', 0.0)
        if brake_val > 0.1:
            reward += 1.5    # se l'agente attiva il pedale del freno (brake > 0.1), riceve un bonus

    # ---- ZONA CRITICA: cambio pendenza esatto indica che si ci trova proprio nel Corkscrew ----
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

    # L'ultima parte della funzione serve a gestire la conclusione anticipata di un episodio (il fallimento o il successo)
    # Se l'interruttore force_done viene settato su True. L'episodio si interrompe immediatamente, resettando TORCS. 
    # L'agente ricorderà quel crash come la peggiore esperienza possibile.

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
# BLEND ACTOR/ANCHOR — transizione graduale 
# Questa funzione calcola il valore di $\alpha$ (Alpha), un coefficiente matematico compreso tra 0.0 e 1.0 che decide, 
# metro dopo metro, quanta autorità dare all'Actor (il rinforzo) e quanta all'Anchor (l'imitazione congelata).
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
# Questa funzione prende la telemetria grezza memorizzata nel dizionario di TORCS (S) e la trasforma in un unico vettore NumPy normalizzato.
# La regola d'oro qui è la coerenza assoluta: i divisori numerici utilizzati in questo script sono identici al millimetro a quelli che avete usato nel file train_imitation.py della rete supervisionata.
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
# All'inizio della funzione, il codice configura l'architettura. Qui vedi nascere l'Ancora e le due coppie di Actor/Critic:
"""
               ┌──► ACTOR ONLINE (Impara ad ogni curva) ───┐
               │                                           ├── Soft Update (τ)
               ├──► ACTOR TARGET (Copia rallentata fissa) ─┘
Input Stato ───┤
               ├──► CRITIC ONLINE (Vota le mosse correnti) ┐
               │                                           ├── Soft Update (τ)
               ├──► CRITIC TARGET (Vota le mosse future) ──┘
               │
               └──► ANCHOR (Imitation Learning — Congelato 100%)
"""

def reinforce(base_model_path: str = None):
    # =========================================================================
    # FASE 1: SELEZIONE E CARICAMENTO DEL MODELLO PRE-ADDESTRATO (IMITATION)
    # =========================================================================
    # Se non passiamo un percorso specifico, lo script cerca automaticamente
    # il "diploma" ottenuto dall'Imitation Learning nei file locali.
    if base_model_path is None:
        for candidate in ["actor_IL_CASO1.pth", "actor_GOLDEN_STABLE.pth"]:
            if os.path.exists(candidate):
                base_model_path = candidate
                break

    # Se non trova nessun file .pth, interrompe l'esecuzione: l'RL non può
    # partire da zero, ha bisogno delle fondamenta umane.
    if base_model_path is None or not os.path.exists(base_model_path):
        print("ERRORE: Nessun modello base trovato.")
        print("Esegui prima: python caso1_imitation_learning.py")
        sys.exit(1)

    print(f"[Base model] {base_model_path}")

    # =========================================================================
    # FASE 2: SETUP DELL'AMBIENTE E DELLE 5 ENTITÀ NEURALI (LE RETI)
    # =========================================================================
    # Inizializza il simulatore TORCS disattivando la grafica 3D per velocizzare
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)

    # 1. RETE ACTOR ONLINE: Il pilota attivo che impara e cambia ad ogni curva
    actor  = Actor(INPUT_SIZE)
    actor.load_state_dict(torch.load(base_model_path))

    # 2. RETE ANCHOR (ANCORA): Copia congelata al 100% dell'imitazione umana.
    # Non verrà mai modificata. Serve come guida sicura e termine di paragone.
    anchor = Actor(INPUT_SIZE)
    anchor.load_state_dict(torch.load(base_model_path))
    anchor.eval() # Disattiva comportamenti di addestramento (es. LayerNorm fisso)
    for p in anchor.parameters():
        p.requires_grad = False # Spegne il calcolo dei gradienti (congelamento)

    # 3. RETE ACTOR TARGET: L'ombra rallentata dell'Actor per stabilizzare i calcoli
    target_actor  = Actor(INPUT_SIZE)
    target_actor.load_state_dict(actor.state_dict())

    # 4. RETE CRITIC ONLINE: L'ingegnere di pista che assegna i voti (Valore Q) alle mosse
    critic = Critic(INPUT_SIZE)
    
    # 5. RETE CRITIC TARGET: L'ombra rallentata del Critic per calcolare i voti futuri
    target_critic = Critic(INPUT_SIZE)
    target_critic.load_state_dict(critic.state_dict())

    # =========================================================================
    # FASE 3: SETUP DEGLI OTTIMIZZATORI, SCHEDULER E MEMORIA (PER)
    # =========================================================================
    # Configura gli algoritmi Adam per aggiornare i pesi di Actor e Critic
    actor_optimizer  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # I Schedulers riducono automaticamente il Learning Rate se il punteggio
    # (la reward) ristagna per più di 15 episodi, evitando di sballare i neuroni.
    actor_scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        actor_optimizer, patience=15, factor=0.5, min_lr=1e-6)
    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, patience=15, factor=0.5, min_lr=1e-6)

    # Inizializza il Diario dei Ricordi prioritario con capienza 200.000 eventi
    memory = PrioritizedReplayBuffer(BUFFER_SIZE)

    # Setup del file CSV per registrare la telemetria delle epoche di studio
    log_path = "rl_optimization_results_caso2.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "steps", "reward", "last_lap_time",
        "dist_raced", "max_dist_from_start",
        "cork_crashes", "noise_sigma", "lr_actor"
    ])

    # Variabili di stato per monitorare l'avanzamento del training
    noise_sigma      = NOISE_SIGMA_INIT
    crash_history    = deque(maxlen=CRASH_WINDOW) # Registra gli ultimi 3 schianti
    best_dist_raced  = 0.0
    best_lap_time    = float("inf")
    total_updates    = 0

    print("\n" + "=" * 60)
    print("CASO 2 — RL Fine-tuning Corkscrew")
    print("=" * 60)

    # =========================================================================
    # FASE 4: IL MACRO-CICLO DEGLI EPISODI (1000 TENTATIVI IN PISTA)
    # =========================================================================
    for episode in range(1000):
        
        # Sistema di connessione robusto: se TORCS va in crash, riavvia il server UDP
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

        # Estrazione e inizializzazione dello stato corrente della macchina
        obs              = env.client.S.d
        prev_obs         = obs.copy()
        state            = preprocess_state(obs)
        start_dist_raced = obs.get('distRaced', 0.0)
        episode_reward   = 0.0
        done             = False
        steps            = 0
        max_dist_from_start = 0.0
        cork_crash_this_ep  = False

        # MECCANISMO DI COOLDOWN: Se l'auto si è schiantata per 3 volte di fila
        # nello stesso identico punto del Corkscrew, dimezza temporaneamente
        # il rumore (esplorazione) per costringerla ad andare sul sicuro.
        recent_crashes = list(crash_history)
        if (len(recent_crashes) >= CRASH_WINDOW and
                all(abs(d - CRITICAL_POINT) < CRASH_DIST_TOL
                    for d in recent_crashes)):
            effective_sigma = COOLDOWN_SIGMA
            print(f"  [Cooldown] Crash ripetuti a ~{CRITICAL_POINT}m → σ={COOLDOWN_SIGMA}")
        else:
            effective_sigma = noise_sigma

    # =========================================================================
    # FASE 5: IL MICRO-CICLO DI GUIDA IN PISTA (PASSO DOPO PASSO)
    # =========================================================================
        actor.eval() # Mette l'actor in modalità valutazione mentre guida

        while not done:
            # Converte lo stato NumPy in un Tensore PyTorch pronto per le reti
            state_t     = torch.FloatTensor(state).unsqueeze(0)
            track_idx   = float(obs.get('distFromStart', 0.0))
            
            # Calcola il fattore di miscelazione ALPHA in base ai metri percorsi
            alpha        = compute_blend_factor(track_idx)
            max_dist_from_start = max(max_dist_from_start, track_idx)

            # Interroga contemporaneamente l'Actor Online e l'Ancora ad imitazione
            with torch.no_grad():
                actor_action  = actor(state_t).numpy()[0]
                anchor_action = anchor(state_t).numpy()[0]

            # FUSIONE GEOMETRICA CONTINUA: Calcola la media pesata dei comandi
            raw_action = (alpha * actor_action + (1.0 - alpha) * anchor_action)

            # ESPLORAZIONE MIRATA SUL CORKSCREW: Se siamo nella zona calda (alpha > 0.5)
            # e sono passati i primi 3 giri di warmup, inietta il rumore gaussiano concentrato su sterzo e freno per spingere l'Actor a testare nuove staccate..
            if alpha > 0.5 and episode >= ACTOR_WARMUP_EPISODES:
                noise_vec = np.random.normal(0, effective_sigma, size=OUTPUT_SIZE).astype(np.float32)
                noise_vec[0] *= 1.5    # Amplifica il rumore sullo sterzo per testare traiettorie
                noise_vec[2] *= 2.0    # Amplifica il rumore sul freno per testare staccate profonde
                raw_action = raw_action + noise_vec

            # CLIP E ADATTAMENTO FISICO: Costringe i pedali e lo sterzo nei limiti fisici.
            env_action = raw_action.copy()
            env_action[0] = np.clip(env_action[0], -1.0,  1.0)  # Sterzo bloccato tra -1 e +1
            env_action[1] = np.clip(env_action[1],  0.0,  1.0)  # Gas bloccato tra 0 e 1
            env_action[2] = np.clip(env_action[2],  0.0,  1.0)  # Freno bloccato tra 0 e 1
            
            # De-normalizzazione della marcia: trasforma il valore [0,1] in marcia reale [1,6]
            gear = int(round(np.clip(raw_action[3], -1.0, 1.0) * 5.0 + 1.0))
            env_action[3] = float(max(1, min(6, gear)))

            # Invia materialmente i 4 comandi al simulatore TORCS
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

            # Calcola la reward personalizzata tramite il Reward Shaping
            custom_reward, force_done = compute_reward(obs, prev_obs, done, start_dist_raced)
            if force_done:
                done = True # Interrompe l'episodio se tocca un muro o completa il giro

            dist_raced = obs.get('distRaced', 0.0) - start_dist_raced

            # Se l'auto si schianta (done=True) mentre l'RL aveva il controllo (alpha > 0.3),
            # registra la posizione del crash nel diario storico.
            if done and alpha > 0.3:
                crash_history.append(track_idx)
                cork_crash_this_ep = True

            # Accumula la reward dell'episodio e scrive l'evento nel Replay Buffer prioritario
            episode_reward += custom_reward
            memory.push(state, raw_action, custom_reward, preprocess_state(obs), done, track_idx)
            
            # Aggiorna lo stato temporale per il frame successivo
            state     = preprocess_state(obs)
            prev_obs  = obs.copy()
            steps    += 1

            if steps > 12000: # Limite di sicurezza antibradipo (evita giri infiniti)
                done = True

    # =========================================================================
    # FASE 6: IL CICLO DI OTTIMIZZAZIONE E AGGIORNAMENTO NEURALE (UPDATE RL)
    # =========================================================================
        # Una volta terminato l'episodio in pista, se ci sono abbastanza ricordi nel diario (BATCH_SIZE), lo script spegne i motori dell'auto e avvia la fase di ottimizzazione matematica.
        # e se sono passati i 3 episodi iniziali di warmup.
        if len(memory) > BATCH_SIZE and episode >= ACTOR_WARMUP_EPISODES:
            num_updates = min(100, steps // 4) # Bilancia il tempo di studio con la lunghezza del giro
            actor.train() # Mette l'actor in modalità addestramento (sblocca i gradienti)

            for upd_step in range(num_updates):
                total_updates += 1
                # Incrementa linearmente il parametro BETA dell'Importance Sampling
                memory.anneal_beta(total_updates, total_steps=500_000)

                # Estrae un pacchetto di 256 ricordi prioritari dal diario
                (b_s, b_a, b_r, b_ns, b_d,
                 b_ti, b_idx, b_iw) = memory.sample(BATCH_SIZE)

                # Trasforma l'intero batch estratto in tensori PyTorch
                b_s   = torch.FloatTensor(b_s)  # Stati correnti
                b_a   = torch.FloatTensor(b_a)  # Azioni eseguite
                b_r   = torch.FloatTensor(b_r).unsqueeze(1) # Ricompense
                b_ns  = torch.FloatTensor(b_ns) # Stati successivi
                b_d   = torch.FloatTensor(b_d).unsqueeze(1) # Flag di fine episodio
                b_iw  = torch.FloatTensor(b_iw).unsqueeze(1) # Pesi d'importanza (IS weights)

                # -------------------------------------------------------------
                # AGGIORNAMENTO DEL CRITIC (L'INGEGNERE DI PISTA)
                # -------------------------------------------------------------
                # Calcola il voto futuro ideale usando le reti TARGET (Equazione di Bellman)
                with torch.no_grad():
                    q_next   = target_critic(b_ns, target_actor(b_ns))
                    q_target = b_r + (1 - b_d) * GAMMA * q_next

                # Il Critic Online fa la sua stima sul presente
                q_pred      = critic(b_s, b_a)
                
                # Il TD-Error misura lo scarto assoluto tra la stima e l'obiettivo ideale
                td_errors   = (q_pred - q_target).detach().cpu().numpy().squeeze()
                
                # Calcola la Loss del Critic pesandola con l'Importance Sampling
                critic_loss = (b_iw * F.mse_loss(q_pred, q_target, reduction='none')).mean()

                # Ottimizzazione dei neuroni del Critic tramite retropropagazione
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0) # Ritaglio gradienti anti-esplosione
                critic_optimizer.step()

                # Aggiorna dinamicamente le priorità del Replay Buffer con il nuovo TD-Error
                memory.update_priorities(b_idx, td_errors)

                # -------------------------------------------------------------
                # AGGIORNAMENTO DELL'ACTOR (IL PILOTA) — LOSS IBRIDA
                # -------------------------------------------------------------
                # Genera una maschera binaria spaziale: vale 1 solo se il campione
                # si trova dentro il Corkscrew, altrimenti vale 0.
                cork_mask = torch.FloatTensor(
                    [1.0 if CORK_HARD_START < t < CORK_HARD_END else 0.0 for t in b_ti]
                ).unsqueeze(1)

                # L'Actor aggiorna i suoi neuroni SOLO se ci sono campioni del Corkscrew nel batch
                if cork_mask.sum() > 0:
                    pred_a    = actor(b_s)
                    
                    # 1. Componente Rinforzo (85%): Cerca di massimizzare il voto del Critic
                    rl_loss   = -critic(b_s, pred_a)

                    # 2. Componente Imitazione (15%): Calcola lo scostamento rispetto all'Ancora frozen
                    with torch.no_grad():
                        anch_a = anchor(b_s)
                    imit_loss = F.mse_loss(pred_a, anch_a, reduction='none').mean(dim=1, keepdim=True)

                    # FUSIONE FINALE DELLE DUE LOSS: Moltiplica l'obiettivo per la maschera spaziale
                    total_loss = (
                        cork_mask * (
                            (1.0 - IMITATION_WEIGHT) * rl_loss + IMITATION_WEIGHT * imit_loss
                        )
                    ).sum() / cork_mask.sum()

                    # Ottimizzazione dei 128, 128 e 64 neuroni dell'Actor Online
                    actor_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5) # Ritaglio rigoroso dei gradienti
                    actor_optimizer.step()

                # -------------------------------------------------------------
                # MECCANISMO DI SOFT UPDATE (L'ALLINEAMENTO DELLE RETI TARGET)
                # -------------------------------------------------------------
                # Le reti Target compiono un micro-passo dello 0.5% (TAU=0.005) per inseguire
                # le reti Online che hanno appena aggiornato i propri pesi sinaptici.
                soft_update(target_actor,  actor,  TAU)
                soft_update(target_critic, critic, TAU)

        # Riduci progressivamente l'ampiezza del rumore ad ogni episodio (Decay)
        noise_sigma = max(NOISE_SIGMA_MIN, noise_sigma * NOISE_DECAY)

        # Comunica il punteggio finale dell'episodio agli scheduler per regolare il LR
        if episode >= ACTOR_WARMUP_EPISODES:
            actor_scheduler.step(-episode_reward)
            critic_scheduler.step(-episode_reward)

    # =========================================================================
    # FASE 7: SALVATAGGIO DEI MODELLI MIGLIORI E CONSOLIDAMENTO SUL DISCO
    # =========================================================================
        lap_time  = obs.get('lastLapTime', 0.0) if obs else 0.0
        
        # Se l'auto ha stabilito un record di distanza percorsa, salva il checkpoint
        if dist_raced > best_dist_raced:
            best_dist_raced = dist_raced
            torch.save(actor.state_dict(), "actor_CASO2_best_dist.pth")
            print(f"  ★ New best distance: {dist_raced:.1f}m")

        # Se l'auto ha completato il giro stabilendo il record di tempo, salva il checkpoint
        if 0 < lap_time < best_lap_time:
            best_lap_time = lap_time
            torch.save(actor.state_dict(), "actor_CASO2_best_laptime.pth")
            print(f"  ★ New best lap time: {lap_time:.2f}s")

        # Checkpoint periodico di sicurezza ogni 50 tentativi
        if episode % 50 == 0:
            torch.save(actor.state_dict(), f"actor_CASO2_ep{episode}.pth")

        # Scrittura dei parametri di performance correnti nel log CSV
        lr_now = actor_optimizer.param_groups[0]['lr']
        log_writer.writerow([
            episode, steps, f"{episode_reward:.2f}", f"{lap_time:.2f}",
            f"{dist_raced:.1f}", f"{max_dist_from_start:.1f}",
            int(cork_crash_this_ep), f"{noise_sigma:.4f}", f"{lr_now:.2e}"
        ])
        log_file.flush()

        # Stampa a schermo l'andamento telemetrico dell'episodio per l'operatore
        print(f"Ep {episode:4d} | "
              f"dist={dist_raced:6.1f}m "
              f"maxPos={max_dist_from_start:6.1f}m "
              f"R={episode_reward:8.1f} "
              f"σ={noise_sigma:.3f} "
              f"lr={lr_now:.1e}")

    # Chiusura dei canali e spegnimento del simulatore al termine dei 1000 episodi
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
