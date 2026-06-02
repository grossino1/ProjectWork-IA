import snakeoil3_jm2 as snakeoil
import numpy as np
import torch
import os

# 1. Trova il percorso ASSOLUTO della cartella in cui si trova questo script
cartella_script = os.path.dirname(os.path.abspath(__file__))

# 2. Costruisci il percorso esatto
nome_file_modello = 'torcs_driver_jit.pt'  # Sostituisci con il tuo nome reale
percorso_modello = os.path.join(cartella_script, 'models', nome_file_modello)

# --- BLOCCO DI DEBUG (stampiamo a schermo cosa sta succedendo) ---
print("-" * 50)
print(f"Cartella di lavoro attuale (CWD): {os.getcwd()}")
print(f"Cartella in cui si trova lo script: {cartella_script}")
print(f"PERCORSO ESATTO IN CUI STO CERCANDO IL MODELLO:\n{percorso_modello}")
print(f"IL FILE ESISTE DAVVERO QUI? ---> {os.path.exists(percorso_modello)}")
print("-" * 50)

# 3. Caricamento del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if not os.path.exists(percorso_modello):
        raise FileNotFoundError(f"File non trovato nel percorso: {percorso_modello}")
        
    model = torch.jit.load(percorso_modello, map_location=device)
    model.eval()
    print(f"Modello caricato con successo su {device}!")
except Exception as e:
    print(f"ERRORE CRITICO: {e}")
    exit()

# ... (qui continua con def drive_example(c): ecc.)

def drive_example(c):
    """
    Logica di guida che trasforma i sensori in input per la rete neurale PyTorch.
    """
    # Estrazione delle feature dai sensori
    feature_vector = [
        c.S.d['angle'],
        c.S.d['speedX'],
        c.S.d['speedY'],
        c.S.d['trackPos'],
        c.S.d['rpm'],
        *c.S.d['track']
    ]
    
    # 3. Conversione dei dati in Tensore PyTorch
    # Le reti neurali di default usano float32
    input_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(device)

    # 4. Predizione
    # torch.no_grad() dice a PyTorch di non calcolare i gradienti. 
    # Risparmia moltissima memoria e rende la predizione molto più veloce in tempo reale.
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Convertiamo il risultato (Tensore) di nuovo in un array Numpy (e lo spostiamo su CPU se era su GPU)
    output = prediction.cpu().numpy()[0]
    
    # Estraiamo i valori (la rete restituisce [sterzo, accelerazione, freno, marcia])
    steer, accel, brake = output[0], output[1], output[2]
    predicted_gear_norm = output[3]

    # Sicurezza: limitiamo i valori per evitare comandi fuori range che TORCS potrebbe ignorare
    steer = float(np.clip(steer, -1.0, 1.0))
    accel = float(np.clip(accel, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))

    # 5. Invio dei comandi al simulatore
    c.R.d['steer'] = steer
    c.R.d['accel'] = accel
    c.R.d['brake'] = brake
    
    # --- LOGICA CAMBIO ALLINEATA AL CONTROLLO MANUALE ---
    speed = c.S.d['speedX']
    predicted_gear = int(round(predicted_gear_norm * 6.0))

    # Se la rete neurale suggerisce la retromarcia (e siamo quasi fermi), oppure andiamo già indietro
    if (predicted_gear == -1 and speed < 5.0) or speed < -1.0:
        gear = -1
    else:
        # Logica marce in avanti identica a controllo_manuale.py
        target_gear = 1
        for i, th in enumerate([0, 45, 90, 145, 200, 250]):
            if speed > th: target_gear = i + 1
        
        current_gear = c.S.d.get('gear', 1)
        # Mantieni la marcia in curva per stabilità
        gear = current_gear if abs(steer) > 0.4 else target_gear

    c.R.d['gear'] = gear

if __name__ == "__main__":
    # Inizializzazione del client Snakeoil forzando la porta 3001
    C = snakeoil.Client(p=3001)
    
    for i in range(C.maxSteps):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
        
    C.shutdown()