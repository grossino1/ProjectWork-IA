#############################################
# MODULAR DRIVE LOGIC CON AI & TUNING       #
#############################################

import math

# ================= 1. SOGLIE CONFIGURABILI (Tuning per il Gruppo) =================
TARGET_SPEED_MAX = 195    # Velocità massima in rettilineo
TARGET_SPEED_CURVE = 55   # Velocità di sicurezza per le curve strette (Corkscrew)
DIST_LOOKAHEAD = 140      # Quanti metri prima della curva iniziare a rallentare
INTENSITY_THRESH = 0.10   # Sensibilità del radar per le curve (più basso = frena prima)
BRAKE_STRENGTH = 1.5      # Potenza del freno (Numeri piccoli = frena fortissimo)
ENABLE_TRACTION_CONTROL = True 

# ================= 2. HELPER FUNCTION (Il Radar di Granite) =================
def calculate_curve_intensity(track):
    diff_totale = 0
    coppie = [(0, 18), (2, 16), (4, 14), (6, 12)]
    for sx, dx in coppie:
        diff_totale += (track[sx] - track[dx])
    # Valore negativo = curva a sinistra, positivo = destra
    return clip(diff_totale / 80.0, -1.0, 1.0)

# ================= 3. MAIN DRIVE FUNCTION (Il Cervello) =================
def drive_modular(c):
    S, R = c.S.d, c.R.d
    
    # --- A. PERCEZIONE ---
    intensita = calculate_curve_intensity(S['track'])
    dist_davanti = S['track'][9]

    # --- B. DECISIONE VELOCITÀ ---
    if abs(intensita) > INTENSITY_THRESH or dist_davanti < DIST_LOOKAHEAD:
        target_speed = TARGET_SPEED_CURVE
    else:
        target_speed = TARGET_SPEED_MAX

    # --- C. ACCELERATORE E FRENO ---
    if S['speedX'] < target_speed:
        R['accel'] = 0.8
        R['brake'] = 0.0
    else:
        R['accel'] = 0.0
        # Freno dinamico calcolato sullo scarto di velocità
        R['brake'] = (S['speedX'] - target_speed) / BRAKE_STRENGTH

    # Panic Brake (emergenza muro)
    if dist_davanti < 65 and S['speedX'] > 80:
        R['brake'] = 1.0

    # --- D. STERZATA STABILE ---
    # L'angolo base è smorzato (* 0.8), l'intensità fa girare la macchina nelle curve
    R['steer'] = (S['angle'] * 0.8) - (S['trackPos'] * 0.1) + (intensita * 0.6)

    # --- E. GESTIONE MARCE ---
    if S['speedX'] < 60: R['gear'] = 1
    elif S['speedX'] < 100: R['gear'] = 2
    elif S['speedX'] < 140: R['gear'] = 3
    elif S['speedX'] < 180: R['gear'] = 4
    else: R['gear'] = 5

    # --- F. TRACTION CONTROL ---
    if ENABLE_TRACTION_CONTROL:
        slip = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])
        if slip > 4.0:
            R['accel'] -= 0.3

    # Limiti di sicurezza finali imposti dal simulatore
    R['accel'] = max(0.0, min(1.0, R['accel']))
    R['brake'] = max(0.0, min(1.0, R['brake']))
    R['steer'] = max(-1.0, min(1.0, R['steer']))

    # Stampa in terminale per capire cosa succede
    print(f"Vel: {int(S['speedX'])} | Freno: {R['brake']:.2f} | Dist: {dist_davanti:.0f} | Steer: {R['steer']:.2f}")

    return

# ================= 4. MAIN LOOP (NON TOCCARE) =================
if __name__ == "__main__":
    C = Client(p=3001)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_modular(C)
        C.respond_to_server()
    C.shutdown()