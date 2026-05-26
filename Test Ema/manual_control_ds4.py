import pygame
import socket
import sys
import os
import time
import csv
import numpy as np

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# Percorso richiesto dall'utente
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "manual_laps.csv")

# Mapping DualShock 4
AXIS_STEER = 0 
AXIS_ACCEL = 5 
AXIS_BRAKE = 4 

class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = self.destringify(w[1:])
    def destringify(self, s):
        if not s: return s
        if type(s) is str:
            try: return float(s)
            except ValueError: return s
        elif type(s) is list:
            if len(s) < 2: return self.destringify(s[0])
            else: return [self.destringify(i) for i in s]

class DriverAction():
    def __init__(self):
        self.d = {'accel': 0, 'brake': 0, 'clutch': 0, 'gear': 1, 'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

def get_joystick_input(joystick, current_speed):
    pygame.event.pump()
    raw_steer = -joystick.get_axis(AXIS_STEER)
    
    if abs(raw_steer) < 0.02:
        steer = 0.0
    else:
        steer = np.sign(raw_steer) * (abs(raw_steer) ** 2.0)
        if current_speed > 50:
            steer *= max(0.4, 1.0 - (current_speed - 50) / 300.0)

    accel = (joystick.get_axis(AXIS_ACCEL) + 1.0) / 2.0
    brake = (joystick.get_axis(AXIS_BRAKE) + 1.0) / 2.0
    return steer, accel, brake

def manual_recording():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERRORE: Collega il controller!")
        return
    js = pygame.joystick.Joystick(0)
    js.init()

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1)
    
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"
    so.sendto(initmsg.encode(), (HOST, PORT))
    
    while True:
        try:
            sockdata, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in sockdata.decode():
                print(">>> CONNESSO A TORCS! REGISTRAZIONE CONTINUA ATTIVA.")
                break
        except:
            so.sendto(initmsg.encode(), (HOST, PORT))

    S = ServerState()
    R = DriverAction()
    KEYS_TO_IGNORE = ['opponents', 'focus', 'fuel', 'damage', 'z', 'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced', 'racePos']
    
    # Apertura file in modalità append
    file_exists = os.path.isfile(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0
    csv_file = open(DATASET_PATH, "a", newline='')
    writer = csv.writer(csv_file)
    
    headers_written = file_exists
    step_count = 0
    t0 = time.time()
    initial_damage = None

    print(f"I dati vengono scritti in tempo reale in: {DATASET_PATH}")

    try:
        while True:
            try:
                sockdata, _ = so.recvfrom(DATA_SIZE)
                sockstr = sockdata.decode()
                S.parse_server_str(sockstr)
            except: continue

            if initial_damage is None:
                initial_damage = S.d.get('damage', 0)

            # Controlli
            speed = S.d.get('speedX', 0)
            steer, accel, brake = get_joystick_input(js, speed)
            
            # --- RIPRISTINO AIUTI ALLA GUIDA (TCS & ABS) ---
            wheel_vel = S.d.get('wheelSpinVel', [0,0,0,0])
            if isinstance(wheel_vel, list) and len(wheel_vel) == 4:
                # Traction Control
                if (wheel_vel[2]+wheel_vel[3]) - (wheel_vel[0]+wheel_vel[1]) > 15:
                    accel *= 0.5
                # ABS base
                if brake > 0.1 and speed > 15 and (wheel_vel[0]+wheel_vel[1])/2.0 < 5:
                    brake *= 0.1
            
            # Steering Priority (Evita bloccaggio in curva)
            if brake > 0.1 and abs(steer) > 0.15:
                brake *= (1.0 - abs(steer)*0.8)
            
            # --- LOGICA CAMBIO RIPRISTINATA ---
            target_gear = 1
            for i, th in enumerate([0, 45, 90, 145, 200, 250]):
                if speed > th: target_gear = i + 1
            
            # Mantieni la marcia in curva per stabilità (come nel codice originale)
            current_gear = S.d.get('gear', 1)
            gear = current_gear if abs(steer) > 0.4 else target_gear

            R.d['steer'], R.d['accel'], R.d['brake'], R.d['gear'] = steer, accel, brake, gear

            # Scrittura Header
            if not headers_written:
                headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
                for k in sorted(S.d.keys()):
                    if k in KEYS_TO_IGNORE: continue
                    val = S.d[k]
                    if isinstance(val, list): headers.extend([f"{k}_{i}" for i in range(len(val))])
                    else: headers.append(k)
                writer.writerow(headers)
                headers_written = True

            # Scrittura Riga
            row = [time.time()-t0, steer, accel, brake, target_gear]
            for k in sorted(S.d.keys()):
                if k in KEYS_TO_IGNORE: continue
                val = S.d[k]
                if isinstance(val, list): row.extend(val)
                else: row.append(val)
            writer.writerow(row)
            
            step_count += 1
            if step_count % 100 == 0:
                damage_curr = S.d.get('damage', 0) - initial_damage
                status = "OK" if damage_curr < 1 else "DANNEGGIATA"
                print(f"\rStep: {step_count} | Vel: {int(speed)} km/h | Stato: {status}", end="")

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\n\n>>> Registrazione interrotta dall'utente.")
    finally:
        csv_file.close()
        so.close()
        pygame.quit()
        print(f">>> File chiuso con successo: {DATASET_PATH}")

if __name__ == "__main__":
    manual_recording()
