import socket
import sys
import getopt
import os
import time
import math
import csv

PI= 3.14159265359
data_size = 2**17

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    if not w: return ''
    if x<mn: x= mn
    if x>mx: x= mx
    tx= mx-mn
    if tx<=0: return 'backwards'
    upw= tx/float(w)
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0:
        if x < 0:
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else:
            negnonpu= -mn + min(0,mx)
    if mx > 0:
        if x > 0:
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else:
            posnonpu= mx - max(0,mn)
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision
        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes= 10 # Forzato a 10 giri per la registrazione
        self.trackname= 'unknown'
        self.stage= 3 
        self.debug= False
        self.maxSteps= 100000  
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg='%s(init %s)' % (self.sid,a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def get_servers_input(self):
        if not self.so: return
        sockdata= str()
        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                continue
            if '***identified***' in sockdata:
                continue
            elif '***shutdown***' in sockdata or '***restart***' in sockdata:
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                break

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            sys.exit(-1)

    def shutdown(self):
        if not self.so: return
        self.so.close()
        self.so = None

class ServerState():
    def __init__(self):
        self.d= dict()
    def parse_server_str(self, server_string):
        servstr= server_string.strip()[:-1]
        sslisted= servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

class DriverAction():
    def __init__(self):
       self.d= { 'accel':0.2, 'brake':0, 'clutch':0, 'gear':1, 'steer':0, 'focus':[-90,-45,0,45,90], 'meta':0 }
    def __repr__(self):
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list: out+= '%.3f' % v
            else: out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out

def destringify(s):
    if not s: return s
    if type(s) is str:
        try: return float(s)
        except ValueError: return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        else: return [destringify(i) for i in s]

# ================= USER CONFIGURABLE PARAMETERS =================
TARGET_SPEED = 300       
STEER_GAIN = 20          
CENTERING_GAIN = 0.1     
GEAR_SPEEDS = [0, 65, 90, 145, 195, 250]  
ENABLE_TRACTION_CONTROL = True 

def calculate_steering(S):
    # --- STABILITY CONTROL (Anticipo Sbandata) ---
    # Se l'auto ha una velocità laterale eccessiva (speedY), significa che sta scivolando.
    # Ridurre lo sterzo in questo momento aiuta a riprendere aderenza invece di innescare un testacoda.
    slip_correction = 1.0
    if abs(S['speedY']) > 3.0: # Soglia di intervento stabilità
        slip_correction = 0.6 # Riduce la sterzata per stabilizzare l'auto
    
    dynamic_gain = STEER_GAIN
    if S['speedX'] < 80:
        dynamic_gain = STEER_GAIN * 1.5

    steer = ((S['angle'] * dynamic_gain / math.pi) - (S['trackPos'] * CENTERING_GAIN)) * slip_correction
    return max(-1.0, min(1.0, steer))

def calculate_speed_logic(S):
    speed = S['speedX']
    front_vision = S['track'][5:14]
    max_dist_ahead = max(front_vision)

    # --- PREDICTIVE BRAKING (Analisi Curvatura) ---
    curvature = abs(S['track'][0] - S['track'][18])
    
    # Moltiplicatore molto più conservativo per curve strette (Corkscrew)
    multiplier = 2.5
    if max_dist_ahead < 100: multiplier = 1.8 # Curva vicina
    if curvature > 40: multiplier = 1.2      # Curva stretta
    if curvature > 70: multiplier = 0.8      # Curva estrema (Corkscrew Apex)
    
    safe_speed = max_dist_ahead * multiplier
    
    # Se siamo molto veloci e la pista curva, forziamo un target basso
    if speed > 150 and curvature > 30:
        safe_speed = min(safe_speed, 100.0)

    safe_speed = max(50.0, min(TARGET_SPEED, safe_speed))

    accel = 0.0
    brake = 0.0
    diff = safe_speed - speed 

    if diff > 0:
        accel = min(1.0, diff / 15.0) 
        if speed < 80: 
            accel = 1.0
    else:
        # Frenata molto più decisa se sopra safe_speed
        brake = min(1.0, abs(diff) / 10.0) 

    # Trigger di emergenza (ostacolo/curva cieca vicina)
    if S['track'][9] < 60 and speed > 100:
        brake = 1.0

    return accel, brake

def shift_gears(S):
    gear = 1
    speed = S['speedX']
    for i, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = i + 1
    return gear

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        slip = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])
        if slip > 5:
            accel -= 0.3
    return max(0.0, accel)

def drive_modular(c):
    S, R = c.S.d, c.R.d
    R['steer'] = calculate_steering(S)
    accel, brake = calculate_speed_logic(S)
    brake = brake * (1.0 - (abs(R['steer']) * 0.5))
    R['accel'] = traction_control(S, accel)
    R['brake'] = max(0.0, min(1.0, brake))
    R['gear'] = shift_gears(S)
    if R['brake'] > 0.05:
        R['accel'] = 0.0

if __name__ == "__main__":
    dataset_filename = "dataset_bot_154_clean.csv"
    csv_file = open(dataset_filename, "w", newline='')
    csv_writer = csv.writer(csv_file)
    
    KEYS_TO_IGNORE = [
        'opponents', 'focus', 'fuel', 'damage', 'z', 
        'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced', 'racePos'
    ]
    
    headers_written = False
    step_count = 0
    t0 = time.time()
    
    C = Client(p=3001)
    
    try:
        for episode in range(10): # Registra 10 giri
            print(f"--- INIZIO REGISTRAZIONE GIRO {episode+1}/10 ---")
            if episode > 0:
                C.setup_connection()
            
            while True:
                C.get_servers_input()
                if C.so is None: break
                
                drive_modular(C)
                
                if not headers_written:
                    headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
                    for key, value in sorted(C.S.d.items()):
                        if key in KEYS_TO_IGNORE: continue
                        if isinstance(value, list):
                            for i in range(len(value)): headers.append(f"{key}_{i}")
                        else: headers.append(key)
                    csv_writer.writerow(headers)
                    headers_written = True

                current_time = time.time() - t0
                row = [current_time, C.R.d['steer'], C.R.d['accel'], C.R.d['brake'], C.R.d['gear']]
                for key in sorted(C.S.d.keys()):
                    if key in KEYS_TO_IGNORE: continue
                    val = C.S.d[key]
                    if isinstance(val, list): row.extend(val)
                    else: row.append(val)
                csv_writer.writerow(row)
                step_count += 1
                
                if step_count % 100 == 0:
                    print(f"Giro {episode+1} | Step: {step_count} | Vel: {int(C.S.d['speedX'])} km/h")

                C.respond_to_server()
                
                # Se il giro finisce (meta=True o il server chiude), passa al prossimo
                if C.R.d['meta']: break
                
    finally:
        csv_file.close()
        C.shutdown()
        print(f"Registrazione completata in {dataset_filename}. Step totali: {step_count}")
