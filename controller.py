import pygame
import snakeoil3_jm2 as snakeoil3
import time
import json

class ArcadeController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("ERRORE: Nessun controller rilevato! Collegalo e riavvia lo script.")
            exit()
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Controller rilevato: {self.joystick.get_name()}")

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }

    def update(self, sensors):
        pygame.event.pump()
        speed = sensors.get('speedX', 0)

        # ========================
        # 1. STERZO (Ottimizzato per F1)
        # ========================
        raw_steer = -self.joystick.get_axis(0)
        
        if abs(raw_steer) < 0.08:
            raw_steer = 0.0
            
        # Aumentato il limite minimo (da 0.15 a 0.3) perché le F1 curvano bene anche veloci
        max_steer_allowed = max(0.3, 1.0 - (abs(speed) / 300.0))
        target_steer = raw_steer * max_steer_allowed

        # Reattività aumentata (0.3 invece di 0.2) per un inserimento in curva più rapido
        self.state['steer'] += (target_steer - self.state['steer']) * 0.3

        # ========================
        # 2. ACCELERATORE, FRENO E ASSISTENZE (TCS + ABS)
        # ========================
        gas_axis = self.joystick.get_axis(5)
        brake_axis = self.joystick.get_axis(4)
        
        target_accel = max(0.0, (gas_axis + 1.0) / 2.0)
        target_brake = max(0.0, (brake_axis + 1.0) / 2.0)

        # --- ABS & TRAIL BRAKING ASSIST (NOVITÀ) ---
        # Se stai sterzando, riduciamo il freno massimo in base a quanto sterzi.
        # Così le ruote anteriori non si bloccano e mantengono aderenza direzionale!
        current_steer_abs = abs(self.state['steer'])
        if current_steer_abs > 0.05:
            # Più giri il volante, più la pressione del freno viene limitata
            max_brake_allowed = 1.0 - (current_steer_abs * 0.6) 
            target_brake = min(target_brake, max_brake_allowed)

        # --- LIMITATORE MARCE BASSE (F1) ---
        if self.state['gear'] == 1:
            target_accel = min(target_accel, 0.4)
        elif self.state['gear'] == 2:
            target_accel = min(target_accel, 0.7)

        # --- TRACTION CONTROL SYSTEM (TCS Aggressivo) ---
        wheel_spin = sensors.get('wheelSpinVel', [0, 0, 0, 0])
        
        if len(wheel_spin) >= 4:
            front_spin = wheel_spin[0] + wheel_spin[1]
            rear_spin = wheel_spin[2] + wheel_spin[3]
            slip = rear_spin - front_spin
            
            if slip > 2.0 and speed > 0: 
                target_accel = target_accel * 0.05 

        # Smoothing sui pedali
        self.state['accel'] += (target_accel - self.state['accel']) * 0.2
        # Lo smoothing del freno è stato abbassato a 0.3 per renderlo più pronto
        self.state['brake'] += (target_brake - self.state['brake']) * 0.3 

        # ========================
        # 3. CAMBIO AUTOMATICO E RETRO
        # ========================
        rb_pressed = self.joystick.get_button(5)

        if rb_pressed:
            self.state['gear'] = -1
        else:
            forward_speed = max(0, speed)
            if forward_speed < 60:
                self.state['gear'] = 1
            elif forward_speed < 80:
                self.state['gear'] = 2
            elif forward_speed < 110:
                self.state['gear'] = 3
            elif forward_speed < 150:
                self.state['gear'] = 4
            elif forward_speed < 190:
                self.state['gear'] = 5
            else:
                self.state['gear'] = 6

        # Clamp finale
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))
        self.state['gear'] = max(-1, min(6, self.state['gear']))


# ============================================================
# MAIN
# ============================================================

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()

    client.get_servers_input()

    print("==================================")
    print("Arcade driving mode attivo (Controller)")
    print("Levetta SX per sterzare, Grilletti per Gas/Freno")
    print("Tieni premuto RB (Dorsale Destro) per la Retromarcia!")
    print("==================================")

    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    log_json = []
    
    t0 = time.time()
    step = 0

    while True:
        S = client.S.d

        controller.update(S)
        a = controller.state
        
        print(f"steer={a['steer']:.2f} accel={a['accel']:.2f} brake={a['brake']:.2f} gear={a['gear']}", end='\r')

        client.R.d['steer'] = a['steer']
        client.R.d['accel'] = a['accel']
        client.R.d['brake'] = a['brake']
        client.R.d['gear'] = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()
        client.get_servers_input()

        current_time = time.time() - t0

        log_csv.write(
            f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
            f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
            f"{S.get('rpm',0)},{S.get('damage',0)}\n"
        )

        log_json.append({
            "step": step,
            "time": current_time,
            "action": {
                "steer": a['steer'],
                "accel": a['accel'],
                "brake": a['brake'],
                "gear": a['gear']
            },
            "state": {
                "speedX": S.get('speedX', 0),
                "trackPos": S.get('trackPos', 0),
                "angle": S.get('angle', 0),
                "rpm": S.get('rpm', 0),
                "damage": S.get('damage', 0)
            }
        })

        step += 1

        if step % 100 == 0:
            with open("manual_log.json", "w") as f:
                json.dump(log_json, f, indent=2)
                
        time.sleep(0.02)


if __name__ == "__main__":
    main()