import snakeoil3_jm2 as snakeoil3
import time
import json
import pygame

class ArcadeController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Controller rilevato: {self.joystick.get_name()}")
        else:
            print("ATTENZIONE: Nessun controller rilevato! Uso logica automatica senza input.")

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }
        
        # ====================================================
        # PARAMETRI CAMBIO AUTOMATICO - SETUP FORMULA 1
        # ====================================================
        self.rpm_uphift = 16000    
        self.rpm_downshift = 10000 
        self.last_shift_time = 0   
        self.shift_delay = 0.15    
        self.max_gear = 7          

    def auto_shifter(self, rpm, gear):
        """Logica del cambio automatico basata su RPM aggressivi"""
        current_time = time.time()
        
        if current_time - self.last_shift_time < self.shift_delay:
            return gear

        new_gear = gear

        if rpm > self.rpm_uphift and gear < self.max_gear:
            new_gear = gear + 1
            self.last_shift_time = current_time
        
        elif rpm < self.rpm_downshift and gear > 1:
            new_gear = gear - 1
            self.last_shift_time = current_time
            
        return new_gear

    def update(self, sensors):
        pygame.event.pump()
        
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)
        rpm = sensors.get('rpm', 0)
        
        # ============================================================
        # GESTIONE MARCE E RETROMARCIA (SICURA)
        # ============================================================
        rb_pressed = False
        if self.joystick:
            rb_pressed = self.joystick.get_button(5) # RB button
            
        forward_speed = abs(speed)

        if rb_pressed:
            # Se la macchina è ferma o quasi (< 2 km/h), metti la retro
            if forward_speed < 0.01:
                self.state['gear'] = -1
            else:
                # Sistema Anti-Crash: se premi RB mentre vai veloce, metti in folle
                self.state['gear'] = 0
        else:
            # Se eravamo in retro o folle e rilasciamo il tasto, torna in prima
            if self.state['gear'] in [-1, 0]:
                self.state['gear'] = 1
                
            # Logica di cambio automatico per F1
            self.state['gear'] = self.auto_shifter(rpm, self.state['gear'])

        # ============================================================
        # GESTIONE INPUT ANALOGICI
        # ============================================================
        if self.joystick:
            steer_input = -self.joystick.get_axis(0) 

            raw_accel = self.joystick.get_axis(5)
            raw_brake = self.joystick.get_axis(4)
            
            target_accel = (raw_accel + 1.0) / 2.0 if abs(raw_accel) > 0.001 else 0.0
            target_brake = (raw_brake + 1.0) / 2.0 if abs(raw_brake) > 0.001 else 0.0
        else:
            steer_input = 0.0
            target_accel = 0.5 
            target_brake = 0.0

        # ========================
        # LOGICA DI GUIDA (SMOOTHING)
        # ========================
        self.state['accel'] += (target_accel - self.state['accel']) * 0.4
        self.state['brake'] += (target_brake - self.state['brake']) * 0.5

        max_steer = max(0.15, 1.0 - speed / 280.0)
        steer_input *= max_steer

        if abs(steer_input) < 0.05:
            steer_target = 0.0
        else:
            stability = angle * 0.3
            steer_target = steer_input - stability

        self.state['steer'] += (steer_target - self.state['steer']) * 0.3

        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))


def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()

    client.get_servers_input()
    print("\n--- SETUP FORMULA 1 CARICATO ---")
    print("Sistema AVVIATO: Controller + Cambio Automatico Aggressivo")
    print("Tieni premuto RB da fermo per la retromarcia.")

    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,rpm\n")

    t0 = time.time()

    try:
        while True:
            S = client.S.d
            controller.update(S)
            a = controller.state
            
            # Formattazione per mostrare correttamente N (Folle) e REV (Retromarcia)
            if a['gear'] == -1:
                gear_label = "REV"
            elif a['gear'] == 0:
                gear_label = " N "
            else:
                gear_label = f" {a['gear']} "
            
            print(f"\rRPM: {int(S.get('rpm',0)):5d} | MARCIA: {gear_label} | VEL: {int(S.get('speedX',0)):3d} km/h | GAS: {a['accel']:.2f} ", end="")

            client.R.d['steer'] = a['steer']
            client.R.d['accel'] = a['accel']
            client.R.d['brake'] = a['brake']
            client.R.d['gear'] = int(a['gear'])
            client.R.d['clutch'] = 0.0

            client.respond_to_server()
            client.get_servers_input()

            current_time = time.time() - t0
            log_csv.write(f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},{S.get('speedX',0)},{S.get('rpm',0)}\n")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nChiusura in corso...")
    finally:
        log_csv.close()
        pygame.quit()

if __name__ == "__main__":
    main()