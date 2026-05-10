import csv
import json
import time
import pygame # <--- Nuova libreria per il controller
import snakeoil3_jm2 as snakeoil3

class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.state = {'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 1}
        self.last_gear_up = False
        self.last_gear_down = False
        
        # Controlliamo se c'est un joystick connesso
        if pygame.joystick.get_count() == 0:
            print("ERRORE: Nessun controller rilevato!")
            print("Assicurati di averlo collegato alla Macchina Virtuale Parallels.")
            self.joy = None
        else:
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()
            print(f"Controller connesso con successo: {self.joy.get_name()}")
            print(f"Assi rilevati: {self.joy.get_numaxes()}, Pulsanti: {self.joy.get_numbuttons()}")

    def update(self, sensors):
        if not self.joy:
            return # Se non c'è controller, non fare nulla

        # Aggiorna gli eventi di pygame (necessario per leggere i dati freschi)
        pygame.event.pump()

        # ========================
        # 1. STERZO (Levetta Sinistra - Asse 0)
        # ========================
        # I valori vanno da -1.0 (sinistra) a 1.0 (destra)
        steer_axis = -self.joy.get_axis(0)
        
        # Deadzone: i controller vecchi hanno un po' di "gioco" al centro. 
        # Ignoriamo i micro-movimenti per andare dritti perfetti.
        if abs(steer_axis) < 0.05:
            steer_axis = 0.0
            
        self.state['steer'] = steer_axis

        # ========================
        # 2. PEDALI (Grilletti - Assi 4 e 5)
        # ========================
        # IMPORTANTE: Su Windows/Xbox i grilletti partono da -1.0 (rilasciati) 
        # e arrivano a 1.0 (premuti). Dobbiamo convertirli in un range da 0.0 a 1.0.
        
        # Acceleratore (Grilletto destro, di solito Asse 5)
        raw_accel = self.joy.get_axis(5) 
        self.state['accel'] = max(0.0, (raw_accel + 1.0) / 2.0)
        
        # Freno (Grilletto sinistro, di solito Asse 4)
        raw_brake = self.joy.get_axis(4)
        self.state['brake'] = max(0.0, (raw_brake + 1.0) / 2.0)

        # ========================
        # 3. CAMBIO MARCE (Pulsanti 0 e 1)
        # ========================
        # Pulsante 0 (A su Xbox, Croce su PS) -> Marcia Su
        current_gear_up = self.joy.get_button(0)
        if current_gear_up and not self.last_gear_up:
            self.state['gear'] = min(6, self.state['gear'] + 1)
        self.last_gear_up = current_gear_up

        # Pulsante 1 (B su Xbox, Cerchio su PS) -> Marcia Giù
        current_gear_down = self.joy.get_button(1)
        if current_gear_down and not self.last_gear_down:
            self.state['gear'] = max(-1, self.state['gear'] - 1)
        self.last_gear_down = current_gear_down

        # Sicurezza
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = GamepadController()
    client.get_servers_input()
    
    print("LOGGING CONTROLLER ATTIVO - Tutto il traffico dati verrà salvato.")

    S_init = client.S.d
    headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
    
    for key, value in sorted(S_init.items()):
        if isinstance(value, list):
            for i in range(len(value)):
                headers.append(f"{key}_{i}")
        else:
            headers.append(key)

    with open("dataset_gamepad.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        t0 = time.time()
        step = 0
        
        try:
            while True:
                S = client.S.d
                controller.update(S)
                a = controller.state
                
                client.R.d.update(a)
                client.respond_to_server()
                client.get_servers_input()
                
                current_time = time.time() - t0
                row = [current_time, a['steer'], a['accel'], a['brake'], a['gear']]
                
                for key in sorted(S.keys()):
                    val = S[key]
                    if isinstance(val, list):
                        row.extend(val)
                    else:
                        row.append(val)
                
                writer.writerow(row)
                step += 1
                
                if step % 100 == 0:
                    print(f"Step: {step} | Vel: {S.get('speedX',0):.0f} | Steer: {a['steer']:.2f} | Accel: {a['accel']:.2f} | Brake: {a['brake']:.2f}")
                    
        except KeyboardInterrupt:
            print("\nSalvataggio completato. Dataset pronto.")
            pygame.quit()

if __name__ == "__main__":
    main()