import pygame
import snakeoil3_jm2 as snakeoil3
import time
import json


class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller trovato! Collega un gamepad e riprova.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Controller rilevato: {self.joystick.get_name()}")

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }

        # -------------------------------------------------------
        # MAPPING — modifica questi indici se il tuo controller
        # si comporta in modo diverso (vedi sezione DEBUG sotto)
        # -------------------------------------------------------
        self.AXIS_STEER  = 0   # Analog stick sinistro orizzontale
        self.AXIS_ACCEL  = 5   # Trigger destro  (RT / R2)
        self.AXIS_BRAKE  = 2   # Trigger sinistro (LT / L2)

        # -------------------------------------------------------
        # CAMBIO AUTOMATICO
        # -------------------------------------------------------
        # RPM sopra cui si scala su, RPM sotto cui si scala giù
        self.UPSHIFT_RPM   = 8000
        self.DOWNSHIFT_RPM = 3000
        # Tempo minimo (secondi) tra un cambio e l'altro (evita scalate continue)
        self.SHIFT_COOLDOWN = 0.8
        self._last_shift_time = 0.0

    def _axis(self, idx):
        """Legge un asse e restituisce il valore (-1.0 … 1.0)."""
        try:
            return self.joystick.get_axis(idx)
        except pygame.error:
            return 0.0

    def _btn(self, idx):
        """Legge un pulsante (True/False)."""
        try:
            return bool(self.joystick.get_button(idx))
        except pygame.error:
            return False

    def _auto_shift(self, sensors):
        rpm   = sensors.get('rpm', 0)
        gear  = self.state['gear']
        now   = time.time()

        # Non toccare la retromarcia
        if gear < 1:
            return

        if now - self._last_shift_time < self.SHIFT_COOLDOWN:
            return

        if rpm > self.UPSHIFT_RPM and gear < 6:
            self.state['gear'] += 1
            self._last_shift_time = now
        elif rpm < self.DOWNSHIFT_RPM and gear > 1:
            self.state['gear'] -= 1
            self._last_shift_time = now

    def _trigger_to_01(self, raw):
        """
        Converte un trigger in (0 … 1).
        - Se a riposo il trigger vale -1  → usa la formula (raw+1)/2
        - Se a riposo il trigger vale  0  → usa direttamente max(0, raw)
        Cambia TRIGGER_REST_VALUE in base all'output del debug.
        """
        TRIGGER_REST_VALUE = 0.0   # ← metti -1.0 se il debug mostra -1 a riposo
        if TRIGGER_REST_VALUE < -0.5:
            return (raw + 1.0) / 2.0
        else:
            return max(0.0, raw)

    def update(self, sensors):
        # Processa gli eventi pygame (necessario per aggiornare gli assi)
        pygame.event.pump()

        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)

        # ========================
        # STERZO
        # ========================
        raw_steer = self._axis(self.AXIS_STEER)

        # Sterzo invertito: cambia il segno qui se necessario
        raw_steer = -raw_steer

        # Dead zone centrale
        if abs(raw_steer) < 0.05:
            raw_steer = 0.0

        # Limite sterzata in funzione della velocità
        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_input = raw_steer * max_steer

        # Correzione angolo (stabilità)
        if abs(steer_input) > 0.01:
            steer_input -= angle * 0.3

        # Smooth
        self.state['steer'] += (steer_input - self.state['steer']) * 0.3
        if abs(self.state['steer']) < 0.02:
            self.state['steer'] = 0.0

        # ========================
        # ACCELERATORE
        # ========================
        raw_accel = self._trigger_to_01(self._axis(self.AXIS_ACCEL))
        self.state['accel'] += (raw_accel - self.state['accel']) * 0.15

        # ========================
        # FRENO
        # ========================
        raw_brake = self._trigger_to_01(self._axis(self.AXIS_BRAKE))
        self.state['brake'] += (raw_brake - self.state['brake']) * 0.2

        # ========================
        # CAMBIO AUTOMATICO
        # ========================
        self._auto_shift(sensors)

        # Clamp
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0,  min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0,  min(1.0, self.state['brake']))
        self.state['gear']  = max(-1,   min(6,   self.state['gear']))


# ============================================================
# DEBUG — esegui questo per vedere gli indici del tuo controller
# ============================================================
def debug_controller():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("Nessun controller trovato.")
        return
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}")
    print(f"Assi: {joy.get_numaxes()}  |  Pulsanti: {joy.get_numbuttons()}")
    print("Premi Ctrl+C per uscire\n")
    try:
        while True:
            pygame.event.pump()
            axes   = [round(joy.get_axis(i), 2) for i in range(joy.get_numaxes())]
            btns   = [joy.get_button(i)         for i in range(joy.get_numbuttons())]
            print(f"Assi: {axes}   Pulsanti: {btns}", end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


# ============================================================
# MAIN
# ============================================================
def main():
    client     = snakeoil3.Client(p=3001, vision=False)
    controller = GamepadController()

    client.get_servers_input()

    print("Gamepad driving mode attivo")
    print("Stick SX → sterzo | RT → gas | LT → freno | RB/LB → marce")

    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    log_json = []
    t0   = time.time()
    step = 0

    while True:
        S = client.S.d

        controller.update(S)
        a = controller.state

        print(f"steer={a['steer']:+.2f}  accel={a['accel']:.2f}  "
              f"brake={a['brake']:.2f}  gear={a['gear']}  "
              f"rpm={S.get('rpm',0):.0f}  speed={S.get('speedX',0):.1f}")

        client.R.d['steer']  = a['steer']
        client.R.d['accel']  = a['accel']
        client.R.d['brake']  = a['brake']
        client.R.d['gear']   = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta']   = 0

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
            "action": {k: a[k] for k in ('steer','accel','brake','gear')},
            "state":  {k: S.get(k, 0) for k in ('speedX','trackPos','angle','rpm','damage')}
        })

        step += 1

        if step % 100 == 0:
            with open("manual_log.json", "w") as f:
                json.dump(log_json, f, indent=2)

        time.sleep(0.02)


if __name__ == "__main__":
    # Per testare il controller senza TORCS:
    #   debug_controller()
    main()