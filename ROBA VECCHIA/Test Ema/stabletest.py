import torch
import numpy as np
import time
from gym_torcs import TorcsEnv

# Sostituisci 'nome_del_tuo_script' con il nome reale del tuo file Python (senza .py)
from reinforce_optimization import Actor, preprocess_state, INPUT_SIZE

def test_golden_stable_loop():
    print(">>> AVVIO TEST MULTI-EPISODIO: SOLO actor lap complete <<<")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    
    expert = Actor(INPUT_SIZE)
    try:
        expert.load_state_dict(torch.load("actor_GOLDEN_STABLE.pth"))
        expert.eval() # Fondamentale: blocca i pesi e garantisce un comportamento deterministico
    except Exception as e:
        print(f"Errore critico: Impossibile caricare il modello. {e}")
        return

    # Eseguiamo 10 episodi per verificare la costanza dei fallimenti
    for episode in range(1, 11):
        relaunch = False
        connected = False
        while not connected:
            try:
                env.reset(relaunch=relaunch)
                connected = True
            except:
                relaunch = True
                time.sleep(5.0)

        obs = env.client.S.d
        state = preprocess_state(obs)
        start_dist_raced = obs.get('distRaced', 0)
        done = False

        print(f"\n>>> INIZIO EPISODIO {episode}/10 <<<")

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_t = expert(state_t)
                
            env_action = action_t.numpy()[0].copy()
            env_action[3] = int(round(env_action[3] * 5.0 + 1.0))

            try:
                _, _, env_done, _ = env.step(env_action)
                if env_done: done = True
            except:
                print("Errore di connessione a TORCS, forzo riavvio.")
                done = True
                break

            obs = env.client.S.d
            if not obs: break
            
            track_index = obs.get('distFromStart', 0)
            current_dist_raced = obs.get('distRaced', 0) - start_dist_raced
            track_pos = obs.get('trackPos', 0.0)
            speed = obs.get('speedX', 0.0)
            angle = obs.get('angle', 0.0)

            # Log intensivo solo nell'area problematica (Corkscrew)
            if 2350 < track_index < 2750:
                print(f"Dist: {track_index:.1f}m | Vel: {speed:.1f} km/h | Posizione: {track_pos:.2f} | Angolo: {angle:.3f}")

            if current_dist_raced > 3610:
                print(f"--- EPISODIO {episode} CONCLUSO: GIRO COMPLETATO ---")
                done = True
                
            if abs(track_pos) > 2.1 or done:
                print(f"--- FALLIMENTO EPISODIO {episode}: SCHIANTO AL METRO {track_index:.1f} ---")
                done = True
                
            state = preprocess_state(obs)

    env.end()
    print("\n>>> TEST TERMINATO <<<")

if __name__ == "__main__":
    test_golden_stable_loop()