import json
import csv
import os
import glob

def merge_laps_to_csv(input_dir, output_file):
    json_files = sorted(glob.glob(os.path.join(input_dir, "lap*.json")))
    
    if not json_files:
        print(f"Nessun file lap*.json trovato in {input_dir}")
        return

    # Headers compatibili con dataset_bot_154_clean.csv e ExpertModel
    headers = [
        "timestamp", "target_steer", "target_accel", "target_brake", "target_gear",
        "angle", "gear", "rpm", "speedX", "speedY", "speedZ"
    ]
    # Aggiungiamo track sensors
    for i in range(19):
        headers.append(f"track_{i}")
    
    headers.append("trackPos")
    
    # Aggiungiamo wheel spin velocity
    for i in range(4):
        headers.append(f"wheelSpinVel_{i}")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        total_steps = 0
        cumulative_time = 0.0
        delta_t = 0.02 # Assumiamo 50Hz
        
        for json_file in json_files:
            print(f"Elaborazione {json_file}...")
            with open(json_file, 'r') as jf:
                try:
                    data = json.load(jf)
                except json.JSONDecodeError as e:
                    print(f"Errore nel file {json_file}: {e}")
                    continue
                
                last_gear = 1
                for entry in data:
                    sensors = entry.get('sensors', {})
                    inputs = entry.get('input', {})
                    
                    # 1. CLEANING: Accel/Brake Conflict
                    # In TORCS, you shouldn't press both. Priority to brake for safety.
                    accel = inputs.get('accel', 0.0)
                    brake = inputs.get('brake', 0.0)
                    if brake > 0.1:
                        accel = 0.0
                    
                    # 2. SMOOTHING: Gear shifts
                    target_gear = inputs.get('gear', 1)
                    if target_gear == 0: target_gear = 1 # Avoid neutral
                    
                    # 3. HEURISTICS: Estimate Angle and trackPos
                    # track sensors: 0 is far left (-45 deg), 18 is far right (45 deg), 9 is straight (0 deg)
                    track = sensors.get('track', [0.0]*19)
                    if len(track) < 19:
                        track.extend([0.0] * (19 - len(track)))
                    
                    # Estimate angle: difference between left-side (0-4) and right-side (14-18)
                    left_sum = sum(track[0:4])
                    right_sum = sum(track[15:19])
                    estimated_angle = (right_sum - left_sum) / 500.0 # Heuristic scaling
                    estimated_angle = max(-0.5, min(0.5, estimated_angle))
                    
                    # Estimate trackPos: compare track[0] (left edge) and track[18] (right edge)
                    # If track[0] is small, we are near the left edge (trackPos -> 1.0)
                    # If track[18] is small, we are near the right edge (trackPos -> -1.0)
                    t0, t18 = track[0], track[18]
                    estimated_trackPos = 0.0
                    if (t0 + t18) > 0:
                        estimated_trackPos = (t18 - t0) / (t0 + t18 + 0.1)
                    
                    # Mapping JSON -> CSV
                    row = [
                        cumulative_time,
                        inputs.get('steer', 0.0),
                        accel,
                        brake,
                        target_gear,
                        estimated_angle, 
                        target_gear, # gear (as feature)
                        sensors.get('rpm', 0.0),
                        sensors.get('speed', 0.0), # speed as speedX
                        0.0, # speedY
                        0.0, # speedZ
                    ]
                    
                    row.extend(track[:19])
                    row.append(estimated_trackPos)
                    
                    # WheelSpinVel
                    row.extend([0.0, 0.0, 0.0, 0.0])
                    
                    writer.writerow(row)
                    cumulative_time += delta_t
                    total_steps += 1
                    
    print(f"Unione completata! Creato {output_file} con {total_steps} record.")

if __name__ == "__main__":
    merge_laps_to_csv("Documents/torcs/laps", "Documents/torcs/gym_torcs/dataset_laps.csv")
