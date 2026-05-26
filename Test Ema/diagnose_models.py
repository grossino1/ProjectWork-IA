import torch
import os

def diagnose_model(path):
    if not os.path.exists(path):
        print(f"File {path} non trovato.")
        return None
    
    state_dict = torch.load(path, map_location='cpu')
    print(f"\n--- Diagnosi di: {path} ---")
    
    for key, weight in state_dict.items():
        if torch.isnan(weight).any():
            print(f"ERRORE: NaN rilevati in {key}")
        if torch.isinf(weight).any():
            print(f"ERRORE: Inf rilevati in {key}")
        
        max_val = weight.abs().max().item()
        mean_val = weight.abs().mean().item()
        print(f"Layer {key:20} | Max: {max_val:.4f} | Media: {mean_val:.4f}")
    
    return state_dict

def compare_models(dict1, dict2, name1, name2):
    print(f"\n--- Confronto tra {name1} e {name2} ---")
    for key in dict1.keys():
        if key in dict2:
            diff = (dict1[key] - dict2[key]).abs().mean().item()
            print(f"Differenza media in {key:20}: {diff:.6f}")

if __name__ == "__main__":
    # Confrontiamo l'attuale col Golden (che sappiamo essere sano)
    m1 = diagnose_model("Documents/torcs/gym_torcs/actor_lap_complete.pth")
    m2 = diagnose_model("Documents/torcs/gym_torcs/actor_GOLDEN_STABLE.pth")
    
    if m1 and m2:
        compare_models(m1, m2, "Active", "Golden")
