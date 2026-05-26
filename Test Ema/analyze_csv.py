import pandas as pd
import sys

def analyze(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset: {csv_path}")
        print(f"Rows: {len(df)}")
        print(f"Mean target_steer: {df['target_steer'].mean():.4f}")
        print(f"Mean angle: {df['angle'].mean():.4f}")
        print(f"Mean trackPos: {df['trackPos'].mean():.4f}")
        print(f"Mean speedX: {df['speedX'].mean():.4f}")
        
        # Check first 100 steps
        df_start = df.head(100)
        print(f"\nFirst 100 steps:")
        print(f"Mean target_steer: {df_start['target_steer'].mean():.4f}")
        print(f"Mean angle: {df_start['angle'].mean():.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze("dataset_laps.csv")
