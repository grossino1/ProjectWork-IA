import csv

def find_laps(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        last_timestamp = -1
        laps_found = 0
        line_count = 0
        session_starts = []
        
        for row in reader:
            line_count += 1
            try:
                ts = float(row[0])
                if ts < last_timestamp:
                    session_starts.append(line_count) # 1-based (not counting header)
                last_timestamp = ts
            except:
                continue
        
        print(f"Total lines (data): {line_count}")
        print(f"Session starts at lines: {session_starts}")
        return line_count, session_starts

if __name__ == "__main__":
    find_laps("Documents/torcs/gym_torcs/manualtot.csv")
