import json
import time
import os
from datetime import datetime

def monitor_experiment(jsonl_file: str, refresh_interval: int = 30):
    """
    Monitor the progress of your experiment in real-time
    """
    print(f"Monitoring: {jsonl_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    start_time = time.time()
    
    try:
        while True:
            if os.path.exists(jsonl_file):
                # Count lines (entries)
                with open(jsonl_file, 'r') as f:
                    entries = sum(1 for line in f if line.strip())
                
                if entries > last_size:
                    # Read last few entries for status
                    success_count = 0
                    error_count = 0
                    models_seen = set()
                    
                    with open(jsonl_file, 'r') as f:
                        lines = f.readlines()
                        
                    # Analyze last 100 entries
                    for line in lines[-100:]:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                if entry['output']['status'] == 'success':
                                    success_count += 1
                                else:
                                    error_count += 1
                                models_seen.add(entry['input']['model_requested'])
                            except:
                                continue
                    
                    elapsed = time.time() - start_time
                    rate = entries / elapsed if elapsed > 0 else 0
                    
                    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                    
                    print(f"=== EXPERIMENT MONITOR ===")
                    print(f"File: {jsonl_file}")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"")
                    print(f"Total Entries: {entries}")
                    print(f"Rate: {rate:.2f} entries/sec")
                    print(f"Elapsed: {elapsed/3600:.1f} hours")
                    print(f"")
                    print(f"Recent 100 entries:")
                    print(f"  Success: {success_count}")
                    print(f"  Errors: {error_count}")
                    print(f"  Success Rate: {success_count/(success_count+error_count)*100:.1f}%")
                    print(f"")
                    print(f"Models active: {', '.join(models_seen)}")
                    print(f"")
                    print(f"File size: {os.path.getsize(jsonl_file)/1024/1024:.1f} MB")
                    
                    if len(lines) > 0:
                        try:
                            last_entry = json.loads(lines[-1])
                            print(f"Last processed: {last_entry['input']['image_path']}")
                        except:
                            pass
                    
                    last_size = entries
                
            else:
                print(f"Waiting for file: {jsonl_file}")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        monitor_experiment(sys.argv[1])
    else:
        print("Usage: python monitor_progress.py <jsonl_file>")