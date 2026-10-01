import urllib.request
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
extract_dir = "hydraulic_data"
zip_path = "hydraulic_data.zip"

# Download and Extract
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
    print("Downloading dataset directly from UCI...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete!")

# Load labels
profile_path = os.path.join(extract_dir, "profile.txt")
targets = pd.read_csv(profile_path, sep='\t', header=None, 
                      names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable'])

# Load sensor data
fs1_path = os.path.join(extract_dir, "FS1.txt")
fs1_data = pd.read_csv(fs1_path, sep=r'\s+', header=None)

fs1_labeled = fs1_data.copy()
fs1_labeled['Valve_Condition'] = targets['Valve']

# Split data
optimal_cycles = fs1_labeled[fs1_labeled['Valve_Condition'] == 100].drop(columns=['Valve_Condition'])
failing_cycles = fs1_labeled[fs1_labeled['Valve_Condition'] == 73].drop(columns=['Valve_Condition'])

# Generate target baseline
baseline_flow_profile = optimal_cycles.mean(axis=0).values

# Visualization block is gated so it does not block the RL training script
if __name__ == "__main__":
    print(f"Optimal cycles (100): {len(optimal_cycles)}")
    print(f"Close to failure (73): {len(failing_cycles)}")