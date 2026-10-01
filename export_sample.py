import os
import pandas as pd

extract_dir = "hydraulic_data"

# Load the target labels
profile_path = os.path.join(extract_dir, "profile.txt")
targets = pd.read_csv(profile_path, sep='\t', header=None, 
                      names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable'])

# Load the FS1 sensor data
fs1_path = os.path.join(extract_dir, "FS1.txt")
fs1_data = pd.read_csv(fs1_path, sep=r'\s+', header=None)

# Combine them for the faculty sample (showing just the first 10 time steps of the 600)
faculty_sample = fs1_data.iloc[:, :10].copy()
faculty_sample.columns = [f'Flow_t{i}' for i in range(10)]
faculty_sample['Target_Valve_Condition'] = targets['Valve']

# Map the numeric labels to readable text for the review panel
condition_map = {100: 'Optimal', 90: 'Small Lag', 80: 'Severe Lag', 73: 'Failing'}
faculty_sample['Condition_Status'] = faculty_sample['Target_Valve_Condition'].map(condition_map)

# Save the first 20 rows to a clean CSV
faculty_sample.to_csv("full_faculty_dataset.csv", index=False)
print("Saved 'faculty_dataset_sample.csv' to your project folder!")