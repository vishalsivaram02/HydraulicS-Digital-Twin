import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO

from data_loader import baseline_flow_profile, failing_cycles
from hydraulic_env import HydraulicValveEnv

env = HydraulicValveEnv(baseline_profile=baseline_flow_profile, 
                        degraded_cycles=failing_cycles)

print("Loading trained Prescriptive Twin...")
model = RecurrentPPO.load("prescriptive_valve_twin")

print("Running inference...")
obs, info = env.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

simulated_flows = []
terminated = False

while not terminated:
    action, lstm_states = model.predict(
        obs, 
        state=lstm_states, 
        episode_start=episode_starts, 
        deterministic=True
    )
    
    obs, reward, terminated, truncated, info = env.step(action)
    episode_starts = np.zeros((1,), dtype=bool)
    simulated_flows.append(info["simulated_flow"])

plt.figure(figsize=(10, 5))
plt.plot(baseline_flow_profile, label='Optimal Baseline Setpoint (100)', color='green', linewidth=2)
plt.plot(failing_cycles.iloc[0].values, label='Close to Failure (73) - Uncompensated', color='red', alpha=0.4)
plt.plot(simulated_flows, label='RL Compensated Flow (Prescriptive Twin)', color='blue', linewidth=2)
plt.title('Digital Twin Validation: RL Compensated vs. Failing Flow')
plt.xlabel('Time Step (10 Hz = 600 steps/min)')
plt.ylabel('Flow Rate (l/min)')
plt.legend()
plt.grid(True)

plt.savefig("compensation_results.png", dpi=300)
print("Success! Plot saved as 'compensation_results.png'. Open it in the IDE sidebar to view.")