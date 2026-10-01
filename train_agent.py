import numpy as np
from sb3_contrib import RecurrentPPO

from data_loader import baseline_flow_profile, failing_cycles
from hydraulic_env import HydraulicValveEnv

env = HydraulicValveEnv(baseline_profile=baseline_flow_profile, 
                        degraded_cycles=failing_cycles)

print("Initializing Recurrent PPO Agent...")

model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.003, # Aggressive learning rate
    n_steps=600, 
    batch_size=60
)

print("Starting emergency fast training run (100,000 steps)...")
model.learn(total_timesteps=100000)

model.save("prescriptive_valve_twin")
print("Training complete and model saved. Run evaluate_twin.py to see results.")