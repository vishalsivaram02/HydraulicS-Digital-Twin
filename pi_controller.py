import numpy as np
import matplotlib.pyplot as plt
from data_loader import baseline_flow_profile, failing_cycles

# 1. Load the Data
target_flow = baseline_flow_profile
failing_flow = failing_cycles.iloc[0].values
simulated_flow = np.zeros(600)

# 2. PI Controller Tuning Parameters
Kp = 0.05  
Ki = 0.01  
integral = 0
valve_authority = 15.0  
u_prev = 0.0 # Tracks the mechanical position of the valve

# 3. Run the Closed-Loop Simulation
for i in range(600):
    base_flow = failing_flow[i]
    
    if i == 0:
        error = 0
        u = 0
    else:
        # Calculate error based on the previous step
        error = target_flow[i-1] - simulated_flow[i-1]
        integral += error
        integral = np.clip(integral, -50, 50)
        
        # Raw PI Control Equation
        raw_u = (Kp * error) + (Ki * integral)
        
        # SLEW RATE LIMITER: The physical valve can only move 5% per time step
        u = np.clip(raw_u, u_prev - 0.05, u_prev + 0.05)
        u = np.clip(u, 0, 1.0) # Absolute mechanical limits (0% to 100%)
        
    u_prev = u # Save current position for the next time step
    
    # Apply the control signal to the failing system physics
    simulated_flow[i] = base_flow + (u * valve_authority)
    
    # Add a tiny bit of realistic sensor noise
    if i > 0:
        simulated_flow[i] += np.random.normal(0, 0.03)

# 4. Generate the Final Review Plot
plt.figure(figsize=(10, 5))
plt.plot(target_flow, label='Optimal Baseline Setpoint (100)', color='green', linewidth=2)
plt.plot(failing_flow, label='Close to Failure (73) - Uncompensated', color='red', alpha=0.4)
plt.plot(simulated_flow, label='PI Compensated Flow (Digital Twin)', color='blue', linewidth=2)

plt.title('Digital Twin Validation: Closed-Loop PI Compensation vs. Internal Leakage')
plt.xlabel('Time Step (10 Hz = 600 steps/min)')
plt.ylabel('Flow Rate (l/min)')
plt.legend()
plt.grid(True)

plt.savefig("final_presentation_plot.png", dpi=300)
print("Success! Plot saved as 'final_presentation_plot.png'.")