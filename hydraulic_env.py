import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HydraulicValveEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, baseline_profile, degraded_cycles):
        super(HydraulicValveEnv, self).__init__()
        
        self.baseline = baseline_profile
        self.degraded_cycles = degraded_cycles
        self.max_steps = 600 
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.current_step = 0
        self.current_cycle_data = None
        self.previous_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_action = 0.0
        
        random_index = np.random.randint(0, len(self.degraded_cycles))
        self.current_cycle_data = self.degraded_cycles.iloc[random_index].values
        
        obs = np.array([
            self.baseline[self.current_step],
            self.current_cycle_data[self.current_step],
            self.previous_action
        ], dtype=np.float32)
        
        return obs, {}

    def step(self, action):
        action_val = float(action[0])
        
        raw_flow = self.current_cycle_data[self.current_step]
        # Massive authority overhead so the agent can overcome the deep leak
        simulated_flow = raw_flow + (action_val * 15.0) 
        
        target_flow = self.baseline[self.current_step]
        reward = self._calculate_reward(target_flow, simulated_flow, action_val)
        
        self.current_step += 1
        self.previous_action = action_val
        
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        
        if not terminated:
            next_obs = np.array([
                self.baseline[self.current_step],
                self.current_cycle_data[self.current_step], 
                self.previous_action
            ], dtype=np.float32)
        else:
            next_obs = np.zeros(3, dtype=np.float32)
            
        info = {"simulated_flow": simulated_flow}
        
        return next_obs, reward, terminated, truncated, info
        
    def _calculate_reward(self, target, actual, action):
        error = target - actual
        tracking_penalty = -(error ** 2)
        
        action_delta = abs(action - self.previous_action)
        # Reduced jitter penalty to allow for aggressive corrections
        jitter_penalty = -0.1 * action_delta
        
        reward = tracking_penalty + jitter_penalty
        return float(reward)