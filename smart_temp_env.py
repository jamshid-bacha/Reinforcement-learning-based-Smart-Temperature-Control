import numpy as np
import gym
from gym import spaces

class SmartTemperatureEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # State: [current_temp, outside_temp]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -30.0]),
            high=np.array([50.0, 50.0]),
            dtype=np.float32
        )

        # Action: heating/cooling power [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.target_temp = 22.0
        self.max_steps = 200
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_temp = np.random.uniform(15, 30)
        self.outside_temp = np.random.uniform(-5, 35)
        self.step_count = 0

        return np.array([self.current_temp, self.outside_temp], dtype=np.float32), {}

    def step(self, action):
        action = np.clip(action, -1, 1)[0]

        # Dynamics
        heat_effect = action * 2.0
        outside_effect = 0.1 * (self.outside_temp - self.current_temp)

        self.current_temp += heat_effect + outside_effect

        # Reward: stay near target + penalize energy use
        temp_error = abs(self.current_temp - self.target_temp)
        reward = -temp_error / 10.0 - 0.01 * (action ** 2)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        state = np.array([
            self.current_temp / 50.0,
            self.outside_temp / 50.0
        ], dtype=np.float32)

        return state, reward, done, False, {}

    def render(self):
        print(f"Temp: {self.current_temp:.2f}°C | Outside: {self.outside_temp:.2f}°C")