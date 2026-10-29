import numpy as np
import gymnasium as gym

class MinesweeperEnv(gym.Env):
    """Minimal skeletal Gymnasium environment stub for dependency resolution."""

    def __init__(self, width: int = 8, height: int = 8, num_mines: int = 10):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(11, height, width), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(width * height)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)