
### `rl_agent/env_simulator.py`

import numpy as np

class WMNSimEnv:
    """Simplified wireless mesh network simulator for debugging RL agents.
    Each node has tx_power (dBm), queue_length, delay, and neighbor count.
    The environment evolves using heuristic equations from the paper.
    """

    def __init__(self, n_nodes=10, max_steps=200, seed=42):
        np.random.seed(seed)
        self.n = n_nodes
        self.max_steps = max_steps
        self.cur_step = 0
        self.state_dim = 4
        self.tx_power = np.random.uniform(75, 90, size=n_nodes)
        self.queue_len = np.random.uniform(0, 1, size=n_nodes)
        self.delay = np.random.uniform(0.1, 0.3, size=n_nodes)
        self.neighbors = np.random.randint(3, 6, size=n_nodes)

    def reset(self):
        self.cur_step = 0
        self.tx_power = np.random.uniform(75, 90, size=self.n)
        self.queue_len = np.random.uniform(0, 1, size=self.n)
        self.delay = np.random.uniform(0.1, 0.3, size=self.n)
        self.neighbors = np.random.randint(3, 6, size=self.n)
        return self._get_obs()

    def step(self, actions):
        self.cur_step += 1
        # Apply actions (0: +0.5dBm, 1: -0.5dBm, 2: no change)
        delta = np.array([0.5 if a == 0 else (-0.5 if a == 1 else 0) for a in actions])
        self.tx_power = np.clip(self.tx_power + delta, 70, 97)

        # Update queue and delay based on pseudo network dynamics
        congestion_factor = np.mean(self.tx_power) / 100
        self.queue_len = np.clip(self.queue_len + np.random.uniform(-0.05, 0.05, self.n) + congestion_factor * 0.1, 0, 1)
        self.delay = np.clip(0.1 + 0.3 * (1 - self.tx_power / 100) + self.queue_len * 0.2, 0.05, 0.4)

        rewards = self._compute_rewards()
        obs = self._get_obs()
        done = self.cur_step >= self.max_steps
        return obs, rewards, done, {}

    def _get_obs(self):
        return {'per_node': [
            [self.tx_power[i] / 100, self.queue_len[i], self.delay[i], self.neighbors[i] / 10]
            for i in range(self.n)
        ]}

    def _compute_rewards(self):
        # Eq. (9) reward approximation
        w1, w2, w3 = 0.4, 0.4, 0.2
        sigma1, sigma2 = 0.1, 0.1
        Pt = self.tx_power / 100
        T = self.delay
        rewards = -w1 * np.log(Pt + 1e-5) + w2 * T + w3 * np.exp(-(Pt**2 / (2 * sigma1**2) + T**2 / (2 * sigma2**2)))
        return rewards.tolist()

    def render(self):
        print(f"Step {self.cur_step}: Avg Power={np.mean(self.tx_power):.2f} dBm | Avg Delay={np.mean(self.delay):.3f}s")


