
### Updated `train.py`

from rl_agent.env_simulator import WMNSimEnv
from agent import MultiAgentSystem
from utils import ReplayBuffer
import numpy as np
import yaml

def train_loop(config):
    n_agents = config['n_agents']
    state_dim = config['state_dim']
    mas = MultiAgentSystem(n_agents, state_dim)
    env = WMNSimEnv(n_nodes=n_agents, max_steps=config['steps_per_episode'])

    for ep in range(config['episodes']):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            states = [obs['per_node'][i] for i in range(n_agents)]
            actions, _ = mas.select_actions(states)
            discrete = [int(np.random.choice(len(a), p=a)) for a in actions]
            next_obs, rewards, done, _ = env.step(discrete)
            total_reward += sum(rewards)
            obs = next_obs
        env.render()
        print(f"Episode {ep+1}: Total Reward={total_reward:.3f}")

if __name__ == '__main__':
    with open('../configs/hyperparams.yaml') as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)