import torch
import torch.optim as optim
from rl_agent.actor import Actor
from rl_agent.critic import CriticLSTM
from rl_agent.coordinator import Coordinator

class MultiAgentSystem:
    def __init__(self, n_agents, state_dim, device='cpu'):
        self.n = n_agents
        self.device = device
        self.actors = [Actor(state_dim).to(device) for _ in range(n_agents)]
        self.actor_opt = [optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.coord = Coordinator(n_agents, embed_dim=64).to(device)
        self.critic = CriticLSTM(input_dim=n_agents * 64).to(device)
        self.critic_target = CriticLSTM(input_dim=n_agents * 64).to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-2)

    def select_actions(self, states):
        xs = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        signals = torch.stack([self.actors[i].encode_signal(xs[i]) for i in range(self.n)])
        merged, weights = self.coord(signals)
        actions = []
        for i in range(self.n):
            probs = self.actors[i](xs[i])
            actions.append(probs.detach().cpu().numpy())
        return actions, weights.detach().cpu().numpy()
