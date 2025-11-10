import torch
import torch.nn as nn
import torch.nn.functional as F

class Coordinator(nn.Module):
    def __init__(self, num_agents, embed_dim):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.imp_mlp = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, signals):
        scores = self.imp_mlp(signals)  # shape: (num_agents, 1)
        weights = F.softmax(scores.view(1, -1), dim=1)
        weighted = torch.matmul(weights, signals)  # combine signals
        merged = weighted.repeat(self.num_agents, 1)
        return merged, weights
