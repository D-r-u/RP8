import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim, hidden=64, action_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc_out(x), dim=-1)

    def encode_signal(self, x):
        # Encodes state into signal for coordination
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
