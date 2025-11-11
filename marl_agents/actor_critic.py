import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Defines the Policy Network (Actor) for a single agent.
    Input: Local State s_i + Self-Coordination Signal g_n^k
    Output: Action probabilities (3 discrete actions)
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, coord_signal):
        # Concatenate state and coordination signal
        x = torch.cat([state, coord_signal], dim=1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Use softmax for probability distribution over discrete actions
        return F.softmax(self.fc3(x), dim=-1)


class CriticNetwork(nn.Module):
    """
    Defines the centralized Critic Network.
    Input: All Agents' States (S) + All Agents' Actions (A)
    Output: Q-value
    """
    def __init__(self, full_state_dim, full_action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        # full_state_dim = N * state_dim
        # full_action_dim = N * action_dim
        input_dim = full_state_dim + full_action_dim 

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Output single Q-value

    def forward(self, states, actions):
        # Flatten all states and actions and concatenate
        x = torch.cat([states.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)