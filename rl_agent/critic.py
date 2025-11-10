import torch
import torch.nn as nn

class CriticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        last = out[:, -1, :]
        q = self.fc(last)
        return q.squeeze(-1), hx
