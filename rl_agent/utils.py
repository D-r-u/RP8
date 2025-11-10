import numpy as np
import torch

def to_tensor(x, device='cpu', dtype=torch.float32):
    return torch.tensor(x, device=device, dtype=dtype)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0
    def push(self, pkt):
        if len(self.buf) < self.capacity:
            self.buf.append(pkt)
        else:
            self.buf[self.pos] = pkt
            self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)
