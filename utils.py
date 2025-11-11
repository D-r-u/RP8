from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """Stores a trajectory transition."""
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Unpack the sampled transitions
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in indices])
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)