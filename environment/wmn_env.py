import numpy as np

class WMNEnvironment:
    """
    Simulates the Wireless Mesh Network (WMN) environment.
    """
    def __init__(self, num_nodes=10, max_power=100, delta_s=10, grid_size=500):
        self.N = num_nodes
        self.MAX_POWER = max_power
        self.DELTA_S = delta_s
        self.GRID_SIZE = grid_size
        self.nodes = self._initialize_nodes()
        self.state_dim = 4 # [Power, Delay, Interference, SINR_min]
        self.action_dim = 3 # 0: Inc, 1: Dec, 2: Keep

    def _initialize_nodes(self):
        # Assign random (x, y) coordinates and initial power
        nodes = []
        for i in range(self.N):
            nodes.append({
                'id': i,
                'pos': np.random.rand(2) * self.GRID_SIZE,
                'power': self.MAX_POWER / 2, # Start with half power
            })
        return nodes

    def reset(self):
        """Resets the environment and returns initial state for all agents."""
        self.nodes = self._initialize_nodes()
        return self.get_full_state()

    def get_full_state(self):
        """Calculates the state vector for all N agents."""
        # Simplified placeholder calculation for state elements
        states = []
        for i in range(self.N):
            power = self.nodes[i]['power']
            delay = np.random.rand() * 50 # Placeholder
            interference = self._calculate_interference(i)
            sinr_min = self._calculate_sinr_min(i)
            
            # State: [Power, Delay, Interference, SINR_min]
            states.append(np.array([power, delay, interference, sinr_min]))
        
        return np.array(states)

    def _calculate_interference(self, i):
        """Simplified interference calculation (sum of received power from others)."""
        interference = 0
        for j in range(self.N):
            if i != j:
                distance = np.linalg.norm(self.nodes[i]['pos'] - self.nodes[j]['pos'])
                # Simplified Path Loss Model: P_received = P_tx / (distance^a)
                # Assume a path loss exponent 'a'
                PL_EXPONENT = 3.0 
                P_rx = self.nodes[j]['power'] / (distance**PL_EXPONENT)
                interference += P_rx
        return interference
    
    def _calculate_sinr_min(self, i):
        """Simplified SINR for the worst neighbor link (or a baseline)."""
        # In a real WMN, this requires defining traffic flows and desired links.
        # Here, we'll use a placeholder or simplify to a minimal link quality.
        return np.random.rand() * 20 # SINR in dB (Placeholder)


    def step(self, actions):
        """
        Applies actions, updates node power, and returns next state, reward, and done status.
        :param actions: List of actions [a_1, a_2, ..., a_N] (0:Inc, 1:Dec, 2:Keep)
        """
        assert len(actions) == self.N, "Action list must match number of nodes."
        
        # 1. Update Power (Action Execution)
        for i in range(self.N):
            a = actions[i]
            current_power = self.nodes[i]['power']
            
            if a == 0: # Increase power
                new_power = min(current_power + self.DELTA_S, self.MAX_POWER)
            elif a == 1: # Decrease power
                new_power = max(current_power - self.DELTA_S, 0)
            else: # Keep power
                new_power = current_power
                
            self.nodes[i]['power'] = new_power
            
        # 2. Get Next State
        next_states = self.get_full_state()
        
        # 3. Calculate Reward (The complex part - based on paper's function)
        reward, total_delay, total_power = self._calculate_reward(next_states, mode="balanced")
        
        # 4. Done condition (simplified to fixed number of steps)
        done = False 
        
        # We return a single global reward for centralized training
        return next_states, reward, done, {'total_delay': total_delay, 'total_power': total_power}
    
    def _calculate_reward(self, states, mode="balanced"):
        """
        Computes the reward for the current environment state.

        mode:
            "balanced"  → stable, power–delay tradeoff reward (recommended)
            "gaussian"  → purely fitness-based reward for comparison
        """
        # Extract metrics
        P_t = states[:, 0].sum()  # total power
        T_S = states[:, 1].mean() # average delay

        # Avoid numerical issues
        P_t = max(P_t, 1e-6)

        # Reward hyperparameters
        w1, w2, w3 = 0.4, 0.4, 0.2
        mu1, mu2 = 5.3, 25.0
        sigma1, sigma2 = 0.8, 6.0

        # Compute logarithmic term safely
        log_Pt = np.log(P_t + 1e-6)

        # --- OPTION 1: Balanced Reward (default) ---
        if mode == "balanced":
            # Primary reward components
            term1 = -w1 * np.log(P_t + 1.0)  # penalize excessive power, but stable near zero
            term2 = -w2 * T_S                # penalize large delay

            # Gaussian fitness to encourage target balance
            power_norm = (log_Pt - mu1) / sigma1
            delay_norm = (T_S - mu2) / sigma2
            term3 = w3 * np.exp(-0.5 * (power_norm**2 + delay_norm**2))

            reward = term1 + term2 + term3

            # --- NEW ADDITION: Penalize zero or too-low power ---
            if P_t < 50:  # depends on N and MAX_POWER
                reward -= 100  # large penalty for communication breakdown

        # --- OPTION 2: Gaussian-Only Reward ---
        elif mode == "gaussian":
            power_norm = (log_Pt - mu1) / sigma1
            delay_norm = (T_S - mu2) / sigma2
            reward = np.exp(-0.5 * (power_norm**2 + delay_norm**2))
            # normalize range ~[0,1]
            reward = float(reward)
            # scale to match approximate magnitude of other version
            reward *= 100

        else:
            raise ValueError("Unknown reward mode. Use 'balanced' or 'gaussian'.")

        return reward, T_S, P_t
