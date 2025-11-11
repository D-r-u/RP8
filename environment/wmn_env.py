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
        reward, total_delay, total_power = self._calculate_reward(next_states)
        
        # 4. Done condition (simplified to fixed number of steps)
        done = False 
        
        # We return a single global reward for centralized training
        return next_states, reward, done, {'total_delay': total_delay, 'total_power': total_power}
    
    def _calculate_reward(self, states):
        """Implements the paper's custom reward function (R(S))."""
        
        # Aggregate Power and Delay from the states
        # Assuming the state is [Power, Delay, Interference, SINR_min]
        P_t = states[:, 0].sum() # Total Power Consumption
        T_S = states[:, 1].mean() # Average E2E Delay (Approximation)

        # Hyper-parameters from the paper (or chosen values)
        w1, w2, w3 = 0.4, 0.4, 0.2
        
        # --- ADJUSTED TARGETS BASED ON OBSERVATION ---
        # Observed P_t ~ 200, so mu1 is set to log(200) ~ 5.3
        # Observed T_S ~ 25, so mu2 is set to 25.0
        mu1, mu2 = 5.3, 25.0 
        # --- END ADJUSTED TARGETS ---
        
        sigma1, sigma2 = 0.1, 0.1
        
        # Ensure total power is non-zero before taking log
        log_Pt = np.log(P_t) if P_t > 1e-6 else np.log(1e-6)

        # Term 1: Minimize Power (Negative Log)
        term1 = -w1 * log_Pt
        
        # Term 2: Minimize Delay (Linear)
        term2 = -w2 * T_S
        
        # Term 3: Gaussian Fitness (Encourage balance near the targets)
        power_norm = (log_Pt - mu1) / sigma1
        delay_norm = (T_S - mu2) / sigma2
        term3 = w3 * np.exp(-0.5 * (power_norm**2 + delay_norm**2))
        
        reward = term1 + term2 + term3
        
        return reward, T_S, P_t