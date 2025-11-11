import torch
import torch.optim as optim
from .actor_critic import ActorNetwork, CriticNetwork
from utils import ReplayBuffer
import torch.nn.functional as F

class SelfCoordinatedMADDPG:
    """
    Implements the Dynamic Self-Coordinated Topology Optimization Algorithm (Algorithm 1).
    Uses a MADDPG-inspired architecture with an explicit self-coordination step.
    """
    def __init__(self, env, hidden_dim=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        self.env = env
        self.N = env.N
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.gamma = gamma # Discount factor
        self.tau = tau # Soft update rate
        self.buffer = ReplayBuffer(100000)

        # 1. Initialize Networks for N Agents
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        # Actor input must include local state (state_dim) + coordination signal (N*state_dim)
        actor_input_dim = self.state_dim + self.N * self.state_dim 
        
        # Critic input is all states (N * state_dim) + all actions (N * action_dim)
        full_state_dim = self.N * self.state_dim
        full_action_dim = self.N * self.action_dim

        for i in range(self.N):
            # Online Networks
            actor = ActorNetwork(actor_input_dim, hidden_dim, self.action_dim)
            critic = CriticNetwork(full_state_dim, full_action_dim, hidden_dim)
            self.actors.append(actor)
            self.critics.append(critic)
            
            # Target Networks (copies of online networks for stability)
            self.target_actors.append(ActorNetwork(actor_input_dim, hidden_dim, self.action_dim))
            self.target_critics.append(CriticNetwork(full_state_dim, full_action_dim, hidden_dim))
            self.target_actors[i].load_state_dict(actor.state_dict())
            self.target_critics[i].load_state_dict(critic.state_dict())

            # Optimizers
            self.actor_optimizers.append(optim.Adam(self.actors[i].parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(self.critics[i].parameters(), lr=lr_critic))

    def _get_coordination_signals(self, states):
        """
        Implements the self-coordination mechanism (Steps 4-7, Algorithm 1).
        The coordination signal for agent i is the concatenation of ALL agents' states.
        
        :param states: A tensor of shape (Batch_Size, N, state_dim)
        :return: A tensor of shape (Batch_Size, N, N * state_dim)
        """
        
        # 1. Flatten the global state for the batch: (Batch_Size, N * state_dim)
        # This represents the full coordination signal for the entire network.
        global_state_flat = states.flatten(start_dim=1) 
        
        # 2. Prepare the signal for all N agents in the batch.
        # We want to broadcast this flat global state (1 signal) to all N agents.
        # Shape: (Batch_Size, N, N * state_dim)
        
        # Unsqueeze adds a dimension for the agents (N)
        # Repeat copies the signal N times across this new dimension
        coord_signal = global_state_flat.unsqueeze(1).repeat(1, self.N, 1)

        # The coordination signal must have shape (Batch_Size * N, N * state_dim)
        # to be concatenated with the local state (Batch_Size * N, state_dim)
        # in the Actor Network forward pass.
        
        return coord_signal

    def choose_actions(self, states):
        """Decentralized Execution: Each agent chooses an action."""
        actions = []
        
        # 1. Convert NumPy state (N, state_dim) to tensor and add batch dim (1, N, state_dim)
        states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        
        # 2. Get the coordination signal: (1, N, N*state_dim)
        # Note: The fixed _get_coordination_signals returns the correct batch shape.
        coord_signals_batch = self._get_coordination_signals(states_tensor)

        for i in range(self.N):
            # 3. Extract single-batch tensors for Actor input
            # local_state: (1, state_dim) - Slice out the specific agent's state
            local_state = states_tensor[:, i, :] 
            
            # coord_signal: (1, N*state_dim) - Slice out the specific agent's signal
            coord_signal = coord_signals_batch[:, i, :]
            
            # 4. Forward Pass to Actor
            # Both tensors are now 2D (batch_size=1, feature_size)
            probs = self.actors[i](local_state, coord_signal)
            
            # Sample action from the distribution
            action = torch.multinomial(probs, 1).item()
            actions.append(action)
        
        return actions

    def learn(self, batch_size):
        """Centralized Training: Update all networks."""
        if len(self.buffer) < batch_size:
            return 
            
        # Sample a batch from replay buffer (S, A, R, S_next)
        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        
        # Convert to Tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # 1. CRITIC UPDATE (Minimize L(Q))
        
        # a) Get next actions A_next from TARGET actors (Decentralized target policy)
        next_actions = []
        for i in range(self.N):
            # Get coordination signal for next state
            next_coord_signals = self._get_coordination_signals(next_states)
            local_next_state = next_states[:, i, :]
            next_coord_signal = next_coord_signals[:, i, :]
            
            # Get action probabilities from Target Actor
            probs = self.target_actors[i](local_next_state, next_coord_signal)
            
            # Convert probabilities to a one-hot vector (for Q-network input)
            one_hot_action = F.one_hot(probs.argmax(dim=1), self.action_dim)
            next_actions.append(one_hot_action)
        
        A_next = torch.stack(next_actions, dim=1) # Shape: (Batch, N, action_dim)
        
        # b) Calculate Target Q-Value y^l = r + gamma * Q_target(S_next, A_next)
        for i in range(self.N):
            # Calculate Q-target for agent i
            Q_target_next = self.target_critics[i](next_states, A_next)
            y_target = rewards + self.gamma * Q_target_next

            # c) Calculate Current Q-Value Q_online(S, A)
            # Convert sampled discrete action index (actions) to one-hot for Q-input
            A_online = F.one_hot(actions, self.action_dim)
            Q_online = self.critics[i](states, A_online)

            # d) Critic Loss (MSE) and Update
            critic_loss = F.mse_loss(Q_online, y_target.detach())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # 2. ACTOR UPDATE (Maximize Q_online(S, A) - Policy Gradient)
        
        # We need the current actions according to the current online policy
        current_actions_for_critics = []
        
        # --- FIX STARTS HERE ---
        # Calculate coordination signals once for the whole batch
        coord_signals_batch = self._get_coordination_signals(states) 
        # --- FIX ENDS HERE ---
        
        for i in range(self.N):
            
            # --- FIX STARTS HERE ---
            # Use the calculated batch signals and slice for agent i
            local_state = states[:, i, :] # Shape: (Batch, state_dim)
            coord_signal = coord_signals_batch[:, i, :] # Shape: (Batch, N*state_dim)
            
            # Get action probabilities from Online Actor
            probs = self.actors[i](local_state, coord_signal)
            # --- FIX ENDS HERE ---
            
            # Choose the action that maximizes Q (for the policy gradient)
            # This is correct as is
            one_hot_action = F.one_hot(probs.argmax(dim=1), self.action_dim) 
            current_actions_for_critics.append(one_hot_action)

        
        A_online_policy = torch.stack(current_actions_for_critics, dim=1)

        for i in range(self.N):
            # Actor Loss = -mean(Q_online(S, A_online_policy))
            # We want to maximize Q, so we minimize -Q
            actor_loss = -self.critics[i](states, A_online_policy).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
        # 3. SOFT UPDATE TARGET NETWORKS (Step 22)
        self._soft_update_targets()

    def _soft_update_targets(self):
        """Update target networks using soft update (Q_t = tau*Q_o + (1-tau)*Q_t)."""
        for i in range(self.N):
            # Soft update for Actor
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
            # Soft update for Critic
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)