import torch
import torch.optim as optim
from .actor_critic import ActorNetwork, CriticNetwork
from utils import ReplayBuffer
import torch.nn.functional as F
import numpy as np

class SelfCoordinatedMADDPG:
    """
    Self-Coordinated Multi-Agent DDPG with Gumbel-Softmax for discrete actions
    and normalized state inputs for stability.
    """

    def __init__(self, env, hidden_dim=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        self.env = env
        self.N = env.N
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(100000)

        # Running mean/std for state normalization
        self.state_mean = np.zeros(self.state_dim)
        self.state_std = np.ones(self.state_dim)
        self.eps = 1e-8

        # Initialize networks
        self.actors, self.critics = [], []
        self.target_actors, self.target_critics = [], []
        self.actor_optimizers, self.critic_optimizers = [], []

        actor_input_dim = self.state_dim + self.N * self.state_dim
        full_state_dim = self.N * self.state_dim
        full_action_dim = self.N * self.action_dim

        for i in range(self.N):
            actor = ActorNetwork(actor_input_dim, hidden_dim, self.action_dim)
            critic = CriticNetwork(full_state_dim, full_action_dim, hidden_dim)
            self.actors.append(actor)
            self.critics.append(critic)

            target_actor = ActorNetwork(actor_input_dim, hidden_dim, self.action_dim)
            target_critic = CriticNetwork(full_state_dim, full_action_dim, hidden_dim)
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))

    # --- Normalization utilities ---
    def _update_state_stats(self, states):
        mean = np.mean(states, axis=0)
        std = np.std(states, axis=0)
        self.state_mean = 0.99 * self.state_mean + 0.01 * mean
        self.state_std = 0.99 * self.state_std + 0.01 * std

    def _normalize_states(self, states):
        return (states - self.state_mean) / (self.state_std + self.eps)

    # --- Gumbel-Softmax Sampling ---
    def _gumbel_softmax_sample(self, logits, tau=0.8):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        y = (logits + gumbel_noise) / tau
        return F.softmax(y, dim=-1)

    # --- Self-coordination signal ---
    def _get_coordination_signals(self, states, mode="soft"):
        global_state_flat = states.flatten(start_dim=1)
        coord_signal = global_state_flat.unsqueeze(1).repeat(1, self.N, 1)

        if mode == "diag0":  # Remove self influence
            for i in range(self.N):
                coord_signal[:, i, i * self.state_dim:(i + 1) * self.state_dim] = 0

        elif mode == "diag1":  # Only self
            new_signal = torch.zeros_like(coord_signal)
            for i in range(self.N):
                new_signal[:, i, i * self.state_dim:(i + 1) * self.state_dim] = \
                    global_state_flat[:, i * self.state_dim:(i + 1) * self.state_dim]
            coord_signal = new_signal

        elif mode == "binary":  # Example: limited adjacency (ring structure)
            new_signal = torch.zeros_like(coord_signal)
            for i in range(self.N):
                left = (i - 1) % self.N
                right = (i + 1) % self.N
                for neighbor in [left, right]:
                    new_signal[:, i, neighbor * self.state_dim:(neighbor + 1) * self.state_dim] = \
                        global_state_flat[:, neighbor * self.state_dim:(neighbor + 1) * self.state_dim]
            coord_signal = new_signal

        # 'soft' = default (no change)
        return coord_signal


    # --- Choose actions ---
    def choose_actions(self, states):
        actions = []
        self._update_state_stats(states)
        norm_states = self._normalize_states(states)

        states_tensor = torch.tensor(norm_states, dtype=torch.float32).unsqueeze(0)
        coord_signals_batch = self._get_coordination_signals(states_tensor)

        for i in range(self.N):
            local_state = states_tensor[:, i, :]
            coord_signal = coord_signals_batch[:, i, :]
            probs = self.actors[i](local_state, coord_signal)
            action = torch.multinomial(probs, 1).item()
            actions.append(action)
        return actions

    # --- Learning ---
    def learn(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        self._update_state_stats(states)
        norm_states = self._normalize_states(states)
        norm_next_states = self._normalize_states(next_states)

        states = torch.tensor(norm_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(norm_next_states, dtype=torch.float32)

        # 1. CRITIC UPDATE
        next_actions = []
        next_coord_signals = self._get_coordination_signals(next_states)
        for i in range(self.N):
            local_next_state = next_states[:, i, :]
            coord_signal_next = next_coord_signals[:, i, :]
            logits, _ = self.target_actors[i](local_next_state, coord_signal_next, return_logits=True)
            soft_action = self._gumbel_softmax_sample(logits, tau=0.8)
            next_actions.append(soft_action)
        A_next = torch.stack(next_actions, dim=1)

        for i in range(self.N):
            Q_target_next = self.target_critics[i](next_states, A_next)
            y_target = rewards + self.gamma * Q_target_next

            # One-hot encode current actions
            A_online = F.one_hot(actions, self.action_dim).float()
            Q_online = self.critics[i](states, A_online)

            critic_loss = F.mse_loss(Q_online, y_target.detach())
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # 2. ACTOR UPDATE (clean isolated per-agent version)
        coord_signals_batch = self._get_coordination_signals(states)

        for i in range(self.N):
            # Rebuild actions for all agents — only agent i’s action keeps gradients
            current_actions_for_critics = []
            for j in range(self.N):
                local_state = states[:, j, :]
                coord_signal = coord_signals_batch[:, j, :]
                logits, _ = self.actors[j](local_state, coord_signal, return_logits=True)
                soft_action = self._gumbel_softmax_sample(logits, tau=0.8)
                if j != i:
                    soft_action = soft_action.detach()  # detach others to isolate gradient
                current_actions_for_critics.append(soft_action)

            # Stack them to form the full joint action
            A_online_policy = torch.stack(current_actions_for_critics, dim=1)

            # Compute actor loss for agent i only
            actor_loss = -self.critics[i](states, A_online_policy).mean()

            # Standard optimizer step
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # 3. SOFT UPDATE TARGETS
        self._soft_update_targets()

    def _soft_update_targets(self):
        for i in range(self.N):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
