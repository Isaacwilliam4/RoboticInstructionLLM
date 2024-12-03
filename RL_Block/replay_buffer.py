import torch


class ReplayBuffer:
    def __init__(
            self, 
            state_size, 
            num_agents, 
            agent_view_size,
            device, 
            buffer_size=10_000
        ):
        self.device = device
        self.buffer_size = buffer_size
        self.position = 0
        self.size = 0
        self.num_agents = num_agents
        self.agent_view_size = agent_view_size

        # Need the size of the state to pre-allocate the space
        # states (buffer_size, agent_view, agent_view, state_channels)
        self.states = torch.zeros(
            (
                buffer_size, 
                *state_size,
            ), 
            dtype=torch.float32, 
            device=self.device
        )
        # actions (buffer_size, num_agents)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=self.device)
        # rewards (buffer_size, num_agents?)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        # next_states (buffer_size, agent_view, agent_view, state_channels)
        self.next_states = torch.zeros_like(self.states, dtype=torch.float32, device=self.device)
        # dones (buffer_size,)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)

    def add(self, state, actions, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = actions
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self, batch_size, time_steps):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states.index_select(0, indices),
            self.actions.index_select(0, indices),
            self.rewards.index_select(0, indices),
            self.next_states.index_select(0, indices),
            self.dones.index_select(0, indices)
        )

    def sample(self, batch_size, time_steps):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        time_offsets = torch.arange(time_steps, device=self.device).unsqueeze(0)  # Shape: (1, time_steps)

        start_indices = indices.unsqueeze(1) - time_offsets  # Shape: (batch_size, time_steps)
        start_indices = start_indices.clamp(min=0)  # Clamp to ensure no negative indices

        sampled_states = self.states[start_indices]  # Shape: (batch_size, time_steps, state_dim)
        sampled_actions = self.actions[start_indices]  # Shape: (batch_size, time_steps, ...)
        sampled_rewards = self.rewards[start_indices]  # Shape: (batch_size, time_steps)
        sampled_next_states = self.next_states[start_indices]  # Shape: (batch_size, time_steps, state_dim)
        sampled_dones = self.dones[start_indices]  # Shape: (batch_size, time_steps)

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones

    def __len__(self):
        return self.size

