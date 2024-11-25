import torch


class ReplayBuffer:
    def __init__(
            self, 
            state_dim, 
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
                agent_view_size, 
                agent_view_size, 
                state_dim
            ), 
            dtype=torch.float32, 
            device=self.device
        )
        # actions (buffer_size, num_agents)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=self.device)
        # rewards (buffer_size, num_agents?)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        # next_states (buffer_size, agent_view, agent_view, state_channels)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        # dones (buffer_size,)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)

    def add(self, transition):
        state, action, reward, next_state, done = transition
        
        self.states[self.position].copy_(state if isinstance(state, torch.Tensor) else torch.as_tensor(state, device=self.device))
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device=self.device)
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states.index_select(0, indices),
            self.actions.index_select(0, indices),
            self.rewards.index_select(0, indices),
            self.next_states.index_select(0, indices),
            self.dones.index_select(0, indices)
        )

    def __len__(self):
        return self.size

