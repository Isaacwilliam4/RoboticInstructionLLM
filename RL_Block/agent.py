import torch
from torch.amp import GradScaler, autocast
from .neural_net import NeuralNetBlock
from .replay_buffer import ReplayBuffer


class AgentCoallition:
    def __init__(
            self,
            num_agents,
            agent_view_size,
            state_processor,
            action_space_size,
            time_steps=1,
            action_mask=None,
            train_every_n_iters= 10,
            gamma=0.99,
            batch_size=64,
            num_channels=6,
        ):
        self.scaler = GradScaler()
        self.gamma = gamma
        self.batch_size = batch_size
        self.time_steps_to_consider = time_steps
        self.train_every_n_iters = train_every_n_iters 
        self.episode_num = 0
        self.num_agents = num_agents
        self.steps_since_train = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.action_mask = action_mask 
        if action_mask is not None:
            self.action_mask = torch.tensor(self.action_mask)
            z = torch.zeros_like(self.action_mask)
            self.action_mask = torch.where(self.action_mask == 0, torch.full(z.size(), float('-inf')), z).to(self.device)

        self.state_processor = state_processor
        self.action_space_size = action_space_size

        # The "Queen" of the hive
        self.agent_a3c = NeuralNetBlock(
            input_space_size=(agent_view_size * agent_view_size * num_channels * num_agents), 
            output_space_size=self.action_space_size*num_agents,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            time_steps_to_RNN=1,
            action_mask=self.action_mask,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent_a3c.parameters(), lr=0.001)

        # The Queen's memory
        self.memory = ReplayBuffer(
            state_size=(agent_view_size, agent_view_size, num_channels * num_agents), 
            num_agents=self.num_agents,
            agent_view_size=agent_view_size,
            device=self.device
        )

   
    def train(self):
        self.episode_num += 1
        if self.episode_num % self.train_every_n_iters != 0: return 
        for _ in range(self.steps_since_train):
            self.learn_from_replay()
        self.steps_since_train = 0

    def remember(
            self, 
            state, 
            action, 
            reward, 
            next_state, 
            done, 
            instructions
        ):
        state = self.state_processor.multimodal_fusion(
            instructions, 
            state
        )
        next_state = self.state_processor.multimodal_fusion(
            instructions, 
            next_state
        )

        self.memory.add(
            state, 
            action, 
            torch.tensor(reward, device=self.device), 
            next_state, 
            torch.tensor(int(done), device=self.device), 
        )
        self.steps_since_train += 1

    def learn_from_replay(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.time_steps_to_consider)

        with autocast(device_type=self.device.type):
            next_values_y_hat, _ = self.agent_a3c(next_states)
            target_values = rewards + (self.gamma * next_values_y_hat * (1-dones))
            target_values = target_values.detach()
            values, action_probs = self.agent_a3c(states)
            critic_loss = torch.nn.functional.mse_loss(values, target_values)

            actions = actions.view(-1, 1)
            action_probs = action_probs.view(-1, self.action_space_size)
            action_log_probs = torch.log(action_probs.gather(1, actions.long()).squeeze())
            # action_log_probs = torch.log(action_probs[range(self.batch_size), actions.int()])
            advantages = (target_values - values).detach() 
            actor_loss = -torch.mean(action_log_probs*advantages)
       
            total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
       
        self.scaler.update()

    def get_actions(self, state, instruction):
        gated_state_representation = self.state_processor.multimodal_fusion(instruction, state)
        g_s_r = gated_state_representation.unsqueeze(0) # TODO add a lookback at previous steps here
        _, action_probs = self.agent_a3c(g_s_r.unsqueeze(0)) # This is so that it's a batch size of 1
        action_probs = action_probs.squeeze(dim=0)
        action_probs = action_probs.view(self.num_agents, -1)
        return torch.multinomial(action_probs, 1).squeeze(1)
