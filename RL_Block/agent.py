import random
import torch
from torch import nn
import numpy as np
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
            action_mask=None,
            train_every_n_iters= 10,
            gamma=0.99,
            batch_size=64,
        ):
        self.scaler = GradScaler()
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_every_n_iters = train_every_n_iters 
        self.episode_num = 0

        self.action_mask = action_mask if action_mask is not None else [1]*action_space_size
        self.action_mask = torch.tensor(self.action_mask)
        z = torch.zeros_like(self.action_mask)
        self.action_mask = torch.where(self.action_mask == 0, torch.full(z.size(), float('-inf')), z)

        # The worker bees
        self.num_agents = num_agents
        self.coallition = [Agent(i) for i in range(num_agents)]
 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_processor = state_processor
        self.action_space_size = action_space_size

        # The "Queen" of the hive
        self.agent_a3c = NeuralNetBlock(
            input_space_size=self.state_processor.output_size, 
            output_space_size=self.action_space_size,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            time_steps_to_RNN=1,
            action_mask=self.action_mask.to(self.device)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent_a3c.parameters(), lr=0.001)

        # The Queen's memory
        self.memory = ReplayBuffer(
            state_dim=self.state_processor.output_size, 
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

        self.memory.add(state, action, reward, next_state, done)
        self.steps_since_train += 1

    def learn_from_replay(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with autocast(device_type=self.device.type):
            next_values_y_hat, _ = self.agent_a3c(next_states)
            target_values = rewards + (self.gamma * next_values_y_hat * (1-dones))
            values, action_probs = self.agent_a3c(states)
            critic_loss = torch.nn.functional.mse_loss(values, target_values)

            action_log_probs = torch.log(action_probs.gather(1, actions.long().unsqueeze(1)).squeeze())
            # action_log_probs = torch.log(action_probs[range(self.batch_size), actions.int()])
            advantages = (target_values - values).detach() 
            actor_loss = -torch.mean(action_log_probs*advantages)
       
            total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
       
        self.scaler.update()

    def get_actions(self, state, instruction):
        return np.random.randint(
            0, 
            self.action_space_size,
            size=(self.num_agents,)
        )
        # return [random.randint(0, 7) for _ in range(self.num_agents)]

class Agent():
    def __init__(
            self, 
            id,
        ):
        self.id = id
       
    def get_action(self, state):
        probabilities = self.agent_a3c(self.state_processer(state))
        return torch.multinomial(probabilities, 1).item()

