import torch
from torch import nn
import numpy as np
from torch.amp import GradScaler, autocast

class NeuralNetBlock(nn.Module):
    def __init__(
            self, 
            input_space_size, 
            output_space_size,
            lstm_hidden_size,
            lstm_num_layers,
            fc_linear1_output=256,
            time_steps_to_RNN=1,
            action_mask=None,
        ):
        super(NeuralNetBlock, self).__init__()
        self.input_space_size = input_space_size
        self.output_space_size = output_space_size
        self.lstm_hidden_layer_size = lstm_hidden_size
        self.num_lstm_layers = lstm_num_layers
        self.fc_linear_1_output_size = fc_linear1_output
        self.time_steps_fed_to_RNN = time_steps_to_RNN
        self.action_mask = action_mask

        self.h0 = nn.Parameter(torch.zeros(self.num_lstm_layers, 1, self.lstm_hidden_layer_size))
        self.c0 = nn.Parameter(torch.zeros(self.num_lstm_layers, 1, self.lstm_hidden_layer_size))

        self.fc1 = nn.Linear(input_space_size, self.fc_linear_1_output_size)
        self.fc_final = nn.Linear(self.lstm_hidden_layer_size, (self.output_space_size + 1))

        self.activation_layer = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(
            self.fc_linear_1_output_size, 
            self.lstm_hidden_layer_size, 
            self.num_lstm_layers, 
            batch_first=True, 
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor):
        h0 = self.h0.repeat(1, x.size(0), 1)
        c0 = self.h0.repeat(1, x.size(0), 1)
        x = x.view(x.size(0), self.time_steps_fed_to_RNN, -1) ### TODO potentially adjust number of time steps being fed into the model
        x = self.activation_layer(self.fc1(x))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_final(out[:, -1, :])
        values, action_logits = out[:, 0], out[:, 1:]

        if self.action_mask is not None:
            action_logits = torch.where(self.action_mask == float('-inf'), self.action_mask, action_logits)
        
        action_probs = self.softmax(action_logits)
        
        return values, action_probs

class DummyObj:
    def __init__(self):
        self.something = None

    def __call__(self, inp):
        return inp
    

class ReplayBuffer:
    def __init__(self, state_dim, device, buffer_size=10_000):
        self.device = device
        self.buffer_size = buffer_size
        self.position = 0
        self.size = 0

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        # self.intermediate_steps = torch.zeros((buffer_size, intermediate_step_state_size))  ### TODO Do we want to keep track of this state? or the original state?

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



class Agent():
    def __init__(
            self, 
            input_space_size, 
            output_space_size,
            lstm_hidden_size,
            lstm_num_layers,
            fc_linear1_output=256,
            time_steps_to_RNN=1,
            action_mask=[1,1,1,0,0,0,1],
            train_every_n_iters= 10,
            gamma=0.99,
            batch_size=64,
        ):
        assert len(action_mask) == output_space_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.steps_since_train = 0
        self.train_every_n_iters = train_every_n_iters
        self.episode_num = 0
        self.scaler = GradScaler()
        self.gamma = gamma
        self.batch_size = batch_size

        action_mask = torch.tensor(action_mask)
        z = torch.zeros_like(action_mask)
        action_mask = torch.where(action_mask == 0, torch.full(z.size(), float('-inf')), z)
        print(action_mask)

        self.agent_a3c = NeuralNetBlock(
            input_space_size, 
            output_space_size,
            lstm_hidden_size,
            lstm_num_layers,
            fc_linear1_output,
            time_steps_to_RNN=time_steps_to_RNN,
            action_mask=action_mask.to(self.device)
        ).to(self.device)

        self.visual_and_audio_parser = DummyObj()
        self.memory = ReplayBuffer(state_dim=input_space_size, device=self.device)

        self.optimizer = torch.optim.Adam(self.agent_a3c.parameters, lr=0.001)




    def train(self):
        self.episode_num += 1
        if self.episode_num % self.train_every_n_iters != 0: return 
        for _ in range(self.steps_since_train):
            self.learn_from_replay()
        self.steps_since_train = 0
        
    def get_action(self, state):
        probabilities = self.agent_a3c(self.visual_and_audio_parser(state))
        return torch.multinomial(probabilities, 1).item()

    def step(self, state, action, reward, next_state, done):
        ### TODO What to do here?
        # state = self.visual_and_audio_parser(state)
        # next_state = self.visual_and_audio_parser(state)

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
