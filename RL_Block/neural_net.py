import torch
import torch.nn as nn


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
        # x dim = (batch_size, time_steps, agent_view_size, agent_view_size, 6*num_agents)
        x = x.flatten(2)
        h_state = self.h0.repeat(1, x.size(0), 1)
        c_state = self.c0.repeat(1, x.size(0), 1)
        x = self.fc1(x)
        x = self.activation_layer(x)

        out, (h_state, c_state) = self.lstm(x, (h_state, c_state))
        out = self.fc_final(out[:, -1, :])
        values, action_logits = out[:, 0], out[:, 1:]

        if self.action_mask is not None:
            action_logits = torch.where(self.action_mask == float('-inf'), self.action_mask, action_logits)
        
        action_probs = self.softmax(action_logits)
        
        return values, action_probs

