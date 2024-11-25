import torch 
from torch import nn 
from transformers import DistilBertTokenizer, DistilBertModel

class SimpleAttention(nn.Module):
    def __init__(self, input_size, last_layer):
        super(SimpleAttention, self).__init__()
        self.last_layer = last_layer
        self.attention = nn.MultiheadAttention(768, 1)
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, self.last_layer) 
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)
        x = torch.mean(x, dim=1)
        x = self.sig(x)
        return x


class StateProcessingModule(nn.Module):
    def __init__(self, num_agents, view_size) -> None:
        super().__init__()
        self.view_size = view_size
        self.num_agents = num_agents
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.ffn = SimpleAttention(768, 7*7*6*num_agents)

    
    def multimodal_fusion(self, instruction, state):
        inputs = self.tokenizer(instruction, return_tensors='pt')
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        ffn_output = self.ffn(last_hidden_state)
        gate = ffn_output.view(self.view_size, self.view_size, (self.num_agents*6))
        state = torch.tensor(state)
        gated_state_representation = state * gate
        # state_tens = torch.tensor(state).unsqueeze(0)
        # result = torch.cat([last_hidden_state, torch.tensor(state)], dim=1)
        return gated_state_representation




    

