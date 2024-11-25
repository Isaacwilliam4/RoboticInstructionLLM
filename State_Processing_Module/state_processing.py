import torch 
from torch import nn 
from transformers import DistilBertTokenizer, DistilBertModel

# class SimpleFFNN(nn.Module):
#     def __init__(self, input_size, last_layer):
#         super(SimpleFFNN, self).__init__()
#         self.last_layer = last_layer
#         self.fc1 = nn.Linear(input_size, 512)
#         self.fc2 = nn.Linear(512, self.last_layer) 

#     def forward(self, x):
#         x = x.mean(dim=1)
#         x = torch.relu(self.fc1(x)) 
#         x = self.fc2(x)
#         return x


class StateProcessingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_size = 256
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # self.ffn = SimpleFFNN(768, self.output_size)

    
    def multimodal_fusion(self, instruction, state):
        inputs = self.tokenizer(instruction, return_tensors='pt')
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # ffn_output = self.ffn(last_hidden_state)
        # state_tens = torch.tensor(state).unsqueeze(0)
        result = torch.cat([last_hidden_state, torch.tensor(state)], dim=1)
        return result




    

