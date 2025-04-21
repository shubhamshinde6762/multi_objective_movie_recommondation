import torch
import torch.nn as nn
from pathlib import Path

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
    def save(self, path):
        """Save the model to a file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path, input_size, hidden_size, output_size):
        """Load the model from a file"""
        model = cls(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(path))
        return model 