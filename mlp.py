import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Input layer
        self.input1 = nn.Linear(input_size, 250)
        # Dropout here
        self.dropout1 = nn.Dropout(p=0.3)
        # Hidden layer
        self.hidden1 = nn.Linear(250, 100)
        # Output layer
        self.output1 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.tanh(self.input1(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden1(x))
        return self.output1(x)
