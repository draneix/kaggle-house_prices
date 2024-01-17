import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Input layer
        self.input1 = nn.Linear(input_size, 250)
        # Batch norm
        self.batch1 = nn.BatchNorm1d(250)
        # Dropout here
        self.dropout1 = nn.Dropout(p=0.1)
        # Hidden layer
        self.hidden1 = nn.Linear(250, 100)
        self.batch2 = nn.BatchNorm1d(100)
        self.hidden2 = nn.Linear(100, 50)
        self.batch3 = nn.BatchNorm1d(50)
        # Output layer
        self.output1 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.input1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden1(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.batch3(x)
        return self.output1(x)
