import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        net = nn.Sequential(nn.Linear(),
                            nn.ReLU(),
                            nn.Linear(),
                            nn.ReLU(),
                            nn.Linear())
