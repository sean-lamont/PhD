import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 256)
        self.head = nn.Linear(256, 2)
        self.out = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.head(x))
        x = self.out(x)
        return x


class SAValue(nn.Module):
    def __init__(self):
        super(SAValue, self).__init__()
        self.fc1 = nn.Linear(25, 128)
        self.fc2 = nn.Linear(128, 256)
        self.head = nn.Linear(256, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.head(x))
        x = self.out(x)
        return x


class SValue(nn.Module):
    def __init__(self):
        super(SValue, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 256)
        self.head = nn.Linear(256, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.head(x))
        x = self.out(x)
        return x
