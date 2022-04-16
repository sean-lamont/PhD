import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# input size: (1, 1, 4, 64)
class TacPolicy(nn.Module):
    def __init__(self, action_size):
        super(TacPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(2048, 512)
        self.head = nn.Linear(512, action_size)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = self.out(self.head(x))
        return x


class ArgPolicy(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ArgPolicy, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim+hidden_dim*2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)
        self.out = nn.Sigmoid()

        # self.hidden = torch.randn(1,1)

    # x is the previously predicted argument / tactic.
    # candidates is a matrix of possible arguments concatenated with the hidden states.
    def forward(self, x, candidates, hidden):
        x = x.view(1,1,-1)
        candidates = candidates.view(candidates.size()[0], 1, -1)
        c = F.relu(self.fc1(candidates))
        c = F.relu(self.fc2(c))
        scores = self.out(c)
        o, hidden = self.lstm(x, hidden)
        return hidden, scores
