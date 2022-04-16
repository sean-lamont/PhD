import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# better network?
# Ask Christian
# class PolicyNet(nn.Module):
#     def __init__(self, action_size):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.fc = nn.Linear(1024, 512)
#         self.head = nn.Linear(512, action_size)
#         self.out = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.fc(x.view(x.size(0), -1)))
#         x = self.out(self.head(x))
#         return x

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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(4096, 128)
        self.head = nn.Linear(128, 1)
        self.out = nn.Sigmoid()

        # self.hidden = torch.randn(1,1)

    # x is the previously predicted argument / tactic.
    # candidates is a matrix of possible arguments concatenated with the hidden states.
    def forward(self, x, candidates, hidden):
        x = x.view(1,1,-1)
        
        s = self.conv1(candidates)
        s = F.relu(self.bn1(s))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.fc(s.view(s.size(0), -1)))
        scores = self.out(self.head(s))
        
        o, hidden = self.lstm(x, hidden)
        
        return hidden, scores

    
# input size: (n, 2, 4+1+1, 64)
# torch.cat((i,a),0)
# class ArgPolicy(nn.Module):
#     def __init__(self):
#         super(ArgPolicy, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=(2,2), stride=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(2,2), stride=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.fc = nn.Linear(128*93, 128)
#         self.head = nn.Linear(128, 1)
#         self.out = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.fc(x.view(x.size(0), -1)))
#         x = self.out(self.head(x))
#         return x


# input shape: (n,1,6,128)
class TermPolicy(nn.Module):
    def __init__(self):
        super(TermPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(32256, 128)
        self.head = nn.Linear(128, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = self.out(self.head(x))
        return x
