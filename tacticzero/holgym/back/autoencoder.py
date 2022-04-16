# from subprocess import Popen, PIPE
import random
random.seed(0)
import pexpect
import re
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from sys import exit
from exp_env import *
from exp_model import *
from exp_config import *
import json
import timeit
import json
import resource

class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)

        self.decoder = nn.LSTM(self.latent_dim, self.input_dim, self.num_layers)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        
        encoded = last_hidden.repeat(input.shape)

        # Decode
        y, _ = self.decoder(encoded)
        return torch.squeeze(y)


model = LSTM(input_dim=1, latent_dim=40, num_layers=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# figure out how to do padding
y = torch.Tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])

y1 = torch.Tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])
y2 = torch.Tensor([[0.4], [0.8], [0.1]])

y3 = torch.Tensor([[0.4], [0.8], [0.1], [0.6], [0.6], [0.9]])

yt = torch.Tensor([[0.6], [0.3]])

yy = [y2.unsqueeze(2),y1.unsqueeze(2),y3.unsqueeze(2)]

t = torch.Tensor([[[0.0], [0.9]],
                  [[0.1], [0.3]],
                  [[0.2], [0.5]],
                  [[0.3], [0.1]],
                  [[0.4], [0.8]]])

# z = torch.Tensor([[0.2], [0.8], [0.4], [0.9], [0.1]])
# d = [y, z]

# Sequence x batch x dimension
# x = y.view(len(y), 1, -1)

# for i in range(20000):
#     y_pred = model(t)
#     target = t.view(y_pred.shape)
#     optimizer.zero_grad()
#     loss = loss_function(y_pred, target)
#     loss.backward()
#     optimizer.step()
#     if i % 5000 == 0:
#         print(loss.detach())

# print(y_pred)

    optimizer.zero_grad()
    for j in yy:
        y_pred = model(j)
        target = j.view(y_pred.shape)
        loss += loss_function(y_pred, target)
        pd.append(y_pred.detach())
    loss.backward()
    optimizer.step()
    if i % 5000 == 0:
        print(loss.detach())

print(pd)
print(model(yt.unsqueeze(2)))
