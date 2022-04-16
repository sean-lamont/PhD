# from subprocess import Popen, PIPE
from random import sample
# from random import randint
import pexpect
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from sys import exit
from agent import *
from model import *
import json

with open("dict.json") as f:
    dictionary = json.load(f)

# load model
policy_net = torch.load("net.ckpt")
policy_net.eval()

tests = ["!n. n <= n * 3", "!n. n < n + 4", "!p q. q ==> p ==> q"]
env = HolEnv(tests)

state = env.get_states()

s = []

for i in range(5):
    steps = env.random_play_no_back()
    s.append(steps)
    env.reset()

print("Mean timesteps: {}".format(np.mean(s)))
