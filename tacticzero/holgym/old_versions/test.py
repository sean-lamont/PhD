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

# load dictionary
with open("dict.json") as f:
    dictionary = json.load(f)

# load model
policy_net = torch.load("net.ckpt")
policy_net.eval()

tests = ["!n. n <= n * 3", "!n. n < n + 4", "!p q. q ==> p ==> q"]
env = HolEnv(tests)

state = env.get_states()

for t in count():
    state = state.unsqueeze(0).unsqueeze(0)
    probs = policy_net(state)
    m = Categorical(probs)
    # print("Preferences: {}".format(probs.detach()[0]))
    action = m.sample()
    # print(action)
    # exit()

    next_state, reward, done = env.step(action)

    # To mark boundarys between episodes
    # if done:
    #     reward = 0

    state = next_state

    if done:
        print("Proved in {} steps.".format(t+1))
        print("Proof trace: {}\n".format(env.scripts))
        # exit()
        break

    if t > 50:
        print("Failed.")
        break

    
