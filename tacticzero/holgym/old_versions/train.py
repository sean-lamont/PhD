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

# PyCharm

# goals = ["!n. n < n + 1", "!n. n < n * 2", "!m. ~(m = m + 1)"]
# hard1 = ["!n m. ~(2*n = 2*(m+1))"] # false
# hard2 = ["!n m. ~(2*n = 2*m + 1)"] # true

# comp = ["!n. n < n + 1 ==> n < n+2 ==> n < n+3"]

provables = ["!n. n <= n * 2", "!p q. p ==> q ==> p", "!m. ~(m = m + 1)", "!n. n < n + 1", "!n. n < n + 1 ==> n < n+2 ==> n < n+3"]

foo = ["!n. n <= n * 2"]
env = HolEnv(provables)

num_episode = 15
learning_rate = 0.0001
gamma = 0.99 # 0.9

policy_net = PolicyNet(8)
# test = torch.zeros([4, 1, 4, 64], dtype=torch.float)
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

state_pool = []
action_pool = []
reward_pool = []
steps = 0


for e in range(num_episode):
    env.reset()
    state = env.get_states()

    for t in count():
        state = state.unsqueeze(0).unsqueeze(0)
        probs = policy_net(state)
        m = Categorical(probs)
        # print("Preferences: {}".format(probs.detach()[0]))
        action = m.sample()

        next_state, reward, done = env.step(action)

        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)

        state = next_state

        steps += 1

        if done == "metis_timeout":
            print("Failed due to metis timout.")
            print("Reward pool: {}".format(reward_pool))
            print("Mean reward: {}".format(np.mean(reward_pool)))
            break            
        
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Reward pool: {}".format(reward_pool))
            print("Mean reward: {}".format(np.mean(reward_pool)))
            print("Proof trace: {}\n".format(env.scripts))
            # exit()
            break

        if t > 50:
            print("Failed.")
            print("Reward pool: {}".format(reward_pool))
            print("Mean reward: {}".format(np.mean(reward_pool)))
            break

    # Update policy
    if e >= 0:
    # if e % 3 == 0:
        # Discount reward
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(steps):
            reward_pool[i] = (reward_pool[i] - reward_mean) / (reward_std + np.finfo(np.float32).eps) # eps is important, otherwise we get divided by 0

        # Gradient Desent (Ascent)
        optimizer.zero_grad()

        for i in range(steps):
            state = state_pool[i]
            action = torch.FloatTensor([action_pool[i]])
            reward = reward_pool[i]
            
            probs = policy_net(state)
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()

        optimizer.step()

        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0

print("Learned preferences: {}".format(probs.detach()[0]))

torch.save(policy_net, "net.ckpt")

print("Model saved.")

json = json.dumps(dictionary)
with open("dict.json","w") as f:
    f.write(json)

print("Dictionary saved.")
