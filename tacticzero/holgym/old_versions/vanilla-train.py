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
from env import *
from model import *
import json
import timeit
# from guppy import hpy

# PyCharm

ARG_LEN = 5

env = HolEnv(GOALS)

num_episode = 3
learning_rate = 0.0001
gamma = 0.99 # 0.9

tac_net = TacPolicy(len(tactic_pool))

arg_net = ScorePolicy()

# test = torch.zeros([4, 1, 4, 64], dtype=torch.float)
# optimizer = torch.optim.RMSprop(list(tac_net.parameters())+list(arg_net.parameters()), lr=learning_rate)

optimizer = torch.optim.RMSprop(list(tac_net.parameters())+list(arg_net.parameters()), lr=learning_rate)

state_pool = []
tac_pool = []
arg_pool = []
reward_pool = []
steps = 0


for e in range(num_episode):

    print("Game: {}".format(e))
    start = timeit.default_timer()
    
    # h = hpy()
    # print(h.heap())
    
    if e != 0:
        if (e+1) % 10 == 0:
            # release buffer memory
            env.process.terminate(True)
            env = HolEnv(GOALS)
        else:
            env.reset()

    g, state = env.get_states()
    
    for t in count():

        # size: (1, 1, 4, max_len)
        tac_state = state.unsqueeze(0).unsqueeze(0)
        # choose a tactic
        tac_probs = tac_net(tac_state)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        # log directly the log probability
        tac_pool.append(tac_m.log_prob(tac))

        tac_tensor = torch.randn(1,1)
        tac_tensor = tac_tensor.new_full((1,MAX_LEN), tac.item())

        # initialize hidden state
        input = torch.cat([state, tac_tensor])
        candidates = []
        for d in fact_pool:
            d = env.encode(d)
            d = torch.FloatTensor(d)
            d = d.view(-1, env.max_len)
            d = torch.cat([input, d])
            candidates.append(d.unsqueeze(0))

        candidates = torch.stack(candidates)

        # the indices of chosen args
        arg_step = []
        arg_step_probs = []
        for i in range(1):
            scores = arg_net(candidates)
            arg_probs = F.softmax(scores, dim=0)
            arg_m = Categorical(arg_probs.squeeze())
            arg = arg_m.sample()
            arg_step.append(arg)
            arg_step_probs.append(arg_m.log_prob(arg))
            input = torch.FloatTensor(env.encode(fact_pool[arg])) # exact shape (1,-1) doesn't matter
        arg_pool.append(arg_step_probs)

        tac = tactic_pool[tac]
        arg = [fact_pool[i] for i in arg_step]

        action = env.assemble_tactic(tac, arg)
        
        ng, next_state, reward, done = env.step(action)

        state_pool.append(state)
        reward_pool.append(reward)

        state = next_state
        g = ng

        steps += 1
        print("done. {}".format(steps))
        # exit()
        
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Reward pool: {}".format(reward_pool))
            # print("Mean reward: {}".format(np.mean(reward_pool)))
            print("Total reward: {}".format(np.sum(reward_pool)))
            print("Proof trace: {}".format(env.scripts))
            # exit()
            break

        if t > 49:
            print("Failed.")
            print("Reward pool: {}".format(reward_pool))
            # print("Mean reward: {}\n".format(np.mean(reward_pool)))
            print("Total reward: {}".format(np.sum(reward_pool)))
            break
        
    stop = timeit.default_timer()
    print('Time: {}\n'.format(stop - start))
    
    # Update policy
    if e >= 0:
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

        # print("Cumulative: {}\n".format(reward_pool))
        
        # Gradient Desent (Ascent)
        optimizer.zero_grad()

        for i in range(steps):
            # size : (1,1,4,128)
                
            state = state_pool[i]
            reward = reward_pool[i]

            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            
            tac_loss = -tac_pool[i] * (reward)
            loss = tac_loss + arg_loss
            
            loss.backward()

        optimizer.step()

        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
    # print("Loss: {}\n".format(loss))

print("Learned tac preferences: {}".format(tac_probs.detach()[0]))

torch.save(tac_net, "tac_net.ckpt")

torch.save(arg_net, "arg_net.ckpt")

print("Models saved.")

json = json.dumps(dictionary)
with open("dict.json","w") as f:
    f.write(json)

print("Dictionary saved.")
