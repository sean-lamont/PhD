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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

ARG_LEN = 5

env = HolEnv(GOALS)

num_episode = 400
learning_rate = 0.0001
gamma = 0.999 # 0.9

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(MAX_LEN, MAX_LEN)

term_net = TermPolicy()

tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

# test = torch.zeros([4, 1, 4, 64], dtype=torch.float)
# optimizer = torch.optim.RMSprop(list(tac_net.parameters())+list(arg_net.parameters()), lr=learning_rate)

optimizer = torch.optim.RMSprop(list(tac_net.parameters())+list(arg_net.parameters())+list(term_net.parameters()), lr=learning_rate)

state_pool = []
tac_pool = []
arg_pool = []
reward_pool = []
action_pool = []
steps = 0

induct_arg = []
proved = 0

for e in range(num_episode):
    
    print("Game: {}".format(e))
    start = timeit.default_timer()
    
    # h = hpy()
    # print(h.heap())
    
    if e != 0:
        if (e+1) % 25 == 0:
            # release buffer memory
            env.process.terminate(True)
            env = HolEnv(GOALS)
        else:
            env.reset()

    g, state = env.get_states()
    
    for t in count():

        # size: (1, 1, 4, max_len)
        tac_state = state.unsqueeze(0).unsqueeze(0)
        tac_state = tac_state.to(device)
        
        # choose a tactic
        tac_probs = tac_net(tac_state)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        # log directly the log probability
        tac_pool.append(tac_m.log_prob(tac))
        action_pool.append(tactic_pool[tac])
        
        tac_tensor = torch.randn(1,1)
        tac_tensor = tac_tensor.new_full((1,MAX_LEN), tac.item())

        if tactic_pool[tac] == "Induct_on":
            arg_probs = []
            candidates = []
            input = torch.cat([state, tac_tensor])
            tokens = g.split()
            terms = []
            for i in tokens:
                terms.append(i)
                term_tensor = torch.randn(1,1)
                term_tensor = term_tensor.new_full((1,MAX_LEN), dictionary[i])
                candidate = torch.cat([input, term_tensor])
                candidates.append(candidate.unsqueeze(0))
            candidates = torch.stack(candidates)
            candidates = candidates.to(device)
            
            scores = term_net(candidates)
            term_probs = F.softmax(scores, dim=0)
            try:
                term_m = Categorical(term_probs.squeeze(1))
            except:
                print("probs: {}".format(term_probs))                                          
                print("candidates: {}".format(candidates.shape))
                print("scores: {}".format(scores))
                print("tokens: {}".format(tokens))
                exit()
            term = term_m.sample()
            arg_probs.append(term_m.log_prob(term))
            induct_arg.append(terms[term])
            tm = terms[term][1:] # remove headers, e.g., "V" / "C" / ...
            arg_pool.append(arg_probs)
            if tm:
                action = "Induct_on `{}`".format(tm)
            else:
                action = "Induct_on"
            
        else:
            # initialize hidden state
            hidden0 = torch.randn(1,1,MAX_LEN)
            hidden1 = torch.randn(1,1,MAX_LEN)
            hidden0 = hidden0.to(device)
            hidden1 = hidden1.to(device)
            
            hidden = (hidden0, hidden1)
            # concatenate the candidates with hidden states.
            candidates = []
            for d in fact_pool:
                d = env.encode(d)
                d = torch.tensor(d, dtype=torch.float, device=device)
                # d = d.view(-1, env.max_len)
                d = torch.cat([d, hidden[0].view(-1), hidden[1].view(-1)])
                candidates.append(d)

            candidates = torch.stack(candidates)
            input = tac_tensor
            
            input = input.to(device)
            candidates = candidates.to(device)

            # the indices of chosen args
            arg_step = []
            arg_step_probs = []
            for i in range(ARG_LEN):
                hidden, scores = arg_net(input, candidates, hidden)
                arg_probs = F.softmax(scores, dim=0)
                arg_m = Categorical(arg_probs.squeeze())
                arg = arg_m.sample()
                arg_step.append(arg)
                arg_step_probs.append(arg_m.log_prob(arg))
                input = torch.tensor(env.encode(fact_pool[arg]), dtype=torch.float, device=device) # exact shape (1,-1) doesn't matter
                # renew candidates
                candidates = []
                for d in fact_pool:
                    d = env.encode(d)
                    d = torch.tensor(d, dtype=torch.float, device=device)
                    # d = d.view(-1, env.max_len)
                    d = torch.cat([d, hidden[0].view(-1), hidden[1].view(-1)])
                    candidates.append(d)
                candidates = torch.stack(candidates)
                candidates = candidates.to(device)

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
        
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Rewards: {}".format(reward_pool))
            print("Tactics: {}".format(action_pool))

            # print("Mean reward: {}".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_pool)))
            print("Proof trace: {}".format(env.scripts))
            # exit()
            proved += 1
            break

        if t > 49:
            print("Failed.")
            print("Rewards: {}".format(reward_pool))
            print("Tactics: {}".format(action_pool))
            # print("Mean reward: {}\n".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_pool)))
            break
        
    stop = timeit.default_timer()
    print('Time: {}  '.format(stop - start))
    
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
        tac_pool = []
        arg_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
    # print("Loss: {}\n".format(loss))
    print("Induct args: {}".format(induct_arg))
    induct_arg = []
    print("Preferences: {}".format(tac_probs.detach()[0]))
    print("Proved so far: {}\n".format(proved))

# print("Learned tac preferences: {}".format(tac_probs.detach()[0]))


torch.save(tac_net, "tac_net.ckpt")

torch.save(arg_net, "arg_net.ckpt")

print("Models saved.")

json = json.dumps(dictionary)
with open("dict.json","w") as f:
    f.write(json)

print("Dictionary saved.")
