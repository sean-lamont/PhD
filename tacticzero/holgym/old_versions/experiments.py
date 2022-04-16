# from subprocess import Popen, PIPE
from random import sample
import pexpect
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from sys import exit
from new_env import *
from new_model import *
from config import *
import json
import timeit
# from guppy import hpy
import json
import resource

# h = hpy()

# PyCharm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

ARG_LEN = 5

env = HolEnv(GOALS, 0)

num_episode = 10

learning_rate = 1e-6

tac_rate = 1e-5
arg_rate = 1e-5
term_rate = 1e-5


gamma = 0.99 # 0.9

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(MAX_LEN, MAX_LEN)

term_net = TermPolicy()

tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

# test = torch.zeros([4, 1, 4, 64], dtype=torch.float)

# optimizer = torch.optim.RMSprop(list(tac_net.parameters())+list(arg_net.parameters())+list(term_net.parameters()), lr=learning_rate)

optimizer_tac = torch.optim.RMSprop(list(tac_net.parameters()), lr=tac_rate)

optimizer_arg = torch.optim.RMSprop(list(arg_net.parameters()), lr=arg_rate)

optimizer_term = torch.optim.RMSprop(list(term_net.parameters()), lr=term_rate)

state_pool = []
tac_pool = []
arg_pool = []
reward_pool = []
action_pool = []
steps = 0
flag = True
tac_print = []

induct_arg = []
proved = 0

for e in range(num_episode):

    print("Game: {}".format(e))
    print("Facts: {}".format(len(fact_pool)))
    # print("Last: {}".format(fact_pool[-1]))
    start = timeit.default_timer()
    
    # h = hpy()
    # print(h.heap())
    
    if e != 0:
        # if (e+1) % 25 == 0:
        #     counter = env.counter
        #     # release buffer memory
        #     env.process.terminate(True)
        #     env = HolEnv(GOALS, counter)
        # else:
        #     env.reset()
        env.reset()
            
    print("Count: {}".format(env.counter))
    
    pg, state = env.get_states()
    state = state.to(device)
    for t in count():

        # size: (1, max_contexts, max_assumptions+1, max_len)
        tac_state = state.unsqueeze(0)
        tac_state = tac_state.to(device)
        
        # choose a tactic
        tac_probs = tac_net(tac_state)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        # log directly the log probability
        tac_pool.append(tac_m.log_prob(tac))
        action_pool.append(tactic_pool[tac])
        tac_print.append(tac_probs.detach())
        
        tac_tensor = torch.randn(1,1)
        tac_tensor = tac_tensor.new_full((MAX_CONTEXTS,1,MAX_LEN), tac.item())
        tac_tensor = tac_tensor.to(device)

        if tactic_pool[tac] == "Induct_on":
            arg_probs = []
            candidates = []
            input = torch.cat([state, tac_tensor], dim=1)
            tokens = pg.split()
            tokens = list(dict.fromkeys(tokens))
            tokens = [t for t in tokens if t[0] == "V"]
            terms = []
            if tokens:
                for i in tokens:
                    terms.append(i)
                    term_tensor = torch.randn(1,1)
                    term_tensor = term_tensor.new_full((MAX_CONTEXTS,1,MAX_LEN), dictionary[i])
                    term_tensor = term_tensor.to(device)
                    candidate = torch.cat([input, term_tensor], dim=1)
                    candidates.append(candidate)
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
                arg_probs.append(torch.tensor(0))
                induct_arg.append("No variables")
                arg_pool.append(arg_probs)
                action = "Induct_on"
        else:
            # initialize hidden state
            hidden0 = torch.randn(1,1,MAX_LEN)
            hidden1 = torch.randn(1,1,MAX_LEN)
            hidden0 = hidden0.to(device)
            hidden1 = hidden1.to(device)
            
            hidden = (hidden0, hidden1)
            # shape: (MAX_CONTEXTS, 1, MAX_LEN)
            hiddenc0 = hidden[0].squeeze().repeat(MAX_CONTEXTS, 1, 1)
            hiddenc1 = hidden[1].squeeze().repeat(MAX_CONTEXTS, 1, 1)

            arg_state = state
            # concatenate the candidates with hidden states.
            
            candidates = []
            for d in fact_pool:
                d = env.encode(d)
                d = torch.tensor(d, dtype=torch.float, device=device)
                d = d.repeat(MAX_CONTEXTS, 1, 1)
                d = d.to(device)
                # size : (MAX_CONTEXTS, 4+1+1+1+1, max_len)
                d = torch.cat([state, d, tac_tensor, hiddenc0, hiddenc1], dim=1)
                d = d
                candidates.append(d)

            candidates = torch.stack(candidates)
            # because here there are MAX_CONTEXTS many of them
            input = tac_tensor[0]
            
            input = input.to(device)
            candidates = candidates.to(device)

            # the indices of chosen args
            arg_step = []
            arg_step_probs = []
            for i in range(ARG_LEN):
                # print(input.shape)
                # print(candidates.shape)
                # exit()
                hidden, scores = arg_net(input, candidates, hidden)
                arg_probs = F.softmax(scores, dim=0)
                arg_m = Categorical(arg_probs.squeeze())
                arg = arg_m.sample()
                arg_step.append(arg)
                arg_step_probs.append(arg_m.log_prob(arg))

                hiddenc0 = hidden[0].squeeze().repeat(MAX_CONTEXTS, 1, 1)
                hiddenc1 = hidden[1].squeeze().repeat(MAX_CONTEXTS, 1, 1)

                input = torch.tensor(env.encode(fact_pool[arg]), dtype=torch.float, device=device) # exact shape (1,-1) doesn't matter
                inputc = input.repeat(MAX_CONTEXTS, 1, 1)
                # renew candidates
                candidates = []
                for d in fact_pool:
                    d = env.encode(d)
                    d = torch.tensor(d, dtype=torch.float, device=device)
                    d = d.repeat(MAX_CONTEXTS, 1, 1)
                    # size : (MAX_CONTEXTS, 4+1+1+1+1, max_len)
                    d = torch.cat([state, d, inputc, hiddenc0, hiddenc1], dim=1)
                    d = d
                    candidates.append(d)

                candidates = torch.stack(candidates)
                candidates = candidates.to(device)

            arg_pool.append(arg_step_probs)

            tac = tactic_pool[tac]
            arg = [fact_pool[i] for i in arg_step]

            action = env.assemble_tactic(tac, arg)
        try:
            ng, next_state, reward, done = env.step(action)

        except:
            print("Step exception raised.")
            print("Fringe: {}".format(env.fringe))
            print("Handling: {}".format(env.handling))
            print("Using: {}".format(env.using))
            # try again
            counter = env.counter
            env.close()
            print("Reverting to the previous game ...")
            env = HolEnv(GOALS, counter-2)
            flag = False
            break

        # this could be removed
        if reward == UNEXPECTED_REWARD:
            # try again
            counter = env.counter
            env.close()
            print("Reverting to the previous game ...")
            env = HolEnv(GOALS, counter-2)
            flag = False
            break
            
        state_pool.append(state)
        reward_pool.append(reward)

        pg = ng

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

        state = next_state
        state = state.to(device)

    stop = timeit.default_timer()
    print('Time: {}  '.format(stop - start))

    # Update policy
    if flag:
        # Discount reward
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        
        # reward_mean = np.mean(reward_pool)
        # reward_std = np.std(reward_pool)
        # for i in range(steps):
        #     reward_pool[i] = (reward_pool[i] - reward_mean) / (reward_std + np.finfo(np.float32).eps) # eps is important, otherwise we get divided by 0

        # print("Cumulative: {}\n".format(reward_pool))
        
        # Gradient Desent (Ascent)
        # optimizer.zero_grad()
        
        optimizer_tac.zero_grad()
        optimizer_arg.zero_grad()
        optimizer_term.zero_grad()

        
        for i in range(steps):
            # size : (1,1,4,128)
            total_loss = 0
            
            state = state_pool[i]
            reward = reward_pool[i]

            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            
            tac_loss = -tac_pool[i] * (reward)
            loss = tac_loss + arg_loss

            total_loss += loss
            
            # loss.backward()

        total_loss.backward()

        # optimizer.step()
        # optimizer_tac.step()
        optimizer_arg.step()
        optimizer_term.step()

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    state_pool = []
    tac_pool = []
    arg_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    flag = True
        
    # print("Loss: {}\n".format(loss))
    print("Induct args: {}".format(induct_arg))
    induct_arg = []
    prf = torch.mean(torch.stack(tac_print), 0)
    print("Preferences: {}".format(prf))
    tac_print = []
    print("Proved so far: {}\n".format(proved))

torch.save(tac_net, "tac_net.ckpt")

torch.save(arg_net, "arg_net.ckpt")

torch.save(term_net, "term_net.ckpt")

print("Models saved.")

dict_json = json.dumps(dictionary)
with open("dict.json","w") as f:
    f.write(dict_json)

print("Dictionary saved.")
fact_json = json.dumps(fact_pool)
with open("fact_pool.json","w") as f:
    f.write(fact_json)
print("Facts saved.")

new_fact_json = json.dumps(list(new_facts.values()))
with open("new_facts.json","w") as f:
    f.write(new_fact_json)
print("Proved theorems saved.")
