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
from config import *
import json
import timeit
# from guppy import hpy
import resource

# h = hpy()
GAMES = TEST

# PyCharm
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ARG_LEN = 5

env = HolEnv(GAMES, 0)

tac_rate = 1e-5
arg_rate = 1e-4
term_rate = 1e-5

gamma = 0.99 # 0.9

num_episode = 5

# for entropy regularization
trade_off = 1e-2

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(MAX_LEN, MAX_LEN)

term_net = TermPolicy()

device = torch.device('cpu')

trained_episode = 5

fn = "{}-{}-{}-{}-train.ckpt".format(trained_episode, tac_rate, arg_rate, term_rate)

checkpoint =  torch.load(fn)
tac_net.load_state_dict(checkpoint["tac_net_state_dict"])
arg_net.load_state_dict(checkpoint["arg_net_state_dict"])
term_net.load_state_dict(checkpoint["term_net_state_dict"])
tac_net.eval()
arg_net.eval()
term_net.eval()
print("Models loaded.")

# tac_net.load_state_dict(torch.load("tac_net.ckpt", map_location=device))
# arg_net.load_state_dict(torch.load("arg_net.ckpt", map_location=device))
# term_net.load_state_dict(torch.load("term_net.ckpt", map_location=device))

state_pool = []
tac_pool = []
arg_pool = []
reward_pool = []
reward_print = []
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
                    # only to raise an error
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
            env = HolEnv(GAMES, counter-2)
            flag = False
            break

        # this could be removed
        if reward == UNEXPECTED_REWARD:
            # try again
            counter = env.counter
            env.close()
            print("Reverting to the previous game ...")
            env = HolEnv(GAMES, counter-2)
            flag = False
            break

        state_pool.append(state)
        # arg_loss = -torch.sum(torch.stack(arg_step_probs))
        # tac_loss = -tac_pool[-1]
        # entropy = arg_loss + tac_loss
        
        reward_print.append(reward)
        reward_pool.append(reward)
        # reward_pool.append(reward+trade_off*entropy)

        pg = ng

        steps += 1
        
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Rewards: {}".format(reward_print))
            print("Tactics: {}".format(action_pool))
            # print("Mean reward: {}".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_print)))
            print("Proof trace: {}".format(env.scripts))
            # exit()
            proved += 1
            break

        if t > 19:
            print("Failed.")
            print("Rewards: {}".format(reward_print))
            # print("Rewards: {}".format(reward_pool))
            print("Tactics: {}".format(action_pool))
            # print("Mean reward: {}\n".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_print)))
            break

        state = next_state
        state = state.to(device)

    stop = timeit.default_timer()
    print('Time: {}  '.format(stop - start))

    # for i in range(steps):
    #     # size : (1,1,4,128)
    #     total_loss = 0

    #     arg_loss = -torch.sum(torch.stack(arg_pool[i]))

    #     tac_loss = -tac_pool[i]
    #     entropy = tac_loss + arg_loss
    #     reward_pool[i] = reward_pool[i] + trade_off * entropy

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
        
        # optimizer_tac.zero_grad()
        # optimizer_arg.zero_grad()
        # optimizer_term.zero_grad()

        
        # for i in range(steps):
        #     # size : (1,1,4,128)
        #     total_loss = 0
            
        #     state = state_pool[i]
        #     reward = reward_pool[i]

        #     arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            
        #     tac_loss = -tac_pool[i] * (reward)
        #     loss = tac_loss + arg_loss

        #     total_loss += loss
            
        #     # loss.backward()

        # total_loss.backward()

        # # optimizer.step()
        # optimizer_tac.step()
        # optimizer_arg.step()
        # optimizer_term.step()

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    state_pool = []
    tac_pool = []
    arg_pool = []
    action_pool = []
    reward_pool = []
    reward_print = []
    steps = 0
    flag = True
        
    # print("Loss: {}\n".format(loss))
    print("Induct args: {}".format(induct_arg))
    induct_arg = []
    prf = torch.mean(torch.stack(tac_print), 0)
    print("Preferences: {}".format(prf))
    tac_print = []
    print("Proved so far: {}\n".format(proved))


print("Models saved.")
env.close()
