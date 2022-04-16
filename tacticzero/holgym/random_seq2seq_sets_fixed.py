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
from sets_env import *
from seq2seq_sets_model import *
from exp_config import *
import json
import timeit
import time
# from guppy import hpy
import json
import resource
# import gc
# h = hpy()

# PyCharm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

record_proofs = False
traces = []
# device = torch.device("cpu")

ARG_LEN = 5

# CONTINUE = True
GAMES = TEST_GOALS
# GAMES = TRAIN
# GAMES = ["∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c"]
# GAMES = ["¬SHORTLEX R l []"]
print("Total games: {}".format(len(GAMES)))

num_episode = 30000
# CONTINUE = True

learning_rate = 1e-5

context_rate = 5e-5
tac_rate = 5e-5
arg_rate = 5e-5
term_rate = 5e-5

gamma = 0.99 # 0.9

# for entropy regularization
trade_off = 1e-2

context_net = ContextPolicy()

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(8, 256)

term_net = TermPolicy(8, 256)

context_net = context_net.to(device)
tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

optimizer_context = torch.optim.RMSprop(list(context_net.parameters()), lr=context_rate)

optimizer_tac = torch.optim.RMSprop(list(tac_net.parameters()), lr=tac_rate)

optimizer_arg = torch.optim.RMSprop(list(arg_net.parameters()), lr=arg_rate)

optimizer_term = torch.optim.RMSprop(list(term_net.parameters()), lr=term_rate)

fn = time.strftime('%Y_%m_%d_%H_%M_%S.ckpt', time.localtime())

if CONTINUE:
    saved = "10_20steps.ckpt"
    checkpoint = torch.load(saved)
    context_net.load_state_dict(checkpoint["context_net_state_dict"])
    tac_net.load_state_dict(checkpoint["tac_net_state_dict"])
    arg_net.load_state_dict(checkpoint["arg_net_state_dict"])
    term_net.load_state_dict(checkpoint["term_net_state_dict"])
    optimizer_context.load_state_dict(checkpoint['optimizer_context_state_dict'])
    optimizer_tac.load_state_dict(checkpoint['optimizer_tac_state_dict'])
    optimizer_arg.load_state_dict(checkpoint['optimizer_arg_state_dict'])
    optimizer_term.load_state_dict(checkpoint['optimizer_term_state_dict'])
    context_net.train()
    tac_net.train()
    arg_net.train()
    term_net.train()
    print("Models loaded.")

env = HolEnv(GAMES, 0)


# state_pool = []
fringe_pool = []
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

num_init_facts = len(fact_pool)

for e in range(num_episode):

    print("Game: {}".format(e))
    
    # print("Last: {}".format(fact_pool[-1]))
    start = timeit.default_timer()

    if e != 0:
        env.reset()
        
    # compute the encoded fact pool at the beginning of each episode
    polished_goal = env.fringe["content"][0]["polished"]["goal"]
    # split_fact_pool = [i.strip().split() for i in fact_pool if i != polished_goal]
    candidate_args = [t for t in fact_pool if t != polished_goal]
    split_fact_pool = [i.strip().split() for i in candidate_args]

    out, _ = batch_encoder.encode(split_fact_pool)
    # reshape the output
    encoded_fact_pool = torch.cat(out.split(1), dim=2).squeeze(0)

    print("Facts: {}+{}".format(num_init_facts, len(split_fact_pool)-num_init_facts))
    
    print("Count: {}".format(env.counter))
    
    state = env.history

    for t in count():

        # gather all the goals in the history
        representations, context_set, fringe_sizes = gather_encoded_content(env.history, batch_encoder)
        representations = representations.to(device)
        context_scores = context_net(representations)
        contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
        fringe_scores = []
        for s in scores_by_fringe:
            # fringe_score = torch.prod(s) # TODO: make it sum
            fringe_score = torch.sum(s) # TODO: make it sum
            fringe_scores.append(fringe_score)
        fringe_scores = torch.stack(fringe_scores)
        fringe_probs = F.softmax(fringe_scores, dim=0)
        fringe_m = Categorical(fringe_probs)
        fringe = fringe_m.sample()
        fringe_pool.append(fringe_m.log_prob(fringe))
        
        # print(fringe_scores)
        # print(fringe_probs)
        # print(fringe)
        # print(fringe_m.log_prob(fringe))
        # print(contexts_by_fringe[fringe.item()][0])
        # print(context_set.index(contexts_by_fringe[fringe.item()][0]))
        # take the first context in the chosen fringe for now
        
        target_context = contexts_by_fringe[fringe][0]
        target_goal = target_context["polished"]["goal"]
        target_representation = representations[context_set.index(target_context)]
        # print(target_representation.shape)
        # exit()
        
        # size: (1, max_contexts, max_assumptions+1, max_len)
        tac_input = target_representation.unsqueeze(0)
        tac_input = tac_input.to(device)
        
        # compute scores of tactics
        tac_probs = tac_net(tac_input)
        # print(tac_probs)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        # log directly the log probability
        tac_pool.append(tac_m.log_prob(tac))
        action_pool.append(tactic_pool[tac])
        tac_print.append(tac_probs.detach())
        # print(len(fact_pool[0].strip().split()))
        # exit()
        
        tac_tensor = tac.to(device)
        

        if tactic_pool[tac] in no_arg_tactic:
            tactic = tactic_pool[tac]
            arg_probs = []
            arg_probs.append(torch.tensor(0))
            arg_pool.append(arg_probs)
        elif tactic_pool[tac] == "Induct_on":
            arg_probs = []
            candidates = []
            # input = torch.cat([target_representation, tac_tensor], dim=1)
            tokens = target_goal.split()
            tokens = list(dict.fromkeys(tokens))
            tokens = [[t] for t in tokens if t[0] == "V"]
            if tokens:
                # concatenate target_representation to token
                # use seq2seq to compute the representation of a token
                # also we don't need to split an element in tokens because they are singletons
                # but we need to make it a list containing a singleton list, i.e., [['Vl']]
                
                token_representations, _ = batch_encoder.encode(tokens)
                # reshaping
                encoded_tokens = torch.cat(token_representations.split(1), dim=2).squeeze(0)
                target_representation_list = [target_representation.unsqueeze(0) for _ in tokens]
                
                target_representations = torch.cat(target_representation_list)
                # size: (len(tokens), 512)
                candidates = torch.cat([encoded_tokens, target_representations], dim=1)
                candidates = candidates.to(device)
                
                # concat = [torch.cat([torch.tensor([input_vocab.stoi[i] for _ in range(256)], dtype=torch.float), target_representation]) for i in tokens]

                # candidates = torch.stack(concat)
                # candidates = candidates.to(device)

                scores = term_net(candidates, tac_tensor)
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
                induct_arg.append(tokens[term])                
                tm = tokens[term][0][1:] # remove headers, e.g., "V" / "C" / ...
                arg_pool.append(arg_probs)
                if tm:
                    tactic = "Induct_on `{}`".format(tm)
                else:
                    # only to raise an error
                    tactic = "Induct_on"
            else:
                arg_probs.append(torch.tensor(0))
                induct_arg.append("No variables")
                arg_pool.append(arg_probs)
                tactic = "Induct_on"
        else:
            # thm_tactic or thms_tactic
            # initialize hidden state
            
            # hidden0 = torch.randn(1,1,MAX_LEN)
            # hidden1 = torch.randn(1,1,MAX_LEN)
            
            # hidden0, hidden1 = target_representation.split(target_representation.shape[0]//2)
            
            # hidden0 = hidden0.unsqueeze(0).unsqueeze(0)
            # hidden1 = hidden1.unsqueeze(0).unsqueeze(0)
            
            # hidden0 = hidden0.to(device)
            # hidden1 = hidden1.to(device)
            
            # hidden = (hidden0, hidden1)
            
            hidden0 = hidden1 = target_representation.unsqueeze(0).unsqueeze(0)
                        
            hidden0 = hidden0.to(device)
            hidden1 = hidden1.to(device)
            
            hidden = (hidden0, hidden1)
            
            # concatenate the candidates with hidden states.
                        
            hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
            hiddenl = [hc.unsqueeze(0) for _ in split_fact_pool]

            hiddenl = torch.cat(hiddenl)

            # size: (len(fact_pool), 512)
            candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
            candidates = candidates.to(device)
            
            input = tac_tensor
            
            # run it once before predicting the first argument
            hidden, _ = arg_net(input, candidates, hidden)

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

                # hiddenc0 = hidden[0].squeeze().repeat(1, 1, 1)
                # hiddenc1 = hidden[1].squeeze().repeat(1, 1, 1)

                # encoded chosen argument
                input = encoded_fact_pool[arg].unsqueeze(0).unsqueeze(0)
                # print(input.shape)

                # renew candidates                
                hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                hiddenl = [hc.unsqueeze(0) for _ in split_fact_pool]

                hiddenl = torch.cat(hiddenl)

                # size: (len(fact_pool), 512)
                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(device)

            arg_pool.append(arg_step_probs)

            tac = tactic_pool[tac]
            arg = [candidate_args[i] for i in arg_step]

            tactic = env.assemble_tactic(tac, arg)
            
        action = (fringe.item(), 0, tactic)
        try:
            # when step is performed, env.history (probably) changes
            reward, done = env.step(action)

        except:
            print("Step exception raised.")
            # print("Fringe: {}".format(env.history))
            print("Handling: {}".format(env.handling))
            print("Using: {}".format(env.using))
            # try again
            counter = env.counter
            frequency = env.frequency
            env.close()
            print("Reverting to the previous game ...")
            env = HolEnv(GAMES, counter-2, frequency)
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
            
        # state_pool.append(state)
        reward_print.append(reward)
        # reward_pool.append(reward+trade_off*entropy)
        reward_pool.append(reward)

        # pg = ng

        steps += 1
        
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Rewards: {}".format(reward_print))
            print("Tactics: {}".format(action_pool))
            # print("Mean reward: {}".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_print)))
            print("Proof trace: {}".format(extract_proof(env.history)))
            # exit()
            proved += 1
            # traces.append(env.history)
            break

        if t == 49:
            print("Failed.")
            print("Rewards: {}".format(reward_print))
            # print("Rewards: {}".format(reward_pool))
            print("Tactics: {}".format(action_pool))
            # print("Mean reward: {}\n".format(np.mean(reward_pool)))
            print("Total: {}".format(np.sum(reward_print)))
            break

        # state = next_state
        # state = state.to(device)
    if record_proofs:
        traces.append(env.history)
    stop = timeit.default_timer()
    print('Time: {}  '.format(stop - start))

    # for i in range(steps):
    #     fringe_p = -fringe_pool[i]
        
    #     arg_p = -torch.sum(torch.stack(arg_pool[i]))

    #     tac_p = -tac_pool[i]
    #     entropy = fringe_p + tac_p + arg_p
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
        
        optimizer_context.zero_grad()
        optimizer_tac.zero_grad()
        optimizer_arg.zero_grad()
        optimizer_term.zero_grad()

        
        for i in range(steps):
            # size : (1,1,4,128)
            total_loss = 0
            
            # state = state_pool[i]
            reward = reward_pool[i]

            fringe_loss = -fringe_pool[i] * (reward)
            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            
            tac_loss = -tac_pool[i] * (reward)
            
            # entropy = fringe_pool[i] + torch.sum(torch.stack(arg_pool[i])) + tac_pool[i]
            
            # loss = fringe_loss + tac_loss + arg_loss + trade_off*entropy
            loss = fringe_loss + tac_loss + arg_loss
            # total_loss += loss
            
            loss.backward()

        # total_loss.backward()

        # optimizer.step()
        
        # optimizer_context.step()
        # optimizer_tac.step()
        # optimizer_arg.step()
        # optimizer_term.step()

    print("Memory: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    # state_pool = []
    fringe_pool = []
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
    # torch.cuda.empty_cache()
    
    # if (e+1) % (len(GAMES)) == 0:
    #     torch.save({
    #         'tac_net_state_dict': tac_net.state_dict(),
    #         'arg_net_state_dict': arg_net.state_dict(),
    #         'term_net_state_dict': term_net.state_dict(),
    #         'optimizer_tac_state_dict': optimizer_tac.state_dict(),
    #         'optimizer_arg_state_dict': optimizer_arg.state_dict(),
    #         'optimizer_term_state_dict': optimizer_term.state_dict(),
    #     }, fn)
    #     print("Models saved.")
    #     fact_json = json.dumps(fact_pool)
    #     with open("fact_pool.json","w") as f:
    #         f.write(fact_json)
    #     print("Facts saved.")


# torch.save({
#     'context_net_state_dict': context_net.state_dict(),
#     'tac_net_state_dict': tac_net.state_dict(),
#     'arg_net_state_dict': arg_net.state_dict(),
#     'term_net_state_dict': term_net.state_dict(),
#     'optimizer_context_state_dict': optimizer_context.state_dict(),
#     'optimizer_tac_state_dict': optimizer_tac.state_dict(),
#     'optimizer_arg_state_dict': optimizer_arg.state_dict(),
#     'optimizer_term_state_dict': optimizer_term.state_dict(),
# }, fn)

print("Models saved.")

# dict_json = json.dumps(dictionary)
# with open("dict.json","w") as f:
#     f.write(dict_json)

# print("Dictionary saved.")
# fact_json = json.dumps(fact_pool)
# with open("fact_pool.json","w") as f:
#     f.write(fact_json)
# print("Facts saved.")

# new_fact_json = json.dumps(list(new_facts.values()))
# with open("new_facts.json","w") as f:
#     f.write(new_fact_json)
# print("Proved theorems saved.")
# # print(env.frequency)

# freq_json = json.dumps(env.frequency)
# with open("frequency.json","w") as f:
#     f.write(freq_json)

# if record_proofs:
#     traces_json = json.dumps(traces)
#     with open("traces.json","w") as f:
#         f.write(traces_json)

    # with open("traces.json") as f:
    #     ttt = json.load(f)
    #     print(ttt[9])


# if record_proofs:
#     # example = traces[-1]
#     for example in traces:
#         draw_tree(example, True)

env.close()
