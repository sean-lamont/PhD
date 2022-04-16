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
from new_env import *
from deeper_seq2seq_sets_model import *
import json
import timeit

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# context_net = ContextPolicy()

# tac_net = TacPolicy(len(tactic_pool))

# arg_net = ArgPolicy(len(tactic_pool), 256)

# term_net = TermPolicy(len(tactic_pool), 256)

# context_net = context_net.to(device)
# tac_net = tac_net.to(device)
# arg_net = arg_net.to(device)
# term_net = term_net.to(device)

# total_params = sum(p.numel() for p in term_net.parameters() if p.requires_grad)

# reward_pool = [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,1]
# gamma = 0.99

# running_add = 0
# for i in reversed(range(len(reward_pool))):
#     if reward_pool[i] == 0:
#         running_add = 0
#     else:
#         running_add = running_add * gamma + reward_pool[i]
#         reward_pool[i] = running_add

# # Normalize reward
# reward_mean = np.mean(reward_pool)
# reward_std = np.std(reward_pool)
# for i in range(len(reward_pool)):
#     reward_pool[i] = (reward_pool[i] - reward_mean) / (reward_std + np.finfo(np.float32).eps) # eps is important, otherwise we

# print(reward_pool)
# print(np.sum(reward_pool))

# rnn = nn.LSTM(1, 64, 1)
# # input = torch.randn(30, 1, 1)
# input = torch.tensor([0,1,3,4,99,77,23,45,12,21,0,1,3,4,99,77,23,45,12,21,0,1,3,4,99,77,23,45,12,21], dtype=torch.float)
# input = input.view(-1,1,1)
# h0 = torch.randn(1, 1, 64)
# c0 = torch.randn(1, 1, 64)
# output, (hn, cn) = rnn(input, (h0, c0))

# try:
#     with open("database.json") as f:
#         database = json.load(f)
#     print("Database loaded.")
# except:
#     failures = {}
#     print("No initial replays.")

# try:
#     with open("typed_database.json") as f:
#         typed_database = json.load(f)
#     print("Typed database loaded.")
# except:
#     failures = {}
#     print("No initial replays.")

# try:
#     with open("provable_subgoals.json") as f:
#         provable_subgoals = json.load(f)
#     print("Provable subgoals loaded.")
# except:
#     failures = {}
#     print("No initial replays.")
    

# try:
#     with open("/home/minchao/supp_dataset.json") as f:
#     # with open("/scratch1/wu099/temp/2021_01_05_16_22_29_checkpoint/replays.json") as f:
#         training = json.load(f)
#     print("Replays loaded.")
# except:
#     replays = {}
#     print("No initial replays.")

with open("humandata/human_list.json") as f:
# with open("/scratch1/wu099/temp/2021_01_05_16_22_29_checkpoint/replays.json") as f:
    human = json.load(f)
print("Human data loaded.")

try:
    with open("humandata/human_list.json") as f:
    # with open("/scratch1/wu099/temp/2021_01_05_16_22_29_checkpoint/replays.json") as f:
        human = json.load(f)
    print("Human data loaded.")
except:
    human = {}
    print("No human data.")

    
# try:
#     with open("/scratch1/wu099/temp/2021_01_13_13_38_57_checkpoint/validation_data.json") as f:
#         validation = json.load(f)
# except:
#     untyped_database = {}


# try:
#     with open("typed_database.json") as f:
#         typed_database = json.load(f)
# except:
#     typed_database = {}


# dataset = training +  validation

# dataset_json = json.dumps(dataset)
# with open("supp_dataset.json","w") as f:
#     f.write(dataset_json)

# print("Dataset saved.")



# trans_names = [value[1] for key, value in untyped_database.items() if value[4] in list(pred_set_replays.keys())[291:] and value[0] == "pred_set"]
# mytrain = [value[4] for key, value in database.items() if value[1] in trans_names]

# replays = {key : [value[0]] for key, value in pred_set_replays.items()}

# count = 0
# for i in list(replays.keys())[291:]:
#     # print(len(replays[i]))
#     if len(replays[i]) == 1:
#         count += 1
# print(count)
# len(replays[list(replays.keys())[291]]);


# test_history = [{'content': [{'polished': {'assumptions': [], 'goal': '@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'}, 'plain': {'assumptions': [], 'goal': 'REVERSE l = [] ⇔ l = []'}}], 'parent': None, 'goal': None, 'by_tactic': '', 'reward': None}, {'content': [{'polished': {'assumptions': [], 'goal': '@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Clist$NIL Clist$NIL @ @ Cmin$= Clist$NIL Clist$NIL'}, 'plain': {'assumptions': [], 'goal': 'REVERSE [] = [] ⇔ [] = []'}}, {'polished': {'assumptions': ['@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'], 'goal': '@ Cbool$! | Vh @ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE @ @ Clist$CONS Vh Vl Clist$NIL @ @ Cmin$= @ @ Clist$CONS Vh Vl Clist$NIL'}, 'plain': {'assumptions': ['REVERSE l = [] ⇔ l = []'], 'goal': '∀h. REVERSE (h::l) = [] ⇔ h::l = []'}}], 'parent': 0, 'goal': 0, 'by_tactic': 'Induct_on `l`', 'reward': 0.1}, {'content': [{'polished': {'assumptions': ['@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'], 'goal': '@ Cbool$! | Vh @ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE @ @ Clist$CONS Vh Vl Clist$NIL @ @ Cmin$= @ @ Clist$CONS Vh Vl Clist$NIL'}, 'plain': {'assumptions': ['REVERSE l = [] ⇔ l = []'], 'goal': '∀h. REVERSE (h::l) = [] ⇔ h::l = []'}}], 'parent': 1, 'goal': 0, 'by_tactic': 'fs[EL_simp_restricted, LENGTH_ZIP, SUM_eq_0, LENGTH_EQ_NUM, FILTER_NEQ_NIL]', 'reward': 0.2}, {'content': [], 'parent': 2, 'goal': 0, 'by_tactic': 'fs[APPEND_LENGTH_EQ, LIST_REL_rules, FILTER_NEQ_NIL, REVERSE_11, LENGTH_REVERSE]', 'reward': 5}]

# t = list(pred_set_replays.keys())[1]
# draw_tree(pred_set_replays[t][0][0])
