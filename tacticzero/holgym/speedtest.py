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
from new_env import *
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
record_subproofs = False
traces = []
subproofs_traces = []
proof_check_failure = []

subgoals_library = {}
train_subgoals = False
iteration_counter = 0

# device = torch.device("cpu")
num_iteration = 1
validation_interval = 5
subgoals_training_interval = 1

CONTINUE = False #True
TRAIN_SET = PROVABLES #GOALS
VALIDATION_SET = ["∀p y. EXISTS (λx. p) y ⇔ y ≠ [] ∧ p",
                  "[] = REVERSE l ⇔ l = []",
                  "∀x. x = [] ∨ ∃y t. y::t = x",
                  "∀l2 l1 l3. l2 ++ (l1 ++ l3) = l2 ++ l1 ++ l3",
                  "∀M M' v f. M = M' ⇒ (M' = [] ⇒ v = v') ⇒ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
                  "l2 ++ l1 = [e] ⇔ l1 = [e] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [e]",
                  "LAST (h::l) = if l = [] then h else LAST l",
                  "LENGTH l = 0 ⇔ l = []",
                  "¬SHORTLEX R x []",
                  "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃h l. x = h::l ∧ f h l = v'",
                  "∀p l3. EXISTS (λx. p) l3 ⇔  p ∧ l3 ≠ []",
                  "[] = REVERSE l1 ⇔ l1 = []",
                  "x = [] ∨ ∃y t. y::t = x",
                  "l2 ++ (l1 ++ l3) = l2 ++ l1 ++ l3",
                  "M = M' ⇒ (M' = [] ⇒ v = v') ⇒ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
                  "l2 ++ l1 = [x] ⇔ l1 = [x] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [x]",
                  "LAST (h::l2) = if l2 = [] then h else LAST l2",
                  "LENGTH l = 0 ⇔ l = []",
                  "¬SHORTLEX R y []",
                  "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃h l1. x = h::l1 ∧ f h l1 = v'"]

reverse_database = {(value[0], value[1]) : key for key, value in database.items()}

replays = {}

# GAMES = TRAIN_GOALS
# GAMES = ["∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c"]
# GAMES = ["¬SHORTLEX R l []"]
print("Total games: {}".format(len(TRAIN_SET)))

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

arg_net = ArgPolicy(len(tactic_pool), 256)

term_net = TermPolicy(len(tactic_pool), 256)

context_net = context_net.to(device)
tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

optimizer_context = torch.optim.RMSprop(list(context_net.parameters()), lr=context_rate)

optimizer_tac = torch.optim.RMSprop(list(tac_net.parameters()), lr=tac_rate)

optimizer_arg = torch.optim.RMSprop(list(arg_net.parameters()), lr=arg_rate)

optimizer_term = torch.optim.RMSprop(list(term_net.parameters()), lr=term_rate)

if CONTINUE:
    saved = "bigger_2020_09_14_19_37_10.ckpt"
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
    saved_replays = "replays_normalized.json"
    try:
        with open(saved_replays) as f:
            replays = json.load(f)
        print("Replays loaded.")
    except:
        replays = {}
        print("No initial replays.")


# split_fact_pool = [i.strip().split() for i in database]

# batch_size = 20
# s1 = timeit.default_timer()
# out_seq = []
# for i in range(0, len(split_fact_pool), batch_size):
#     out, _ = batch_encoder.encode(split_fact_pool[i:i+batch_size])
#     out_seq.append(out)
# encoded_database = torch.cat(out_seq, dim=1)
# encoded_database = torch.cat(encoded_database.split(1), dim=2).squeeze(0)
# s2 = timeit.default_timer()
# print(s2-s1)
# print(encoded_database.shape)

# torch.save(encoded_database, 'encoded_database.pt')

                    
# encoded_database = torch.load('encoded_database.pt', map_location=device)

# print(encoded_database.shape)

# s1 = timeit.default_timer()

# predicted_theories = ["list"]

# # polished_goal = "@ @ Cmin$= @ @ Cmin$= Cnum$0 @ Clist$LENGTH Vl @ @ Cmin$= Vl Clist$NIL"
# polished_goal = "@ @ C$min$ = @ @ C$min$ = C$num$ 0 @ C$list$ LENGTH Vl @ @ C$min$ = Vl C$list$ NIL"
# EXCLUDED_THEORIES = ["min", "bool"]

# # def parse_theory(pg):
# #     theories = re.findall(r'C(.*?)\$', pg)
# #     theories = set(theories)
# #     for th in EXCLUDED_THEORIES:
# #         theories.discard(th)
# #     return list(theories)

# def parse_theory(pg):
#     theories = re.findall(r'C\$(\w+)\$ ', pg)
#     theories = set(theories)
#     for th in EXCLUDED_THEORIES:
#         theories.discard(th)
#     return list(theories)

# allowed_theories = parse_theory(polished_goal)

# # try:
# #     # allowed_arguments_ids = [i for i,t in enumerate(database) if
# #     #                          database[t][0] in allowed_theories and
# #     #                          int(database[t][2]) < int(database[polished_goal][2])]

# #     allowed_arguments_ids = []
# #     candidate_args = []
# #     for i,t in enumerate(database):
# #         if database[t][0] in allowed_theories and int(database[t][2]) < int(database[polished_goal][2]):
# #             allowed_arguments_ids.append(i)
# #             candidate_args.append(t)
    
# # except:
# #     allowed_arguments_ids = []
# #     candidate_args = []
# #     for i,t in enumerate(database):
# #         if database[t][0] in allowed_theories:
# #             allowed_arguments_ids.append(i)
# #             candidate_args.append(t)

# #     # allowed_arguments_ids = [i for i,t in enumerate(database) if
# #     #                          database[t][0] in allowed_theories]

# try:
#     allowed_arguments_ids = []
#     candidate_args = []
#     for i,t in enumerate(database):
#         if database[t][0] in allowed_theories and int(database[t][2]) < int(database[polished_goal][2]):
#             allowed_arguments_ids.append(i)
#             candidate_args.append(t)
    
# except:
#     allowed_arguments_ids = []
#     candidate_args = []
#     for i,t in enumerate(database):
#         if database[t][0] in allowed_theories:
#             allowed_arguments_ids.append(i)
#             candidate_args.append(t)
#     print("Theorem not found in database.")
    
# # candidate_args = [t for i,t in enumerate(database) if i in allowed_arguments_ids]
# encoded_allowed_arguments = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids))

# s2 = timeit.default_timer()

# print(s2-s1)

# split_fact_pool = [i.strip().split() for i in database]
# out, _ = batch_encoder.encode([split_fact_pool[1027]])
# out = torch.cat(out.split(1), dim=2).squeeze(0)

# # print(torch.eq(out[0], encoded_allowed_arguments[0]))

# with open("back/database.json") as f:
#     old_database = json.load(f)





# try:
#     usable_theorems = [t for t in allowed_arguments if
#                        (allowed_arguments[t][2] < allowed_arguments[polished_goal][2] and
#                         allowed_arguments[t][0] == allowed_arguments[polished_goal][0]) or
#                        allowed_arguments[t][0] != allowed_arguments[polished_goal][0]]
# except:
#     print("Theorem not found in database.")
#     usable_theorems = [t for t in allowed_arguments]




e = HolEnv("T")
s1 = timeit.default_timer()
e.query("∀l2 l1 l3. l2 ++ (l1 ++ l3) = l2 ++ l1 ++ l3", "EQ_TAC")

s2 = timeit.default_timer()

print(s2-s1)
