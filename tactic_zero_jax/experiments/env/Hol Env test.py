#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
import haiku as hk
import optax
import sys
from time import sleep
import json
import pexpect
import re
import timeit


# In[2]:


HOLPATH = "/home/sean/Documents/PhD/HOL4/HOL/bin/hol --maxheap=256"
TARGET_THEORIES = ["pred_set"] #["arithmetic"]#["list"] #["rich_list"] # ["integer"] #["arithmetic"] # ["rich_list"] #["pred_set"]


# In[3]:


with open("typed_database.json") as f:
    database = json.load(f)


# In[4]:


#goals = [value[4] for key, value in database.items() if value[3] == "thm" and value[0] in TARGET_THEORIES]


# In[5]:


#goals


# In[6]:


def get_polish(raw_goal):
        goal = construct_goal(raw_goal)
        process.sendline(goal.encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("top_goals();".encode("utf-8"))
        process.expect("val it =")
        process.expect([": goal list", ":\r\n +goal list"])

        polished_raw = process.before.decode("utf-8")
        polished_subgoals = re.sub("“|”","\"", polished_raw)
        polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

        # print("content:{}".format(subgoals))
        # exit()
        pd = eval(polished_subgoals)
        
        process.expect("\r\n>")
        process.sendline("drop();".encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        process.expect("\r\n>")

        data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                 "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data # list(zip(pd, [([], raw_goal)]))f

def construct_goal(goal):
    s = "g " + "`" + goal + "`;"
    return s


# In[7]:


process = pexpect.spawn(HOLPATH)

theories = ["listTheory", "bossLib"]

print("Importing theories...")
process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
for i in theories:
    process.sendline("open {};".format(i).encode("utf-8"))


# In[8]:



# # remove built-in simp lemmas
# print("Removing simp lemmas...")
# process.sendline("delsimps [\"HD\", \"EL_restricted\", \"EL_simp_restricted\"];")
# #process.sendline("delsimps {};".format(dels))
# #process.sendline("delsimps {};".format(dels2))
# # process.sendline("delsimps {};".format(dels3))
# sleep(4)
# # load utils
# print("Loading modules...")
# process.sendline("use \"helper.sml\";")
# sleep(5)
# # process.sendline("val _ = load \"Timeout\";")
# print("Configuration done.")
# process.expect('\r\n>')
# # process.readline()
# process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))

# # consumes hol4 head
# process.expect('\r\n>')
# #print (process.read())
# # setup the goal
# goal = goals[0]
                      
#  # a pair of polished goal list and original goal list
# fringe = get_polish(goal)

# #scripts = []
# goal = construct_goal(goal)
# process.sendline(goal.encode("utf-8"))        
# print("Initialization done. Main goal is:\n{}.".format(goal))


# In[9]:


# print (fringe)


# In[10]:


tactic_zero_path = "/home/sean/Documents/PhD/git/repo/PhD/tacticzero/holgym/"


# In[11]:


test_history = [{'content': [{'polished': {'assumptions': [], 'goal': '@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'}, 'plain': {'assumptions': [], 'goal': 'REVERSE l = [] ⇔ l = []'}}], 'parent': None, 'goal': None, 'by_tactic': '', 'reward': None}, {'content': [{'polished': {'assumptions': [], 'goal': '@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Clist$NIL Clist$NIL @ @ Cmin$= Clist$NIL Clist$NIL'}, 'plain': {'assumptions': [], 'goal': 'REVERSE [] = [] ⇔ [] = []'}}, {'polished': {'assumptions': ['@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'], 'goal': '@ Cbool$! | Vh @ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE @ @ Clist$CONS Vh Vl Clist$NIL @ @ Cmin$= @ @ Clist$CONS Vh Vl Clist$NIL'}, 'plain': {'assumptions': ['REVERSE l = [] ⇔ l = []'], 'goal': '∀h. REVERSE (h::l) = [] ⇔ h::l = []'}}], 'parent': 0, 'goal': 0, 'by_tactic': 'Induct_on `l`', 'reward': 0.1}, {'content': [{'polished': {'assumptions': ['@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL'], 'goal': '@ Cbool$! | Vh @ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE @ @ Clist$CONS Vh Vl Clist$NIL @ @ Cmin$= @ @ Clist$CONS Vh Vl Clist$NIL'}, 'plain': {'assumptions': ['REVERSE l = [] ⇔ l = []'], 'goal': '∀h. REVERSE (h::l) = [] ⇔ h::l = []'}}], 'parent': 1, 'goal': 0, 'by_tactic': 'fs[EL_simp_restricted, LENGTH_ZIP, SUM_eq_0, LENGTH_EQ_NUM, FILTER_NEQ_NIL]', 'reward': 0.2}, {'content': [], 'parent': 2, 'goal': 0, 'by_tactic': 'fs[APPEND_LENGTH_EQ, LIST_REL_rules, FILTER_NEQ_NIL, REVERSE_11, LENGTH_REVERSE]', 'reward': 5}]


# In[12]:


def gather_encoded_content_(history, encoder):
    # figure out why this is slower than tests
    # figured out: remember to do strip().split()
    fringe_sizes = []
    contexts = []
    reverted = []
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        g = revert_with_polish(e)
        reverted.append(g.strip().split())
    out = []
    sizes = []
    for goal in reverted:
        out_, sizes_ = encoder.encode([goal])
        out.append(torch.cat(out_.split(1), dim=2).squeeze(0))
        sizes.append(sizes_)
        
    # s1 = timeit.default_timer()
    #out, sizes = batch_encoder.encode(reverted)
    
    representations = out
    # merge two hidden variables
    #representations = torch.cat(out.split(1), dim=2).squeeze(0)
    # print(representations.shape)
    # s2 = timeit.default_timer()    
    # print(s2-s1)

    return representations, contexts, fringe_sizes


# In[13]:


# ARG_LEN = 5 
# EXCLUDED_THEORIES = ["min"] #["min", "bool"]
# def replay_known_proof(known_history, arg_len=ARG_LEN):
#     print("\nReplaying a known proof of {} ...".format(known_history[0]["content"][0]["plain"]["goal"]))

#     # compute the encoded fact pool at the beginning of each episode
#     polished_goal = known_history[0]["content"][0]["polished"]["goal"]
#     allowed_theories = parse_theory(polished_goal)
#     allowed_theories = [t for t in allowed_theories if t not in EXCLUDED_THEORIES]
    
#     try:
#         allowed_arguments_ids = []
#         candidate_args = []
#         goal_theory = database[polished_goal][0]
#         for i,t in enumerate(database):
#             if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
#                 allowed_arguments_ids.append(i)
#                 candidate_args.append(t)

#     except:
#         allowed_arguments_ids = []
#         candidate_args = []
#         for i,t in enumerate(database):
#             if database[t][0] in allowed_theories:
#                 allowed_arguments_ids.append(i)
#                 candidate_args.append(t)
#         print("Theorem not found in database.")

#     #true_resulting_fringe = known_history[t+1]

#     #print (known_history)

#     representations, context_set, fringe_sizes = gather_encoded_content_(known_history, batch_encoder)

#     return representations, context_set, fringe_sizes

    
def parse_theory(pg):
    theories = re.findall(r'C\$(\w+)\$ ', pg)
    theories = set(theories)
    for th in EXCLUDED_THEORIES:
        theories.discard(th)
    return list(theories)

def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions): 
        goal = "@ @ Dmin$==> {} {}".format(i, goal)
    return goal 

def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs


# In[14]:


from new_env import *


with open("include_probability.json") as f:
    #database  of all theories up to probability?
    #keys are polish notation expressions, values give list with [theory/library, theory_name, number in lib, def/thm, utf encoded expression]
    database = json.load(f)

#all theories in database
#TARGET_THEORIES = ["probability", "martingale", "lebesgue", "borel", "real_borel", "sigma_algebra","util_prob", "fcp", "indexedLists", "rich_list", "list", "pred_set","numpair", "basicSize", "numeral", "arithmetic", "prim_rec", "num","marker", "bool", "min", "normalForms", "relation", "sum", "pair", "sat","while", "bit", "logroot", "transc", "powser", "lim", "seq", "nets","metric", "real", "realax", "hreal", "hrat", "quotient_sum", "quotient","res_quan", "product", "iterate", "cardinal", "wellorder","set_relation", "derivative", "real_topology"]
TARGET_THEORIES = ["bool", "min", "list"]
GOALS = [(key, value[4]) for key, value in database.items() if value[3] == "thm" and value[0] in TARGET_THEORIES]

len(GOALS)

with open("polished_def_dict.json") as f:
    defs = json.load(f)

fact_pool = list(defs.keys())



#parse theory
g = GOALS[19][1]


print (g)

env = HolEnv(g)


theories = re.findall(r'C\$(\w+)\$ ', GOALS[19][0])
theories = set(theories)
theories = list(theories)

print (theories)

allowed_theories = theories

goal_theory = g

print ("Target goal: {}".format(g))
try:
    allowed_arguments_ids = []
    candidate_args = []
    goal_theory = g#database[polished_goal][0] # plain_database[goal][0]
    for i,t in enumerate(database):
        if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
            allowed_arguments_ids.append(i)
            candidate_args.append(t)

    env.toggle_simpset("diminish", goal_theory)
    print("Removed simpset of {}".format(goal_theory))

except:
    allowed_arguments_ids = []
    candidate_args = []
    for i,t in enumerate(database):
        if database[t][0] in allowed_theories:
            allowed_arguments_ids.append(i)
            candidate_args.append(t)
    print("Theorem not found in database.")

print ("Number of candidate facts to use: {}".format(len(candidate_args)))


# print("Facts: {}".format(len(allowed_arguments_ids)))

# state = env.history

#encoded database for representations of all terms in database (easier than computing batch encoding for arguments each iteration)

encoded_database = torch.load('encoded_include_probability.pt')

encoded_fact_pool = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids))

print (env.history)


# In[15]:


# import os
# import argparse
# import logging
# import timeit
# import torch

# import seq2seq
# #from seq2seq.evaluator import BatchPredictor
# #from seq2sebq.util.checkpoint import Checkpoint

# from batch_predictor import BatchPredictor
# from checkpoint import Checkpoint

# #checkpoint_path = "models/2021_02_21_15_46_04" # 98% accuracy model, up to probability theory
# checkpoint_path = "models/2020_04_26_20_11_28" # 95% accuracy model, core theories + integer + sorting

# #checkpoint_path = "models/2021_02_22_16_07_03" # 97-98% accuracy model, up to and include probability theory#
# #checkpoint_path = "models/2020_09_24_23_38_06" # 98% accuracy model, core theories + integer + sorting | separate theory tokens


# # logging.info("loading checkpoint from {}".format(checkpoint_path))

# checkpoint = Checkpoint.load(checkpoint_path)
# seq2seq = checkpoint.model
# input_vocab = checkpoint.input_vocab
# output_vocab = checkpoint.output_vocab

# batch_encoder = BatchPredictor(seq2seq, input_vocab, output_vocab)
# batch_encoder


# In[16]:


import os
import argparse
import logging
import timeit
import torch

import seq2seq
#from seq2seq.evaluator import BatchPredictor
#from seq2sebq.util.checkpoint import Checkpoint

from batch_predictor import BatchPredictor
from checkpoint import Checkpoint



    
    #checkpoint_path = "models/2020_04_22_20_36_50" # 91% accuracy model, only core theories
#checkpoint_path = "models/2020_04_26_20_11_28" # 95% accuracy model, core theories + integer + sorting
#checkpoint_path = "models/2020_09_24_23_38_06" # 98% accuracy model, core theories + integer + sorting | separate theory tokens
#checkpoint_path = "models/2020_11_28_16_45_10" # 96-98% accuracy model, core theories + integer + sorting + real | separate theory tokens

checkpoint_path = "models/2020_12_04_03_47_22" # 97% accuracy model, core theories + integer + sorting + real + bag | separate theory tokens

#checkpoint_path = "models/2021_02_21_15_46_04" # 98% accuracy model, up to probability theory

#checkpoint_path = "models/2021_02_22_16_07_03" # 97-98% accuracy model, up to and include probability theory

checkpoint = Checkpoint.load(checkpoint_path)
print (checkpoint)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

batch_encoder_ = BatchPredictor(seq2seq, input_vocab, output_vocab)



jax_reps, context_set, fringe_sizes = gather_encoded_content_(env.history, batch_encoder_)



print (jax_reps, context_set, fringe_sizes)


# In[17]:


#jax_reps, context_set, fringe_sizes = replay_known_proof(test_history)


# In[18]:


print (context_set, fringe_sizes)


# In[19]:


from policy_networks import *
import policy_networks


# In[20]:


init_context, apply_context = hk.transform(policy_networks._context_forward)
rng_key = random.PRNGKey(100)

#convert to jax
jax_reps = jnp.stack([jnp.array(jax_reps[i][0]) for i in range(len(jax_reps))])


context_params = init_context(rng_key, jax_reps)
apply_context = jax.jit(apply_context)


# In[21]:


# context_scores = apply_context(context_params, rng_key, jax_reps)
# print (context_scores)


# In[22]:



#function to give the log probability of pi(f | s) so gradient can be computed directly
#also returns sampled index and contexts to determine goal to give tactic network
def sample_fringe(context_params, context_net, rng_key, jax_reps, context_set, fringe_sizes):
    context_scores = context_net(context_params, rng_key, jax_reps)
    contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
    fringe_scores = []
    for s in scores_by_fringe:
        fringe_score = jnp.sum(s)
        fringe_scores.append(fringe_score)
    #TODO some fringes can be empty, but still give value 0 which assigns nonzero probability?
    fringe_scores = jnp.stack(fringe_scores)
    fringe_probs = jax.nn.softmax(fringe_scores)

    #samples, gives an index (looks like it does gumbel softmax under the hood to keep differentiability?)
    sampled_idx = random.categorical(rng_key,fringe_probs)

    prob = fringe_probs[sampled_idx]
    log_prob = jnp.log(prob)
    return log_prob, (sampled_idx, contexts_by_fringe)
                                                           
grad_log_context, (fringe_idx, contexts_by_fringe) = jax.grad(sample_fringe, has_aux=True)(context_params, apply_context, rng_key, jax_reps, context_set, fringe_sizes)
print (grad_log_context, fringe_idx, contexts_by_fringe)


# In[23]:


target_context = contexts_by_fringe[fringe_idx][0]
target_goal = target_context["polished"]["goal"]
target_representation = jax_reps[context_set.index(target_context)]
# print(target_representation.shape)
# exit()

# size: (1, max_contexts, max_assumptions+1, max_len)
# tac_input = target_representation.unsqueeze(0)
# tac_input = tac_input.to(device)

# # compute scores of tactics
# tac_probs = tac_net(tac_input)
# # print(tac_probs)
# tac_m = Categorical(tac_probs)
# tac = tac_m.sample()

# print (target_context)
# print (target_goal)
#print (target_representation)


# In[24]:


MORE_TACTICS = True
if not MORE_TACTICS:
    thms_tactic = ["simp", "fs", "metis_tac"]
    thm_tactic = ["irule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac"]
else:
    thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    thm_tactic = ["irule", "drule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac", "EQ_TAC"]
    
    # thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    # thm_tactic = [] #["irule", "drule"] 
    # term_tactic = ["Induct_on"]
    # no_arg_tactic = ["strip_tac", "EQ_TAC", "simp[]", "rw[]", "metis_tac[]", "fs[]"]

tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic
print (len(tactic_pool))


# In[25]:


init_tac, apply_tac = hk.transform(policy_networks._tac_forward)

tactic_params = init_tac(rng_key, jnp.expand_dims(target_representation,0), len(tactic_pool))
apply_tac = partial(jax.jit, static_argnums=3)(apply_tac)


# In[26]:


#takes a goal encoding and samples tactic from network, and returns log prob for gradient 
def sample_tactic(tactic_params, tac_net, rng_key, goal_endcoding, action_size=len(tactic_pool)):
    tac_probs = tac_net(tactic_params, rng_key, goal_endcoding, action_size)[0]
    tac_idx = random.categorical(rng_key, tac_probs)
    log_prob = jnp.log(tac_probs[tac_idx])
    return log_prob, tac_idx

grad_log_tac, tac_idx = jax.grad(sample_tactic, has_aux=True)(tactic_params, apply_tac, rng_key, jnp.expand_dims(target_representation,0), len(tactic_pool))

sampled_tac = tactic_pool[tac_idx]


# In[27]:


#for testing

#if tactic requires no argument
if sampled_tac in no_arg_tactic:
    full_tactic = tactic_pool[tac]
    
#Induct_on case; use term policy to find which term to induct on 
elif sampled_tac in term_tactic:
    goal_tokens = target_goal.split()
    term_tokens = [[t] for t in set(goal_tokens) if t[0] == "V"]
    #add conditional if tokens is empty 
    
    


# In[28]:


goal_tokens = target_goal.split()
#goal_tokens = dict.fromkeys(goal_tokens): set should be better?
term_tokens = [[t] for t in set(goal_tokens) if t[0] == "V"]

print (term_tokens)


# In[29]:


#now want encodings for terms from AE

term_reps = []

for term in term_tokens:
    term_rep, _ = batch_encoder_.encode([term])
    #output is bidirectional so concat vectors
    term_reps.append(torch.cat(term_rep.split(1), dim=2).squeeze(0))

# convert to jax
term_reps = jnp.stack([jnp.array(term_reps[i][0]) for i in range(len(term_reps))])
term_reps


# In[30]:


# now want inputs to term_net to be target_representation (i.e. goal) concatenated with terms
# models the policies conditional dependence of the term given the goal

#stack goal representation for each token
goal_stack = jnp.concatenate([jnp.expand_dims(target_representation,0) for _ in term_tokens])

#concat with term encodings to give candidate matrix
candidates = jnp.concatenate([goal_stack, term_reps], 1)

print (candidates.shape)


# In[31]:


# init_term, apply_term = hk.transform(policy_networks._term_forward)

# #TODO why do we need the tactic? The use of term net implicitly assumes we are using induct_on, and every time this is called the tactic index will be the same?

# term_params = init_term(rng_key, candidates, tac_idx, len(tactic_pool), candidates.shape[1])
# apply_term = partial(jax.jit, static_argnums=(4,5))(apply_term)


# In[32]:


# #takes a goal encoding and samples tactic from network, and returns log prob for gradient 
# def sample_term(term_params, term_net, rng_key, candidates, tactic,TAC_SIZE, MAX_LEN):
#     term_scores = term_net(term_params, rng_key, candidates, tactic, TAC_SIZE, MAX_LEN)
#     term_scores = jnp.ravel(term_scores)
#     term_probs = jax.nn.softmax(term_scores)
#     term_idx = random.categorical(rng_key, term_probs)
#     log_prob = jnp.log(term_probs[term_idx])
#     return log_prob, term_idx

# grad_log_term, tac_term = jax.grad(sample_term, has_aux=True)(term_params, apply_term, rng_key, candidates, tac_idx, len(tactic_pool), candidates.shape[1])

# print (grad_log_term, tac_term)


# In[33]:


#term policy implementation without tactic as an argument (since it will always be Induct_on)
#policy will now learn a mapping for V(term | goal)

init_term, apply_term = hk.transform(policy_networks._term_no_tac_forward)
term_params = init_term(rng_key, candidates)
apply_term = jax.jit(apply_term)


# In[34]:


def sample_term(term_params, term_net, rng_key, candidates):
    term_scores = term_net(term_params, rng_key, candidates)
    term_scores = jnp.ravel(term_scores)
    term_probs = jax.nn.softmax(term_scores)
    term_idx = random.categorical(rng_key, term_probs)
    log_prob = jnp.log(term_probs[term_idx])
    return log_prob, term_idx

grad_log_term, term_idx = jax.grad(sample_term, has_aux=True)(term_params, apply_term, rng_key, candidates)#, tac_idx, len(tactic_pool), candidates.shape[1])

#print (grad_log_term, term_idx)


# In[35]:


sampled_term = term_tokens[term_idx]

tm = sampled_term[0][1:] # remove headers, e.g., "V" / "C" / ...

if tm:
    tactic = "Induct_on `{}`".format(tm)
else:
    # only to raise an error
    tactic = "Induct_on"


# In[36]:


#get candidate arguments

with open("polished_def_dict.json") as f:
    defs = json.load(f)

fact_pool = list(defs.keys())
fact_pool


# In[39]:


encoded_database = torch.load('encoded_include_probability.pt')

encoded_fact_pool = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids))
encoded_fact_pool = jnp.array(encoded_fact_pool)


# In[40]:


# argument network for tactics with one or multiple arguments

   
hidden = jnp.expand_dims(target_representation,0)
# hidden = (hidden0, hidden1)

# concatenate the candidates with hidden states (i.e. goal).

goal_stack = jnp.concatenate([jnp.expand_dims(target_representation,0) for _ in encoded_fact_pool])


candidates = jnp.concatenate([encoded_fact_pool, goal_stack], 1)


init_state = hk.LSTMState(hidden, hidden)

#init_state
tac_idx


# In[41]:


init_arg, apply_arg = hk.transform(policy_networks._arg_forward)
arg_params = init_arg(rng_key, tac_idx, candidates, init_state, len(tactic_pool), 256)

apply_arg = partial(jax.jit, static_argnums=(5,6))(apply_arg)


# In[42]:


#out_arg = apply_arg(arg_params,  rng_key, tac_idx, candidates, init_state, len(tactic_pool), 256)

#function for sampling single argument given previous arguments, 
def sample_arg(arg_params, arg_net, rng_key, input_, candidates, hidden, tactic_size, embedding_dim):
    hidden, arg_scores = arg_net(arg_params, rng_key, input_, candidates, hidden, tactic_size, embedding_dim)
    arg_scores = jnp.ravel(arg_scores)
    arg_probs = jax.nn.softmax(arg_scores)
    arg_idx = random.categorical(rng_key, arg_probs)
    log_prob = jnp.log(arg_probs[arg_idx])
    return log_prob, (arg_idx, hidden)


# In[43]:


#print (init_state)
# run through first with tac_idx to initialise state with tactic as c_0
_, (_, hidden) = jax.grad(sample_arg, has_aux=True)(arg_params, apply_arg, rng_key, tac_idx, candidates, init_state, len(tactic_pool), 256)
#print (hidden)
# Q: Do we need to make the hidden state the goal again, now with updated cell from first pass? or is new hidden fine?


# In[44]:


#log gradient of context policy for all arguments will be sum of log gradients of each argument.
#hence can just take the sum over gradients one at a time 

#should we add an argument terminating character or just let it run to arg_len?


# if tactic_pool[true_tac] in thm_tactic:
#     arg_len = 1
# else:
#     arg_len = ARG_LEN

def run_arg(arg_params, apply_arg, rng_key, input_, candidates, hidden, tac_size, encoding_size):
    gradients = []
    arg_inds = []
    ARG_LEN = 5
    #input_ = tac_idx
    for _ in range(ARG_LEN):
        grad_log_arg, (arg_idx, hidden) = jax.grad(sample_arg, has_aux=True)(arg_params, apply_arg, rng_key, 77, candidates, hidden, len(tactic_pool), 256)
        gradients.append(grad_log_arg)
        arg_inds.append(arg_idx)
        input_ = jnp.expand_dims(encoded_fact_pool[arg_idx], 0)
        #renewing candidates sems redundant from old code?
    return gradients, arg_inds

run_arg(arg_params, apply_arg, rng_key, tac_idx, candidates, hidden, len(tactic_pool), 256)


# In[45]:


#can multiply learning rate by reward for each update to give the policy gradient after grad of log probs is done 
context_lr = 1e-2
tactic_lr = 1e-2 
arg_lr = 1e-2
term_lr = 1e-2

context_optimiser = optax.rmsprop(context_lr)
tactic_optimiser = optax.rmsprop(tactic_lr)
arg_optimiser = optax.rmsprop(arg_lr)
term_optimiser = optax.rmsprop(term_lr)

opt_state_context = context_optimiser.init(context_params)
opt_state_tactic = tactic_optimiser.init(tactic_params)
opt_state_arg = arg_optimiser.init(arg_params)
opt_state_term = term_optimiser.init(term_params)



# In[51]:


#test updating parameters by multiplying learning rate with return

test_return = 5

scaled_lr = term_lr * test_return 

term_optimiser = optax.rmsprop(scaled_lr)

opt_state_term = term_optimiser.init(term_params)


updates, opt_state_term = context_optimiser.update(grad_log_term, opt_state_term)
term_params = optax.apply_updates(term_params, updates)

print (term_params)

new_return = 8

scaled_lr_new = term_lr * new_return 

new_term_optimiser = optax.rmsprop(scaled_lr_new)

#keep old state but apply to new optimiser lr

updates, opt_state_term = new_term_optimiser.update(grad_log_term, opt_state_term)
term_params = optax.apply_updates(term_params, updates)
term_params


# In[ ]:


#can multiply learning rate by reward for each update to give the policy gradient after grad of log probs is done 
# context_lr = 1e-2
# # tactic_lr = 1e-2 
# # arg_lr = 1e-2
# # term_lr = 1e-2

# context_optimiser = optax.rmsprop(context_lr)
# # tactic_optimiser = optax.rmsprop(tactic_lr)
# # arg_optimiser = optax.rmsprop(arg_lr)
# # term_optimiser = optax.rmsprop(term_lr)

# opt_state_context = context_optimiser.init(context_params)
# # opt_state_tactic = tactic_optimiser.init(tactic_params)
# # opt_state_arg = arg_optimiser.init(arg_params)
# # opt_state_term = term_optimiser.init(term_params)





#gradient example. May need to construct separate function for each net during training 

# def compute_probs(params, net, *args):
#     probs = jax.nn.softmax(jnp.ravel(net(params,*args)))
#     logits = jnp.log(probs)
#     ind = random.categorical(rng_key, logits)
#     log_prob = logits[ind]
#     return log_prob

# grad = jax.grad(compute_probs)(context_params, apply_context, rng_key, jax_reps)#c_term, x_arg,TAC_SIZE, MAX_LEN)

# # updates, opt_state_term = context_optimiser.update(grad, opt_state_context)
# context_params = optax.apply_updates(context_params, updates)


# context_params


# In[ ]:


get_polish('@ @ Cmin$= @ @ Cmin$= @ Clist$REVERSE Vl Clist$NIL @ @ Cmin$= Vl Clist$NIL')


# In[ ]:


batch_encoder


# In[ ]:




