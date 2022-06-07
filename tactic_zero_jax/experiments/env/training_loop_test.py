import jax
import jax.numpy as jnp
import haiku as hk
import optax

from jax import random
import sys
from time import sleep
import json
import pexpect
import re
import os
import argparse
import logging
import timeit
import torch

import seq2seq
from batch_predictor import BatchPredictor
from checkpoint import Checkpoint

from policy_networks import *
import policy_networks

from new_env import *

from jax.config import config
config.update("jax_debug_nans", True) 
#jax.config.update("jax_enable_x64", False)


HOLPATH = "/home/sean/Documents/hol/HOL/bin/hol --maxheap=256"
#tactic_zero_path = "/home/sean/Documents/PhD/git/repo/PhD/tacticzero/holgym/"

with open("typed_database.json") as f:
    database = json.load(f)



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
    
tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic



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

        pd = eval(polished_subgoals)
        
        process.expect("\r\n>")
        process.sendline("drop();".encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        process.expect("\r\n>")

        data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                 "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data 
    
def construct_goal(goal):
    s = "g " + "`" + goal + "`;"
    return s


# In[6]:


def gather_encoded_content_(history, encoder):
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

    representations = out

    return representations, contexts, fringe_sizes


# In[7]:


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
        #goal = "@ @ D$min$==> {} {}".format(i, goal)
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)

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


# In[8]:


import torch
import sys
print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())


# In[9]:



with open("include_probability.json") as f:
    database = json.load(f)

#all theories in database
#TARGET_THEORIES = ["probability", "martingale", "lebesgue", "borel", "real_borel", "sigma_algebra","util_prob", "fcp", "indexedLists", "rich_list", "list", "pred_set","numpair", "basicSize", "numeral", "arithmetic", "prim_rec", "num","marker", "bool", "min", "normalForms", "relation", "sum", "pair", "sat","while", "bit", "logroot", "transc", "powser", "lim", "seq", "nets","metric", "real", "realax", "hreal", "hrat", "quotient_sum", "quotient","res_quan", "product", "iterate", "cardinal", "wellorder","set_relation", "derivative", "real_topology"]

with open("polished_def_dict.json") as f:
    defs = json.load(f)

fact_pool = list(defs.keys())

encoded_database = torch.load('encoded_include_probability.pt')


TARGET_THEORIES = ["bool", "min", "list"]
GOALS = [(key, value[4]) for key, value in database.items() if value[3] == "thm" and value[0] in TARGET_THEORIES]

print (GOALS[0][1])


# In[10]:


#checkpoint_path = "models/2020_04_22_20_36_50" # 91% accuracy model, only core theories
#checkpoint_path = "models/2020_04_26_20_11_28" # 95% accuracy model, core theories + integer + sorting
#checkpoint_path = "models/2020_09_24_23_38_06" # 98% accuracy model, core theories + integer + sorting | separate theory tokens
#checkpoint_path = "models/2020_11_28_16_45_10" # 96-98% accuracy model, core theories + integer + sorting + real | separate theory tokens
#checkpoint_path = "models/2020_12_04_03_47_22" # 97% accuracy model, core theories + integer + sorting + real + bag | separate theory tokens

checkpoint_path = "models/2021_02_21_15_46_04" # 98% accuracy model, up to probability theory

#checkpoint_path = "models/2021_02_22_16_07_03" # 97-98% accuracy model, up to and include probability theory

checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

batch_encoder_ = BatchPredictor(seq2seq, input_vocab, output_vocab)


# In[11]:


#function to give the log probability of pi(f | s) so gradient can be computed directly
#also returns sampled index and contexts to determine goal to give tactic network
def sample_fringe(context_params, context_net, rng_key, jax_reps, context_set, fringe_sizes):
    context_scores = context_net(context_params, rng_key, jax_reps)
    contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
    fringe_scores = []
    for s in scores_by_fringe:
        fringe_score = jnp.sum(s)
        fringe_scores.append(fringe_score)
    fringe_scores = jnp.stack(fringe_scores)
    fringe_probs = jax.nn.log_softmax(fringe_scores)

    #samples, gives an index (looks like it does gumbel softmax under the hood to keep differentiability?)
    sampled_idx = jax.random.categorical(rng_key,fringe_probs)

    log_prob = fringe_probs[sampled_idx]
    #log_prob = jnp.log(prob)
    return log_prob, (sampled_idx, contexts_by_fringe)
                                                           
#takes a goal encoding and samples tactic from network, and returns log prob for gradient 
def sample_tactic(tactic_params, tac_net, rng_key, goal_endcoding, action_size=len(tactic_pool)):
    tac_scores = tac_net(tactic_params, rng_key, goal_endcoding, action_size)
    tac_scores = jnp.ravel(tac_scores)
    #tac_scores = tac_scores - max(tac_scores)
    #print (tac_scores)
    #subtract max element for numerical stability 
    tac_probs = jax.nn.log_softmax(tac_scores)
    tac_idx = jax.random.categorical(rng_key, tac_probs)
    log_prob = tac_probs[tac_idx]#jnp.log(tac_probs[tac_idx])
    print (jnp.exp(log_prob).primal)#, jnp.exp(tac_probs), rng_key)
    return log_prob, tac_idx

def sample_term(term_params, term_net, rng_key, candidates):
    term_scores = term_net(term_params, rng_key, candidates)
    term_scores = jnp.ravel(term_scores)
    term_probs = jax.nn.log_softmax(term_scores)
    term_idx = jax.random.categorical(rng_key, term_probs)
    log_prob = term_probs[term_idx]#jnp.log(term_probs[term_idx])
    return log_prob, term_idx

#function for sampling single argument given previous arguments, 
def sample_arg(arg_params, arg_net, rng_key, input_, candidates, hidden, tactic_size, embedding_dim):
    hidden, arg_scores = arg_net(arg_params, rng_key, input_, candidates, hidden, tactic_size, embedding_dim)
    arg_scores = jnp.ravel(arg_scores)
    arg_probs = jax.nn.log_softmax(arg_scores)
    arg_idx = jax.random.categorical(rng_key, arg_probs)
    log_prob = arg_probs[arg_idx]#jnp.log(arg_probs[arg_idx])
    return log_prob, (arg_idx, hidden)

def loss(context_params, tactic_params, term_params, arg_params, context_net, tactic_net, term_net, arg_net, jax_reps, context_set, fringe_sizes, rng_key, env, encoded_fact_pool, candidate_args):
    
    log_context, (fringe_idx, contexts_by_fringe) = sample_fringe(context_params, context_net, rng_key, jax_reps, context_set, fringe_sizes)
    
    try:
        target_context = contexts_by_fringe[fringe_idx][0]
    except:
        print ("error {} {}".format(contexts_by_fringe), fringe_idx)
    target_goal = target_context["polished"]["goal"]
    target_representation = jax_reps[context_set.index(target_context)]
    
    log_tac, tac_idx = sample_tactic(tactic_params, tactic_net, rng_key, jnp.expand_dims(target_representation,0), len(tactic_pool))
    
    sampled_tac = tactic_pool[tac_idx]
    arg_logs = []

    tactic = sampled_tac

    #if tactic requires no argument
    if sampled_tac in no_arg_tactic:
        full_tactic = sampled_tac #tactic_pool[tac]


    #Induct_on case; use term policy to find which term to induct on 
    elif sampled_tac in term_tactic:

        goal_tokens = target_goal.split()
        term_tokens = [[t] for t in set(goal_tokens) if t[0] == "V"]
        #add conditional if tokens is empty 

        #now want encodings for terms from AE

        term_reps = []

        for term in term_tokens:
            term_rep, _ = batch_encoder_.encode([term])
            #output is bidirectional so concat vectors
            term_reps.append(torch.cat(term_rep.split(1), dim=2).squeeze(0))
        
        #no terms in expression, only contains literals (e.g. induct_on `0`)
        if len(term_reps) == 0:
            print ("No variables to induct on for goal {}".format(target_goal))
            #return negative loss for now (positive overall as negative of log prob is positive)
            return 1., -1.
            
            
        # convert to jax
        term_reps = jnp.stack([jnp.array(term_reps[i][0].cpu()) for i in range(len(term_reps))])

        # now want inputs to term_net to be target_representation (i.e. goal) concatenated with terms
        # models the policies conditional dependence of the term given the goal

        #stack goal representation for each token
        goal_stack = jnp.concatenate([jnp.expand_dims(target_representation,0) for _ in term_tokens])

        #concat with term encodings to give candidate matrix
        candidates = jnp.concatenate([goal_stack, term_reps], 1)

        log_term, term_idx = sample_term(term_params, term_net, rng_key, candidates)

        sampled_term = term_tokens[term_idx]

        tm = sampled_term[0][1:] # remove headers, e.g., "V" / "C" / ...
    
        arg_logs = [log_term]
        
        if tm:
            tactic = "Induct_on `{}`".format(tm)
        else:
            # only to raise an error
            tactic = "Induct_on"
        
    #argument tactic
    else:
        #stack goals to possible arguments to feed into FFN
        goal_stack = jnp.concatenate([jnp.expand_dims(target_representation,0) for _ in encoded_fact_pool])
        candidates = jnp.concatenate([encoded_fact_pool, goal_stack], 1)
        
        #initial state set as goal
        hidden = jnp.expand_dims(target_representation,0)
        init_state = hk.LSTMState(hidden,hidden)
    
        # run through first with tac_idx to initialise state with tactic as c_0
        hidden, _ = arg_net(arg_params, rng_key, tac_idx, candidates, init_state, len(tactic_pool), 256)
        
        ARG_LEN = 5
        arg_inds = []
        arg_logs = []
        input_ = tac_idx
        for _ in range(ARG_LEN):
            log_arg, (arg_idx, hidden) = sample_arg(arg_params, arg_net, rng_key, input_, candidates, hidden, len(tactic_pool), 256)
            arg_logs.append(log_arg)
            arg_inds.append(arg_idx)
            input_ = jnp.expand_dims(encoded_fact_pool[arg_idx], 0)
        
        arg = [candidate_args[i] for i in arg_inds]

        tactic = env.assemble_tactic(sampled_tac, arg)
        
    
    
    action = (int(fringe_idx), 0, tactic)
    print ("Action {}:\n".format(action))
    
    try:
        reward, done = env.step(action)

    except:
        print("Step exception raised.")
        # print("Fringe: {}".format(env.history))
        print("Handling: {}".format(env.handling))
        print("Using: {}".format(env.using))
        # try again
        # counter = env.counter
        frequency = env.frequency
        env.close()
        print("Aborting current game ...")
        print("Restarting environment ...")
        print(env.goal)
        env = HolEnv(env.goal)
        flag = False
        return 
        
    print ("Result: Reward {}".format(reward))#, env.history[-1]))

    
    #negative as we want gradient ascent 
    
    loss = (-log_tac - log_context  - sum(arg_logs)) * reward

    return loss, reward


# In[34]:


import pickle

path_dir = "/home/sean/Documents/PhD/tactic_zero_jax/env/model_params"

def save(params, path):
    with open(path, 'wb') as fp:
        pickle.dump(params, fp)

def load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


# In[54]:


def train(goals):

    rng_key = jax.random.PRNGKey(11)

    init_context, apply_context = hk.transform(policy_networks._context_forward)
    apply_context = jax.jit(apply_context)

    init_tac, apply_tac = hk.transform(policy_networks._tac_forward)
    apply_tac = partial(jax.jit, static_argnums=3)(apply_tac)

    init_term, apply_term = hk.transform(policy_networks._term_no_tac_forward)
    apply_term = jax.jit(apply_term)

    init_arg, apply_arg = hk.transform(policy_networks._arg_forward)
    apply_arg = partial(jax.jit, static_argnums=(5,6))(apply_arg)

    #initialise these with e.g. random uniform, glorot, He etc. should exist outside function for action selection 
    context_params = init_context(rng_key, jax.random.normal(rng_key, (1,256)))

    tactic_params = init_tac(rng_key, jax.random.normal(rng_key, (1,256)), len(tactic_pool))

    #term_policy for now is only considering variables for induction, hence does not need any arguments 
    term_params = init_term(rng_key, jax.random.normal(rng_key, (1,512)))

    hidden = jax.random.normal(rng_key, (1,256))

    init_state = hk.LSTMState(hidden, hidden)

    arg_params = init_arg(rng_key, jax.random.randint(rng_key, (), 0, len(tactic_pool)), jax.random.normal(rng_key, (1,512)), init_state, len(tactic_pool), 256)

        
    context_lr = 1e-4
    tactic_lr = 1e-4
    arg_lr = 1e-4
    term_lr = 1e-4

    context_optimiser = optax.rmsprop(context_lr)
    tactic_optimiser = optax.rmsprop(tactic_lr)
    arg_optimiser = optax.rmsprop(arg_lr)
    term_optimiser = optax.rmsprop(term_lr)

    opt_state_context = context_optimiser.init(context_params)
    opt_state_tactic = tactic_optimiser.init(tactic_params)
    opt_state_arg = arg_optimiser.init(arg_params)
    opt_state_term = term_optimiser.init(term_params)
    
    
    for goal in goals[15:]:
        g = goal[1]
            
        env = HolEnv(g)

        theories = re.findall(r'C\$(\w+)\$ ', goal[0])
        theories = set(theories)
        theories = list(theories)

        allowed_theories = theories

        goal_theory = g

        #print ("Target goal: {}".format(g))
        
        try:
            allowed_arguments_ids = []
            candidate_args = []
            goal_theory = g#database[polished_goal][0] # plain_database[goal][0]
            for i,t in enumerate(database):
                if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)

            env.toggle_simpset("diminish", goal_theory)
            #print("Removed simpset of {}".format(goal_theory))

        except:
            allowed_arguments_ids = []
            candidate_args = []
            for i,t in enumerate(database):
                if database[t][0] in allowed_theories:
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)
            #print("Theorem not found in database.")

        print ("Number of candidate facts to use: {}".format(len(candidate_args)))

        encoded_database = torch.load('encoded_include_probability.pt')

        encoded_fact_pool = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids))
        
        encoded_fact_pool = jnp.array(encoded_fact_pool)
        
        for i in range(50):
            
            _, rng_key = jax.random.split(rng_key)

            
            print ("Proof step {} of 50\n".format(i+1))
            #print (env.history)
            #skip if encoding error
            try:
                jax_reps, context_set, fringe_sizes = gather_encoded_content_(env.history, batch_encoder_)
            except:
                continue

            #convert to jax
            jax_reps = jnp.stack([jnp.array(jax_reps[i][0].cpu()) for i in range(len(jax_reps))])

           
            gradients, reward = jax.grad(loss, argnums=(0,1,2,3), has_aux=True)(context_params, tactic_params, term_params, arg_params, apply_context, apply_tac, apply_term, apply_arg, jax_reps, context_set, fringe_sizes, rng_key, env, encoded_fact_pool, candidate_args)
            
#             #for when it returns same action repeatedly
#             if reward == -1:
#                 print (tactic_params)
#                 print (opt_state_tactic)
#                 print (tactic_updates)
            
            #update parameters
            context_updates, opt_state_context = context_optimiser.update(gradients[0], opt_state_context)
            context_params = optax.apply_updates(context_params, context_updates)
            
            tactic_updates, opt_state_tactic = tactic_optimiser.update(gradients[1], opt_state_tactic)
            tactic_params = optax.apply_updates(tactic_params, tactic_updates)
            
            term_updates, opt_state_term = term_optimiser.update(gradients[2], opt_state_term)
            term_params = optax.apply_updates(term_params, term_updates)
            
            arg_updates, opt_state_arg = arg_optimiser.update(gradients[3], opt_state_arg)
            arg_params = optax.apply_updates(arg_params, arg_updates)
            
            #if goal proven
            if reward == 5:
                print ("Goal {} proved in {} steps".format(g, i+1))
                break
        #save params after each proof attempt
        save(context_params, path_dir + "/context_params")
        save(opt_state_context, path_dir+"/context_state")
        save(tactic_params, path_dir+"/tactic_params")
        save(opt_state_tactic, path_dir+"/tactic_state")
        save(term_params, path_dir+"/term_params")
        save(opt_state_term, path_dir+"/term_state")
        save(arg_params, path_dir+"/arg_params")
        save(opt_state_arg, path_dir+"/arg_state")
            


# In[55]:


train(GOALS)


# In[ ]:




