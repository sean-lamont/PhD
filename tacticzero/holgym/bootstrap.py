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
import json
import resource
import os

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
num_iteration = 1000
validation_interval = 50
subgoals_training_interval = 1
checkpoint_interval = 50

ARG_LEN = 5
REPLAY_BUFFER_LEN = 3

CONTINUE = False #True
TRAIN_SET = GOALS #PROVABLES
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

# thms = {value[4] : value[1] for key, value in database.items()}

encoded_database = torch.load('encoded_database.pt', map_location=device)

# GAMES = TRAIN_GOALS
# GAMES = ["∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c"]
# GAMES = ["¬SHORTLEX R l []"]

# CONTINUE = True

learning_rate = 1e-5

context_rate = 1e-5
tac_rate = 1e-5
arg_rate = 1e-5
term_rate = 1e-5

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
    saving_dir = "/scratch1/wu099/temp/2020_12_19_19_38_38_checkpoint"
    checkpoint_fn = "replay-validation-iteration-1499.ckpt"
    checkpoint_fn = os.path.join(saving_dir, checkpoint_fn)
    checkpoint = torch.load(checkpoint_fn)
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
    
train_frequency = {}
validation_frequency = {}
train_rewards = []
validation_rewards = []
train_proved_per_iteration = []
validation_proved_per_iteration = []
statistics = {}

timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
print("Timestamp: {}".format(timestamp))        

if CONTINUE:
    statistics_fn = os.path.join(saving_dir, "statistics.json")
    frequencies_fn = os.path.join(saving_dir, "frequencies.json")
    replays_fn = os.path.join(saving_dir, "replays.json")
    proof_check_failure_fn = os.path.join(saving_dir, "failures.json")

    with open(statistics_fn) as f:
        statistics = json.load(f)

    print("Loading statistics ...")
    train_rewards = statistics["train_episodic_rewards"]
    validation_rewards = statistics["validation_episodic_rewards"]
    train_proved_per_iteration = statistics["train_proved_per_iteration"]
    validation_proved_per_iteration = statistics["validation_proved_per_iteration"]

    with open(frequencies_fn) as f:
        frequencies = json.load(f)

    print("Loading frequencies ...")
    train_frequency = frequencies["Training frequency"]
    validation_frequency = frequencies["Validation frequency"]

    print("Loading failures ...")
    with open(proof_check_failure_fn) as f:
        proof_check_failure = json.load(f)

    try:
        with open(replays_fn) as f:
            replays = json.load(f)
        print("Replays loaded.")
    except:
        replays = {}
        print("No initial replays.")

else:
    saving_dir = time.strftime('/scratch1/wu099/temp/%Y_%m_%d_%H_%M_%S_checkpoint', time.localtime())
    try:
        os.mkdir(saving_dir)
    except OSError:
        print("Creation of the directory %s failed" % saving_dir)

    statistics_fn = os.path.join(saving_dir, "statistics.json")
    frequencies_fn = os.path.join(saving_dir, "frequencies.json")
    replays_fn = os.path.join(saving_dir, "replays.json")
    proof_check_failure_fn = os.path.join(saving_dir, "failures.json")

    try:
        with open("pred_set_replays.json") as f:
            replays = json.load(f)
        print("Replays loaded.")
    except:
        replays = {}
        print("No initial replays.")

    for theorem in list(replays.keys()):
        validation_frequency[theorem] = 0
    for theorem in list(replays.keys()):
        train_frequency[theorem] = 0

    # replays = {key : [value[0]] for key, value in replays.items()}

def revert_plain(agpair):
    goal = agpair[1]
    for i in reversed(agpair[0]): 
        goal = "({}) ==> ({})".format(i, goal)
    return goal

def provable_subgoals(subs, history):
    provables = []
    
    while 1: # TODO: use for
        init_len = len(provables)
        for i in subs:
            plaintext = history[i[0]]["content"][i[1]]["plain"]
            assumptions = plaintext["assumptions"]
            goal = plaintext["goal"]
            entry = (assumptions, goal)
            for j in subs[i]:
                # if j["subgoals"] == [], then it's certainly true
                provable_i = True
                for k in j["subgoals"]:
                    if (k["plain"]["assumptions"], k["plain"]["goal"]) not in provables:
                        provable_i = False
                        break
                if provable_i and entry not in provables:
                    provables.append(entry)
        end_len = len(provables)
        if init_len == end_len:
            return provables


def replay_known_proof(known_history, arg_len=ARG_LEN):
    torch.set_grad_enabled(True)
    print("\nReplaying a known proof of {} ...".format(known_history[0]["content"][0]["plain"]["goal"]))
    start = timeit.default_timer()

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
    iteration_rewards = []

    # compute the encoded fact pool at the beginning of each episode
    polished_goal = known_history[0]["content"][0]["polished"]["goal"]
    allowed_theories = parse_theory(polished_goal)
    allowed_theories = [t for t in allowed_theories if t not in EXCLUDED_THEORIES]
    
    try:
        allowed_arguments_ids = []
        candidate_args = []
        goal_theory = database[polished_goal][0]
        for i,t in enumerate(database):
            if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
                allowed_arguments_ids.append(i)
                candidate_args.append(t)

    except:
        allowed_arguments_ids = []
        candidate_args = []
        for i,t in enumerate(database):
            if database[t][0] in allowed_theories:
                allowed_arguments_ids.append(i)
                candidate_args.append(t)
        print("Theorem not found in database.")

    encoded_fact_pool = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids, device=device))

    # print("Facts: {}+{}".format(num_init_facts, len(split_fact_pool)-num_init_facts))
    print("Facts: {}".format(len(allowed_arguments_ids)))

    for t in count():
        true_resulting_fringe = known_history[t+1]

        # gather all the goals in the history
        try:
            representations, context_set, fringe_sizes = gather_encoded_content(known_history[:t+1], batch_encoder)
        except:
            env.close()
            print("Skipping current game due to encoding error ...")
            print("Restarting environment ...")
            print(env.goal)                
            env = HolEnv(env.goal)
            flag = False
            break
            
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
        # fringe = fringe_m.sample()
        
        true_fringe = torch.tensor([true_resulting_fringe["parent"]])
        true_fringe = true_fringe.to(device)
        
        fringe_pool.append(fringe_m.log_prob(true_fringe))

        # take the first context in the chosen fringe for now

        target_context = contexts_by_fringe[true_fringe][0]
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

        true_tactic_text = true_resulting_fringe["by_tactic"]

        # true_tactic_text = "Induct_on `ll`"
        tac_args = re.findall(r'(.*?)\[(.*?)\]', true_tactic_text)
        tac_term = re.findall(r'(.*?) `(.*?)`', true_tactic_text)     
        tac_arg = re.findall(r'(.*?) (.*)', true_tactic_text)

        if tac_args:
            true_tac_text = tac_args[0][0]
            true_args_text = tac_args[0][1].split(", ")
        elif tac_term: # order matters # TODO: make it irrelavant
            true_tac_text = tac_term[0][0]
            true_args_text = tac_term[0][1]
        elif tac_arg: # order matters because tac_arg could match () ``
            true_tac_text = tac_arg[0][0]
            true_args_text = tac_arg[0][1]
        else:
            true_tac_text = true_tactic_text
        
        true_tac = torch.tensor([tactic_pool.index(true_tac_text)])
        true_tac = true_tac.to(device)
        # tac = tac_m.sample()
        # log directly the log probability
        tac_pool.append(tac_m.log_prob(true_tac))
        
        action_pool.append(tactic_pool[true_tac])
        
        tac_print.append(tac_probs.detach())
        # print(len(fact_pool[0].strip().split()))
        # exit()

        tac_tensor = true_tac.to(device)

        assert tactic_pool[true_tac.item()] == true_tac_text
        
        if tactic_pool[true_tac] in no_arg_tactic:
            tactic = tactic_pool[true_tac]
            arg_probs = []
            arg_probs.append(torch.tensor(0))
            arg_pool.append(arg_probs)
        elif tactic_pool[true_tac] == "Induct_on":
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
                    
                true_term = torch.tensor([tokens.index(["V" + true_args_text])])
                true_term = true_term.to(device)
                # term = term_m.sample()
                
                arg_probs.append(term_m.log_prob(true_term))
                induct_arg.append(tokens[true_term])
                tm = tokens[true_term.item()][0][1:] # remove headers, e.g., "V" / "C" / ...
                
                assert tm == true_args_text
                
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
            hidden0 = hidden1 = target_representation.unsqueeze(0).unsqueeze(0)

            hidden0 = hidden0.to(device)
            hidden1 = hidden1.to(device)

            hidden = (hidden0, hidden1)

            # concatenate the candidates with hidden states.

            hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
            hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

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
            # print("True args text: {}".format(true_args_text))
            if tactic_pool[true_tac] in thm_tactic:
                arg_len = 1
            else:
                arg_len = ARG_LEN

            for i in range(arg_len):
                hidden, scores = arg_net(input, candidates, hidden)
                arg_probs = F.softmax(scores, dim=0)
                arg_m = Categorical(arg_probs.squeeze(1))
                if isinstance(true_args_text, list):
                    try:
                        name_parser = true_args_text[i].split(".")
                    except:
                        print(i)
                        print(true_args_text)
                        print(known_history)
                        exit()
                    theory_name = name_parser[0][:-6] # get rid of the "Theory" substring
                    theorem_name = name_parser[1]
                    true_arg_exp = reverse_database[(theory_name, theorem_name)]
                else:
                    name_parser = true_args_text.split(".")
                    theory_name = name_parser[0][:-6] # get rid of the "Theory" substring
                    theorem_name = name_parser[1]
                    true_arg_exp = reverse_database[(theory_name, theorem_name)]    
                true_arg = torch.tensor(candidate_args.index(true_arg_exp))
                true_arg = true_arg.to(device)
                # arg = arg_m.sample()
                
                arg_step.append(true_arg)
                arg_step_probs.append(arg_m.log_prob(true_arg))

                # hiddenc0 = hidden[0].squeeze().repeat(1, 1, 1)
                # hiddenc1 = hidden[1].squeeze().repeat(1, 1, 1)

                # encoded chosen argument
                input = encoded_fact_pool[true_arg.item()].unsqueeze(0).unsqueeze(0)
                # print(input.shape)

                # renew candidates                
                hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                hiddenl = torch.cat(hiddenl)

                # size: (len(fact_pool), 512)
                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(device)

            arg_pool.append(arg_step_probs)

            # tac = tactic_pool[true_tac]
            # arg = [candidate_args[i] for i in arg_step]

            # tactic = env.assemble_tactic(tac, arg)
            # assert tactic == true_tactic_text

        action = (true_fringe.item(), 0, true_tactic_text)
        try:
            reward = true_resulting_fringe["reward"]
            done = t+2 == len(known_history)

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
            break

        # state_pool.append(state)
        reward_print.append(reward)
        # reward_pool.append(reward+trade_off*entropy)
        reward_pool.append(reward)

        # pg = ng

        steps += 1
        total_reward = float(np.sum(reward_print))
        if done == True:
            print("Proved in {} steps.".format(t+1))
            print("Rewards: {}".format(reward_print))
            # print("Tactics: {}".format(action_pool))
            # # print("Mean reward: {}".format(np.mean(reward_pool)))
            # print("Total: {}".format(total_reward))
            # print("Proof trace: {}".format(extract_proof(known_history)))
            # try:
            #     print("Proof script: {}".format(reconstruct_proof(known_history)))
            # except:
            #     print("Proof check failed with error.")
            #     proof_check_failure.append(known_history)

            # exit()
            proved += 1
            # traces.append(env.history)
            iteration_rewards.append(total_reward)
            break

    stop = timeit.default_timer()
    print('Time: {}  '.format(stop - start))


    # Update policy
    if True:
        # Discount reward
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add

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

        optimizer_context.step()
        optimizer_tac.step()
        optimizer_arg.step()
        # optimizer_term.step()

    prf = torch.mean(torch.stack(tac_print), 0)
    print("Preferences: {}\n".format(prf))

    # state_pool = []
    fringe_pool = []
    tac_pool = []
    arg_pool = []
    action_pool = []
    reward_pool = []
    reward_print = []
    steps = 0
    flag = True

    
def run_iteration(env, goals, arg_len=ARG_LEN, mode="training", frequency={}):
    torch.set_grad_enabled(mode=="training" or mode=="subgoals")
    global iteration_counter
    # state_pool = []
    fringe_pool = []
    tac_pool = []
    arg_pool = []
    reward_pool = []
    reward_print = []
    action_pool = []
    steps = 0
    flag = True
    replay_flag = False
    tac_print = []

    induct_arg = []
    proved = 0
    iteration_rewards = []

    # initialize iter stats
    iter_stats = {}
    for g in goals:
        try:
            theory_g = plain_database[g][0]
        except:
            theory_g = "Unknown theory"
        if theory_g in iter_stats:
            iter_stats[theory_g][2] += 1
        else:
            iter_stats[theory_g] = [0,0,1]
        if g in replays:
            iter_stats[theory_g][1] += 1
        

    # num_init_facts = len(fact_pool)
    for goal_index, goal in enumerate(goals):

        env.reset(goal, frequency)

        print("Game: {}".format(goal_index))

        start = timeit.default_timer()

        # compute the encoded fact pool at the beginning of each episode
        polished_goal = env.fringe["content"][0]["polished"]["goal"]
        allowed_theories = parse_theory(polished_goal)
        print("Theories: {}".format(allowed_theories))
        allowed_theories = [t for t in allowed_theories if t not in EXCLUDED_THEORIES]
        print("Allowed theories: {}".format(allowed_theories))
        if mode == "validation":
            try:
                allowed_arguments_ids = []
                candidate_args = []
                goal_theory = database[polished_goal][0] # plain_database[goal][0]
                for i,t in enumerate(database):
                    if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
                        allowed_arguments_ids.append(i)
                        candidate_args.append(t)
            except:
                allowed_arguments_ids = []
                candidate_args = []
                for i,t in enumerate(database):
                    if database[t][0] in allowed_theories:
                        allowed_arguments_ids.append(i)
                        candidate_args.append(t)
                print("Theorem not found in database.")
                goal_theory = "-"
            env.toggle_simpset("diminish", goal_theory)
            # allowed_arguments_ids = []
            # candidate_args = []
            # for i,t in enumerate(database):
            #     if database[t][0] in allowed_theories:
            #         allowed_arguments_ids.append(i)
            #         candidate_args.append(t)
            # try:
            #     goal_theory = database[polished_goal][0]
            # except:
            #     goal_theory = "-"
            #     print("Validation theorem not found in database. Using all simpset.")
            # env.toggle_simpset("diminish", goal_theory)

        elif mode == "subgoals":
            try:
                dependency_thy = subgoals_library[goal][0]
                dependency_num = subgoals_library[goal][1]
                usable_theorems = [t for t in allowed_arguments if
                                   (int(allowed_arguments[t][2]) < int(dependency_num) and
                                    allowed_arguments[t][0] == dependency_thy) or
                                   allowed_arguments[t][0] != dependency_thy]
            except:
                print("Theorem not found in database.")
                usable_theorems = [t for t in allowed_arguments]

        else:
            # env.toggle_simpset("diminish", goal_theory)
            if iteration_counter >= 0:
                # usable_theorems = [t for t in allowed_arguments if allowed_arguments[t][1] < allowed_arguments[polished_goal][1]]
                # either theorems in the same theory but have a smaller dependency number, or
                # any theorems from other specified theories
                try:
                    allowed_arguments_ids = []
                    candidate_args = []
                    goal_theory = database[polished_goal][0] # plain_database[goal][0]
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
                    
            else:
                # try to bootstrap by learning from self
                try:
                    allowed_arguments_ids = []
                    candidate_args = []
                    for i,t in enumerate(database):
                        if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) <= int(database[polished_goal][2])):
                            allowed_arguments_ids.append(i)
                            candidate_args.append(t)

                except:
                    allowed_arguments_ids = []
                    candidate_args = []
                    for i,t in enumerate(database):
                        if database[t][0] in allowed_theories:
                            allowed_arguments_ids.append(i)
                            candidate_args.append(t)
                    print("Theorem not found in database.")

        encoded_fact_pool = torch.index_select(encoded_database, 0, torch.tensor(allowed_arguments_ids, device=device))

        print("Facts: {}".format(len(allowed_arguments_ids)))

        state = env.history

        for t in count():

            # gather all the goals in the history
            try:
                representations, context_set, fringe_sizes = gather_encoded_content(env.history, batch_encoder)
            except:
                env.close()
                print("Skipping current game due to encoding error ...")
                print("Restarting environment ...")
                print(env.goal)                
                env = HolEnv(env.goal)
                flag = False
                break
                
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
            # if mode == "training":
            #     fringe = fringe_m.sample()
            # else:
            #     fringe = torch.argmax(fringe_probs)
            fringe = fringe_m.sample()
            fringe_pool.append(fringe_m.log_prob(fringe))

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
            # if mode == "training":
            #     tac = tac_m.sample()
            # else:
            #     tac = torch.argmax(tac_probs).unsqueeze(0) # to make shape like tensor([x]) in order to send it to arg_net (for initialization)
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
                    # if mode == "training":
                    #     term = term_m.sample()
                    # else:
                    #     term = torch.argmax(term_probs)
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
                hidden0 = hidden1 = target_representation.unsqueeze(0).unsqueeze(0)

                hidden0 = hidden0.to(device)
                hidden1 = hidden1.to(device)

                hidden = (hidden0, hidden1)

                # concatenate the candidates with hidden states.

                hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

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
                if tactic_pool[tac] in thm_tactic:
                    arg_len = 1
                else:
                    arg_len = ARG_LEN

                for i in range(arg_len):
                    hidden, scores = arg_net(input, candidates, hidden)
                    arg_probs = F.softmax(scores, dim=0)
                
                    arg_m = Categorical(arg_probs.squeeze(1))
                    # if mode == "training":
                    #     arg = arg_m.sample()
                    # else:
                    #     arg = torch.argmax(arg_probs)
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
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    hiddenl = torch.cat(hiddenl)

                    # size: (len(fact_pool), 512)
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(device)

                arg_pool.append(arg_step_probs)

                tac = tactic_pool[tac]
                arg = [candidate_args[i] for i in arg_step]

                tactic = env.assemble_tactic(tac, arg)

            action = (fringe.item(), 0, tactic)

            # reward, done = env.step(action)
            try:
                # when step is performed, env.history (probably) changes
                # if goal_index == 0:
                #     raise "boom"
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
                break

            # this could be removed
            if reward == UNEXPECTED_REWARD:
                # try again
                # counter = env.counter
                env.close()
                print("Skipping current game ...")
                print("Restarting environment ...")
                print(env.goal)                
                env = HolEnv(env.goal)
                flag = False
                break

            if t == 49:
                reward = -5
            # state_pool.append(state)
            reward_print.append(reward)
            # reward_pool.append(reward+trade_off*entropy)
            reward_pool.append(reward)

            # pg = ng

            steps += 1
            total_reward = float(np.sum(reward_print))
            if done == True:
                print("Proved in {} steps.".format(t+1))
                print("Rewards: {}".format(reward_print))
                print("Tactics: {}".format(action_pool))
                # print("Mean reward: {}".format(np.mean(reward_pool)))
                print("Total: {}".format(total_reward))
                print("Proof trace: {}".format(extract_proof(env.history)))
                try:                    
                    check_result = check_proof(env, env.history)
                    if not check_result:
                        print("Proof check failed without error.")
                        proof_check_failure.append(env.history)
                        env.close()
                        print("Skipping current game ...")
                        print("Restarting environment ...")
                        print(env.goal)
                        env = HolEnv('T')
                    else:
                        print("Proof check passed.")
                except:
                    print("Proof check failed with error.")
                    proof_check_failure.append(env.history)
                    # exit()
                proved += 1
                # traces.append(env.history)
                iteration_rewards.append(total_reward)
                
                if mode == "training":
                    # name = thms[env.goal]
                    # name = plain_database[env.goal]
                    # p = env.get_polish(env.goal)
                    # entry = {p[0]["polished"]["goal"]: name}
                    # new_facts.update(entry)
                    # new_facts[p[0]["polished"]["goal"]] = name
                    
                    # try:
                    #     new_facts[env.goal] = plain_database[env.goal]
                    # except:
                    #     print("Theorem not found in database.")

                    replay = env.history
                    replay_score = total_reward

                    if env.goal not in replays:
                        # one more provable theorem discovered
                        iter_stats[goal_theory][1] += 1
                    iter_stats[goal_theory][0] += 1

                    # if env.goal in replays:
                    #     if replay_score >= replays[env.goal][1]:
                    #         replays.update({env.goal: (replay, replay_score)})
                    # else:
                    #     replays.update({env.goal: (replay, replay_score)})

                    # always use the latest replay
                    # replays.update({env.goal: (replay, replay_score)})
                    if env.goal in replays:
                        current_replays = replays[env.goal]
                        if len(current_replays) < REPLAY_BUFFER_LEN:
                            current_replays.append((replay, replay_score))
                        else:
                            current_replays = current_replays[1:]
                            current_replays.append((replay, replay_score))
                        replays[env.goal] = current_replays
                    else:
                        replays[env.goal] = [(replay, replay_score)]
                    
                    # try:
                    #     dependency_thy = allowed_arguments[polished_goal][0]
                    #     dependency_num = allowed_arguments[polished_goal][2]
                    # except:
                    #     print("Theorem not found in database.")
                    #     dependency_thy = "Not found"
                    #     dependency_num = "Not found"

                    # proved_subgoals = provable_subgoals(env.subproofs, env.history)
                    # proved_subgoals = [(revert_plain(g), (dependency_thy, dependency_num)) for g in proved_subgoals]
                    # # subgoals_library.extend(proved_subgoals)
                    # subgoals_library.update(dict(proved_subgoals))
                break

            if t == 49:
                print("Failed.")
                print("Rewards: {}".format(reward_print))
                # print("Rewards: {}".format(reward_pool))
                print("Tactics: {}".format(action_pool))
                # print("Mean reward: {}\n".format(np.mean(reward_pool)))
                print("Total: {}".format(total_reward))
                iteration_rewards.append(total_reward)
                
                if env.goal in replays:
                    replay_flag = True
                break
            
        if record_proofs:
            traces.append(env.history)
        # if record_subproofs:
        #     subproofs_traces.append(env.subproofs)
            
        stop = timeit.default_timer()
        print('Time: {}  '.format(stop - start))

        if replay_flag and (mode == "training"):
            # target_replay = random.sample(replays[env.goal],1)[0]
            target_replay = random.sample(replays[env.goal],1)[0][0]
            # target_replay = replays[env.goal][0]
            replay_known_proof(target_replay)
            flag = False

        # Update policy
        if flag and (mode == "training" or mode == "subgoals"):
            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

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

            optimizer_context.step()
            optimizer_tac.step()
            optimizer_arg.step()
            optimizer_term.step()

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
        replay_flag = False

        # print("Loss: {}\n".format(loss))
        print("Agent induct args: {}".format(induct_arg))
        induct_arg = []
        prf = torch.mean(torch.stack(tac_print), 0)
        print("Agent preferences: {}".format(prf))
        tac_print = []
        print("Proved so far: {}\n".format(proved))

    if mode == "training":
        iteration_counter += 1
        print("Iteration statistics: {}".format(iter_stats))
        
    return proved, iteration_rewards, env


if __name__ == "__main__":
    # global_start = timeit.default_timer()
    # target_goals = list(replays.keys())[291:]
    target_goals = [t for t in list(replays.keys()) if t != "T"]
    # target_goals = [t for t in list(replays.keys())[:100] if t != "T"]
    # target_goals1 = [t for t in list(replays.keys())[:50] if t != "T"]
    # target_goals2 = [t for t in list(replays.keys())[341:] if t != "T"]
    # target_goals = target_goals1 + target_goals2
    print("Total games: {}".format(len(target_goals)))    
    env = HolEnv("T")
    
    print("Validating ...")
    validation_proved, validation_iteration_rewards, env = run_iteration(env, target_goals, ARG_LEN, "validation", validation_frequency)
    validation_frequency = env.frequency
    validation_rewards.append(validation_iteration_rewards)
    validation_proved_per_iteration.append(validation_proved)
    print("End validation.\n")

    if CONTINUE:
        starting_iteration = 1500 #len(statistics["validation_proved_per_iteration"])
    else:
        starting_iteration = 0

    for i in range(num_iteration):
        fn = "replay-validation-iteration-{}.ckpt".format(i+starting_iteration)
        fn = os.path.join(saving_dir, fn)
        
        # the returning env ensures that we have the restarted env in cases there was an error
        for t in target_goals:
            if t != "T":
                target_replay = random.sample(replays[t],1)[0][0]
                replay_known_proof(target_replay)

         
        if (i+1) % validation_interval == 0:
            print("Validating ...")
            validation_proved, validation_iteration_rewards, env = run_iteration(env, target_goals, ARG_LEN, "validation", validation_frequency)
            validation_frequency = env.frequency
            validation_rewards.append(validation_iteration_rewards)
            validation_proved_per_iteration.append(validation_proved)
            print("End validation.\n")
            # print("Training ...")
            # train_proved, train_iteration_rewards, env = run_iteration(env, target_goals, ARG_LEN, "training", train_frequency)
            # train_frequency = env.frequency
            # train_rewards.append(train_iteration_rewards)
            # train_proved_per_iteration.append(train_proved)
            # print("End training.\n")
            
        if (i+1) % checkpoint_interval == 0:
            torch.save({
                'context_net_state_dict': context_net.state_dict(),
                'tac_net_state_dict': tac_net.state_dict(),
                'arg_net_state_dict': arg_net.state_dict(),
                'term_net_state_dict': term_net.state_dict(),
                'optimizer_context_state_dict': optimizer_context.state_dict(),
                'optimizer_tac_state_dict': optimizer_tac.state_dict(),
                'optimizer_arg_state_dict': optimizer_arg.state_dict(),
                'optimizer_term_state_dict': optimizer_term.state_dict(),
            }, fn)

            print("Models saved.")

            statistics["train_episodic_rewards"] = train_rewards # sum(train_rewards,[])
            statistics["validation_episodic_rewards"] = validation_rewards # sum(validation_rewards,[])
            statistics["train_proved_per_iteration"] = train_proved_per_iteration
            statistics["validation_proved_per_iteration"] = validation_proved_per_iteration
            statistics["provables"] = len(target_goals)

            stats_json = json.dumps(statistics)
            with open(statistics_fn,"w") as f:
                f.write(stats_json)

            print("Statistics saved.")

            frequencies = {"Training frequency": train_frequency,
                           "Validation frequency": validation_frequency}
            frequencies_json = json.dumps(frequencies)
            with open(frequencies_fn,"w") as f:
                f.write(frequencies_json)

            replays_json = json.dumps(replays)
            with open(replays_fn,"w") as f:
                f.write(replays_json)

            print("Replays saved.")

            proof_check_failure_dict = json.dumps(proof_check_failure)
            with open(proof_check_failure_fn,"w") as f:
                f.write(proof_check_failure_dict)

            print("Failures saved.")
    
            if record_proofs:
                traces_json = json.dumps(traces)
                with open("traces.json","w") as f:
                    f.write(traces_json)
    env.close()
