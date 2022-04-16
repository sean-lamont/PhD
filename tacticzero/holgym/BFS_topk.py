import random
random.seed(0)
import pexpect
import re
import numpy as np
import torch
torch.manual_seed(0)
from torch.distributions import Categorical
from itertools import count
from sys import exit
from new_env import *
from seq2seq_sets_model import *
from exp_config import *
import json
import timeit
import json
import resource
from collections import deque

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

b_factor = 2
budget = 100
ARG_LEN = 5

context_net = ContextPolicy()

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(len(tactic_pool), 256)

term_net = TermPolicy(len(tactic_pool), 256)

context_net = context_net.to(device)
tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

saved = "trained_models/iteration-799.ckpt"
checkpoint = torch.load(saved, map_location=torch.device('cpu'))
context_net.load_state_dict(checkpoint["context_net_state_dict"])
tac_net.load_state_dict(checkpoint["tac_net_state_dict"])
arg_net.load_state_dict(checkpoint["arg_net_state_dict"])
term_net.load_state_dict(checkpoint["term_net_state_dict"])
print("Models loaded.")

context_net.eval()
tac_net.eval()
arg_net.eval()
term_net.eval()


# with open("trained_models/others_validation.json") as f:
with open("validation_data.json") as f:
    VALIDATION_SET = json.load(f)
print("Validation data loaded.")


iteration_counter = 0
encoded_database = torch.load('encoded_database.pt', map_location=device)
record_proofs = False

def expand_node(env, fringe, tactic):
    target_fringe = fringe
    pre_target = target_fringe["content"][0]
    target = pre_target["plain"]
    # tactic = normalize_args(tactic)
    if target["assumptions"]:
        # there are assumptions
        goal = revert_assumptions(pre_target)
        d = env.query(goal, "rpt strip_tac >> " + tactic)

    else:
        # no assumptions
        goal = target["goal"]
        d = env.query(goal, tactic)

    if d == "unexpected":
        reward = UNEXPECTED_REWARD

    elif d != "exception" and d != "timeout":
        # print("origin: {}".format(pre_target))
        # print("new: {}".format(d))
        # print([pre_target]==d)

        # progress has been made
        if [pre_target] != d:
            new_content = deepcopy(target_fringe["content"])
            new_content.remove(pre_target)
            new_content.extend(d)

            new_fringe = {"content": new_content,
                          "parent": -1,
                          "goal": -1,
                          "by_tactic": tactic,
                          "reward": None}
            
            if new_content == []:
                return [], True

            # new_fringe.update({"reward": reward})
            # new_fringe["reward"] = reward
            return new_fringe, False

        else:
            # nothing changed
            return fringe, False
    else:
        return fringe, False
    return fringe, False


def BFS_predict(env, goals, arg_len=ARG_LEN, mode="validation", frequency={}):
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
    # for g in goals:
    #     try:
    #         theory_g = plain_database[g][0]
    #     except:
    #         theory_g = "Unknown theory"
    #     if theory_g in iter_stats:
    #         iter_stats[theory_g][2] += 1
    #     else:
    #         iter_stats[theory_g] = [0,0,1]
    #     if g in replays:
    #         iter_stats[theory_g][1] += 1
    step_counter = 0
    queue = deque([])  
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

                env.toggle_simpset("diminish", goal_theory)
                print("Removed simpset of {}".format(goal_theory))
            except:            
                allowed_arguments_ids = []
                candidate_args = []
                for i,t in enumerate(database):
                    if database[t][0] in allowed_theories:
                        allowed_arguments_ids.append(i)
                        candidate_args.append(t)
                env.toggle_simpset("diminish", "pred_set")
                print("Removed simpset of pred_set.")
                print("Theorem not found in database.")
            # env.toggle_simpset("diminish", "pred_set")
            
        elif mode == "subgoals":
            pass
            # try:
            #     dependency_thy = subgoals_library[goal][0]
            #     dependency_num = subgoals_library[goal][1]
            #     usable_theorems = [t for t in allowed_arguments if
            #                        (int(allowed_arguments[t][2]) < int(dependency_num) and
            #                         allowed_arguments[t][0] == dependency_thy) or
            #                        allowed_arguments[t][0] != dependency_thy]
            # except:
            #     print("Theorem not found in database.")
            #     usable_theorems = [t for t in allowed_arguments]

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
        queue.append(state[0])
        for t in count():
            try:
                representations, context_set, fringe_sizes = gather_encoded_content(queue, batch_encoder)
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
            
            fringe = queue.pop()
            # print(fringe)
            target_context = fringe["content"][0] # contexts_by_fringe[fringe][0]
            target_goal = target_context["polished"]["goal"]
            target_representation = representations[context_set.index(target_context)]
            # print(target_representation.shape)
            # exit()

            # size: (1, max_contexts, max_assumptions+1, max_len)
            tac_input = target_representation.unsqueeze(0)
            tac_input = tac_input.to(device)

            tac_branches = []
            # for _ in range(b_factor):                
            #     tac_probs = tac_net(tac_input)
            #     # print(tac_probs)
            #     # tac = torch.argmax(tac_probs).unsqueeze(0)
            #     tac_m = Categorical(tac_probs)
            #     tac = tac_m.sample()
            #     # print(tac_probs)
            #     tac_branches.append(tac)

            tac_probs = tac_net(tac_input)
            tac_branches = tac_probs.topk(b_factor).indices
            # tac_branches = tac_branches.squeeze()
            tac_branches = tac_branches.view(tac_branches.shape[1],-1)
            # print(tac_branches)
            # print(tac_branches.squeeze())
            
            # break

            tactics = []
            for tb in tac_branches:
                tac_tensor = tb.to(device)
                if tactic_pool[tb] in no_arg_tactic:
                    tactic = tactic_pool[tb]
                elif tactic_pool[tb] == "Induct_on":
                    candidates = []
                    # input = torch.cat([target_representation, tac_tensor], dim=1)
                    tokens = target_goal.split()
                    tokens = list(dict.fromkeys(tokens))
                    tokens = [[t] for t in tokens if t[0] == "V"]
                    if tokens:
                        token_representations, _ = batch_encoder.encode(tokens)
                        # reshaping
                        encoded_tokens = torch.cat(token_representations.split(1), dim=2).squeeze(0)
                        target_representation_list = [target_representation.unsqueeze(0) for _ in tokens]

                        target_representations = torch.cat(target_representation_list)
                        # size: (len(tokens), 512)
                        candidates = torch.cat([encoded_tokens, target_representations], dim=1)
                        candidates = candidates.to(device)

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
                        induct_arg.append(tokens[term])                
                        tm = tokens[term][0][1:] # remove headers, e.g., "V" / "C" / ...
                        if tm:
                            tactic = "Induct_on `{}`".format(tm)
                        else:
                            print("tm is empty")
                            print(tokens)
                            # only to raise an error
                            tactic = "Induct_on"
                    else:
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
                    if tactic_pool[tb] in thm_tactic:
                        arg_len = 1
                    else:
                        arg_len = ARG_LEN

                    for i in range(arg_len):
                        hidden, scores = arg_net(input, candidates, hidden)
                        arg_probs = F.softmax(scores, dim=0)
                        arg_m = Categorical(arg_probs.squeeze(1))
                        arg = arg_m.sample()
                        
                        arg_step.append(arg)
                        input = encoded_fact_pool[arg].unsqueeze(0).unsqueeze(0)
                        # print(input.shape)

                        # renew candidates                
                        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                        hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                        hiddenl = torch.cat(hiddenl)

                        # size: (len(fact_pool), 512)
                        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                        candidates = candidates.to(device)

                    tac = tactic_pool[tb]
                    arg = [candidate_args[i] for i in arg_step]

                    tactic = env.assemble_tactic(tac, arg)
                    
                tactics.append(tactic)

            # new_nodes = []

            for tct in tactics:
                if step_counter >= budget:
                    return False, "Failed - reaching budget"
                new_fringe, finished = expand_node(env, fringe, tct)
                step_counter += 1
                if finished:
                    return True, "Succeed - proof found"
                # new_nodes.append(new_fringe)
                # print(new_fringe == [])
                if new_fringe != []:
                    queue.appendleft(new_fringe)
                
            # print(new_nodes)
            # print(queue)
            # print(len(queue))
            # print(step_counter)
            # if t == 3:
            #     break



# VALIDATION_SET = ["∀p y. EXISTS (λx. p) y ⇔ y ≠ [] ∧ p",
#                   "[] = REVERSE l ⇔ l = []",
#                   "∀x. x = [] ∨ ∃y t. y::t = x",
#                   "∀l2 l1 l3. l2 ++ (l1 ++ l3) = l2 ++ l1 ++ l3",
#                   "∀M M' v f. M = M' ⇒ (M' = [] ⇒ v = v') ⇒ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
#                   "l2 ++ l1 = [e] ⇔ l1 = [e] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [e]",
#                   "LAST (h::l) = if l = [] then h else LAST l",
#                   "LENGTH l = 0 ⇔ l = []",
#                   "¬SHORTLEX R x []",
#                   "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃h l. x = h::l ∧ f h l = v'",
#                   "∀p l3. EXISTS (λx. p) l3 ⇔  p ∧ l3 ≠ []",
#                   "[] = REVERSE l1 ⇔ l1 = []",
#                   "x = [] ∨ ∃y t. y::t = x",
#                   "l2 ++ (l1 ++ l3) = l2 ++ l1 ++ l3",
#                   "M = M' ⇒ (M' = [] ⇒ v = v') ⇒ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
#                   "l2 ++ l1 = [x] ⇔ l1 = [x] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [x]",
#                   "LAST (h::l2) = if l2 = [] then h else LAST l2",
#                   "LENGTH l = 0 ⇔ l = []",
#                   "¬SHORTLEX R y []",
#                   "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃h l1. x = h::l1 ∧ f h l1 = v'"]


# my_goals = ["l = MAP f (REVERSE []) ==> LENGTH l = 0"]
# my_goals = ["l = MAP f (REVERSE [x]) ==> LENGTH l = 1"]
# my_goals = ["x = [] ∨ ∃y t. y::t = x"]
# my_goals = ["∀x l. MEM x l ==> ¬ (l = [])"]
# my_goals = ["∀x l. MEM x l ==> l ≠ []"]
# # my_goals = ["a ≠ x /\ MEM a (x::l) ==> MEM a l"]



# # these are from mathlib of Lean
# my_goal0 = ["¬ (MEM a (x::l)) ==> ¬ (MEM a l)"]
# my_goal1 = ["MEM a l ==> MEM (f a) (MAP f l)"]
# my_goal2 = ["MAP f l = [] ==> l = []"]
# my_goal3 = ["MEM a l ==> 0 < LENGTH l"]
# my_goal4 = ["0 < LENGTH l ==> ∃a. MEM a l"]
# my_goal5 = ["MEM a (b::l) ==> MEM b l ==> MEM a l"]

# my_goals = ["¬ (MEM a (x::l)) ==> ¬ (MEM a l)"]

# tt = ["!l1 l2 n x. LENGTH l1 <= n ==> (LUPDATE x n (l1 ++ l2) = l1 ++ (LUPDATE x (n-LENGTH l1) l2))"]

# # env = HolEnv("NULL l ==> l = []")
# # env.step((0,0,"strip_tac"))
# # r = env.query("NULL l ==> l = []","strip_tac")
# # env.reset("T")
# env = HolEnv("T")
# _, _, _, h = predict(env, tt, "validation", {})


# # draw_tree(h[0])

tt = ["!(s:'a set) t. (t SUBSET s) /\ (s SUBSET t) ==> (s = t)"]
tt = ["!s t. (t SUBSET s) /\ (s SUBSET t) ==> (s = t)"]
tt = ["!(s:'a set) t u. s UNION (u UNION t) = (s UNION u) UNION t"]
tt = ["!(s:'a set) t u. t UNION (u UNION t) = (t UNION u) UNION t"]
tt = ["!s:'a set. !t. s SUBSET t <=> (t INTER s = s)"]
tt = ["!x y z. y DIFF x DIFF z = y DIFF z DIFF x"]
tt = ["!(s:'a set) u t. DISJOINT (s UNION t) u <=> DISJOINT s u /\ DISJOINT t u"]
tt = ["!s t u v. DISJOINT t s /\ u SUBSET s /\ v SUBSET t ==> DISJOINT u v"]
tt = ["FINITE (∅ :α -> bool)"]
tt = ["!x:'a set. ~(x PSUBSET x)"]
tt = ["!x:'a. !s t. (x INSERT s) SUBSET t <=> s SUBSET t /\ x IN t"]
# tt = ["!Q a s. (!x. x IN (a INSERT s) ==> Q x) <=> Q a /\ (!x. x IN s ==> Q x)"]
tt = ["∀(P :num -> bool) (n :num). EVERY P (COUNT_LIST n) ⇔ ∀(m :num). m < n ⇒ P m"]
tt = ["∀(n :num) (x :α). REPLICATE n x = GENLIST (K x :num -> α) n"]
tt = ["([] :α list) ∈ (s :α list -> bool) ⇒ longest_prefix s = ([] :α list)"]
tt = ["∀(n :num) (l :α list). n < LENGTH l ⇒ ELL n (REVERSE l) = ELL (PRE (LENGTH l − n)) l"]
tt = ["∀(n :num). LENGTH (l1 :α list) < n ⇒ TAKE n (l1 ++ (l2 :α list)) = l1 ++ TAKE (n − LENGTH l1) l2"]
tt = ["∀(l :α list). l ≠ ([] :α list) ⇒ EL (PRE (LENGTH l)) l = LAST l"]
env = HolEnv("T")
tt = ["!f s t. INJ f s t ==> !e. e IN s ==> INJ f (s DELETE e) (t DELETE (f e))"]
tt = ["!f:'a->'b. !s t. SURJ f s t = ((IMAGE f s) = t)"]
# BFS_predict(env, tt)

# draw_tree(env.history)
proved_theorems = []
failed_theorems = []
for i,t in enumerate(VALIDATION_SET):
    print("Proved:{}".format(len(proved_theorems)))
    print("Total:{}".format(i))
    for i in range(1):
        try:
            b, _ = BFS_predict(env, [t])
        except:
            failed_theorems.append(t)
            env.close()
            print("Restarting env ...")
            env = HolEnv("T")
            break
        if b:
            proved_theorems.append(t)
            break
    
print(len(proved_theorems))

# # draw_tree(env.history)
