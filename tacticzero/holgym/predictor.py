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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

timestep_list = []
proved_lengths = []
dep_dict = {}

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

def predict(env, goals, arg_len=ARG_LEN, mode="validation", frequency={}):
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
                exprs = []
                dep_names = []
                for i,t in enumerate(database):
                    if database[t][0] in allowed_theories and (database[t][0] != goal_theory or int(database[t][2]) < int(database[polished_goal][2])):
                        allowed_arguments_ids.append(i)
                        candidate_args.append(t)

                        dep_name = database[t][0] + "Theory." + database[t][1]
                        # expr = database[t][4]
                        dep_names.append(dep_name)
                         
                dep_dict[goal] = dep_names

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
                        print("tm is empty")
                        print(tokens)
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
                timestep_list.append(t+1)
                proof_length = extract_path_id(env.history)
                proved_lengths.append(len(proof_length))
                
                print("Rewards: {}".format(reward_print))
                print("Tactics: {}".format(action_pool))
                # print("Mean reward: {}".format(np.mean(reward_pool)))
                print("Total: {}".format(total_reward))
                print("Proof trace: {}".format(extract_proof(env.history)))
                print("Script: {}".format(reconstruct_proof(env.history)))
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
                        print("Replay buffer length: {}".format(len(current_replays)))
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
                    
                    proved_subgoals = provable_subgoals(env.subproofs, env.history)
                    proved_subgoals = [revert_plain(g) for g in proved_subgoals]
                    subgoals_library.extend(proved_subgoals)
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
                
                # if env.goal in replays:
                #     replay_flag = True
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

# tt = ["!(s:'a set) t. (t SUBSET s) /\ (s SUBSET t) ==> (s = t)"]

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

tt = ["∀(R :α -> β -> bool) (l1 :α list) (l2 :β list) (n :num). LIST_REL R l1 l2 ⇒ LIST_REL R (DROP n l1) (DROP n l2)"]
# b, _, _ = predict(env, tt)

# draw_tree(env.history)
proved_theorems = []
failed_theorems = []
for i, t in enumerate(VALIDATION_SET):
    print("Proved:{}".format(len(proved_theorems)))
    print("Total:{}".format(i))
    for _ in range(1):
        try:
            b, _, _ = predict(env, [t])
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
print("Mean timesteps:")
print(np.mean(timestep_list))
print("Mean lengths:")
print(np.mean(proved_lengths))

# dep_dict_json = json.dumps(dep_dict, ensure_ascii=False)
# with open("test_dep_dict.json", "w", encoding="utf8") as f:
#     f.write(dep_dict_json)

# proved_theorems_json = json.dumps(proved_theorems, ensure_ascii=False)
# with open("proved_theorems_tacticzero.json", "w", encoding="utf8") as f:
#     f.write(proved_theorems_json)
#     print("Proved theorems saved")


# draw_tree(env.history)
