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
from time import sleep
import json
import signal

HOLPATH = "/home/minchao/HOL/bin/hol"
dictionary = {}
MAX_LEN = 128
MAX_ASSUMPTIONS = 3
PRINT_EXCEPTION = False

# DELSIMPS = ["HD", "TL_DEF", "APPEND", "FLAT", "LENGTH", "MAP", "LIST_TO_SET_DEF", "FILTER", "EVERY_DEF", "EXISTS_DEF", "MAP2_DEF", "APPEND_NIL", "LENGTH_APPEND", "MAP_ID", "FLAT_APPEND", "EL_restricted", "EL_simp_restricted", "LIST_REL_def", "LIST_REL_NIL", "REVERSE_DEF", "REVERSE_REVERSE", "REVERSE_11", "MEM_REVERSE", "LENGTH_REVERSE", "REVERSE_EQ_NIL", "REVERSE_EQ_SING", "LAST_CONS", "FRONT_CONS", "LENGTH_FRONT_CONS", "FRONT_CONS_EQ_NIL", "LAST_APPEND_CONS", "TAKE_nil", "TAKE_cons", "DROP_nil", "DROP_cons", "TAKE_0", "TAKE_LENGTH_ID", "LENGTH_TAKE", "DROP_0", "TAKE_DROP", "LENGTH_DROP", "ALL_DISTINCT", "LIST_TO_SET_APPEND", "LIST_TO_SET_EQ_EMPTY", "FINITE_LIST_TO_SET", "LIST_TO_SET_REVERSE", "SET_TO_LIST_EMPTY", "SET_TO_LIST_SING", "ALL_DISTINCT_SET_TO_LIST", "isPREFIX", "isPREFIX_THM", "SNOC", "LENGTH_SNOC", "LAST_SNOC", "FRONT_SNOC", "SNOC_11", "LENGTH_GENLIST", "GENLIST_AUX_compute", "EL_GENLIST", "GENLIST_NUMERALS", "INFINITE_LIST_UNIV", "LENGTH_LUPDATE", "LIST_BIND_THM", "SINGL_LIST_APPLY_L", "SINGL_SINGL_APPLY", "dropWhile_def", "APPEND_11"]

DELSIMPS = ["HD", "TL_DEF", "APPEND", "FLAT", "LENGTH", "MAP", "LIST_TO_SET_DEF", "FILTER", "EVERY_DEF", "EXISTS_DEF", "MAP2_DEF", "APPEND_NIL", "LENGTH_APPEND", "MAP_ID", "FLAT_APPEND", "EL_restricted", "EL_simp_restricted", "LIST_REL_def", "LIST_REL_NIL", "REVERSE_DEF", "REVERSE_REVERSE", "REVERSE_11", "MEM_REVERSE", "LENGTH_REVERSE", "REVERSE_EQ_NIL", "REVERSE_EQ_SING", "LAST_CONS", "FRONT_CONS", "LENGTH_FRONT_CONS", "FRONT_CONS_EQ_NIL", "LAST_APPEND_CONS", "TAKE_nil", "TAKE_cons", "DROP_nil", "DROP_cons", "TAKE_0", "TAKE_LENGTH_ID", "LENGTH_TAKE", "DROP_0", "TAKE_DROP", "LENGTH_DROP", "ALL_DISTINCT", "LIST_TO_SET_APPEND", "LIST_TO_SET_EQ_EMPTY", "FINITE_LIST_TO_SET", "LIST_TO_SET_REVERSE", "SET_TO_LIST_EMPTY", "SET_TO_LIST_SING", "ALL_DISTINCT_SET_TO_LIST", "isPREFIX", "isPREFIX_THM", "SNOC", "LENGTH_SNOC", "LAST_SNOC", "FRONT_SNOC", "SNOC_11", "LENGTH_GENLIST", "GENLIST_AUX_compute", "EL_GENLIST", "GENLIST_NUMERALS", "INFINITE_LIST_UNIV", "LENGTH_LUPDATE", "LIST_BIND_THM", "SINGL_LIST_APPLY_L", "SINGL_SINGL_APPLY", "dropWhile_def", "APPEND_11", "MAP2_NIL", "LENGTH_MAP2", "MAP_EQ_NIL", "MAP_EQ_SING", "LENGTH_NIL", "LENGTH_NIL_SYM", "ALL_DISTINCT_REVERSE", "ALL_DISTINCT_FLAT_REVERSE", "isPREFIX_NILR", "LUPDATE_NIL", "SHORTLEX_THM", "SHORTLEX_NIL2", "WF_SHORTLEX", "LLEX_THM", "LLEX_NIL2", "nub_set", "EVERY2_THM", "oHD_thm", "LIST_TO_SET", "LENGTH_MAP", "MEM_APPEND", "SING_HD", "APPEND_eq_NIL", "NULL_APPEND", "FILTER_F", "FILTER_T", "MEM", "LENGTH_ZIP_MIN", "LAST_MAP", "TAKE_EQ_NIL", "MEM_SET_TO_LIST", "MEM_SNOC", "LIST_REL_eq"]

dels = re.sub("\'","\"",str(DELSIMPS))

with open("polished_def_dict.json") as f:
    defs = json.load(f)
    # print(list(dictionary.keys()))

fact_pool = list(defs.keys())
new_facts = {}

# "irule", "drule", 
tactic_pool = ["simp", "fs", "strip_tac", "metis_tac", "Induct_on"]

with open("thm_dict.json") as f:
    thms = json.load(f)
    # print(list(dictionary.keys()))

GOALS = list(thms.keys())

COUNTER = 0

class HolEnv():
    def __init__(self, goals):
        # Each state is a max_len * (max_assumptions + 1) * max_goals matrix
        # In a simpler case where we don't consider other goals,
        # each state is a max_len * (max_assumptions + 1) matrix

        # maximal length of a goal
        self.max_len = MAX_LEN

        # maximal assumptions to consider
        self.max_assumptions = MAX_ASSUMPTIONS
        # maximal goals to consider
        self.max_goals = 4
        
        self.goals = goals
        
        self.theories = ["listTheory", "bossLib"]
        self.process = pexpect.spawn(HOLPATH)

        # import theories
        print("Importing theories...")
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        for i in self.theories:
            self.process.sendline("open {};".format(i).encode("utf-8"))

        # remove built-in simp lemmas
        print("Removing simp lemmas...")
        # self.process.sendline("delsimps [\"HD\", \"EL_restricted\", \"EL_simp_restricted\"];")
        self.process.sendline("delsimps {};".format(dels))
        sleep(1)

        # load utils
        print("Loading modules...")
        self.process.sendline("use \"helper.sml\";")
        # self.process.sendline("val _ = load \"Timeout\";")
        sleep(3)
        print("Configuration done.")
        self.process.expect('\r\n>')
        # self.process.readline()
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
            
        # consumes hol4 head
        self.process.expect('\r\n>')
        
        # setup the goal
        self.goal = sample(self.goals, 1)[0]
        
        # self.history = [([], self.goal)]
        # self.history = [(["Induct_on `m`","conv_tac"], self.goal)]
        
        # a pair of polished goal list and original goal list
        self.fringe = self.get_polish(self.goal)
        
        self.scripts = []
        # goal = self.construct_goal(self.goal)
        # self.process.sendline(goal.encode("utf-8"))        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))

    def get_names(self, exps):
        # look up the names of exps
        names = []
        for e in exps:
            try:
                i = defs[e]
            except:
                i = new_facts[e]
            names.append(i)
        return names
        
    def assemble_tactic(self, tac, args):
        # args is a list of strings
        if tac == "simp" or tac == "fs" or tac == "metis_tac":
            names = self.get_names(args)
            action = tac + re.sub("'", "", str(names))
        elif tac == "irule" or tac == "drule":
            names = self.get_names(args)
            action = tac + " " + names[0]
        else:
            action = tac
        return action
    
    def construct_goal(self, goal):
        s = "g " + "`" + goal + "`;"
        return s

    def construct_tactic(self, tac):
        s = "e " + "(" + tac + ");"
        return s

    def reset(self):
        # if self.process:
        #     self.process.close(force=True)
        # self.__init__(self.goals)
        
        self.goal = sample(self.goals, 1)[0]

        # self.goal = self.goals[COUNTER]
        
        # self.history = [([], self.goal)]
        self.fringe = self.get_polish(self.goal)
        
        self.scripts = []
        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))

    # def get_states(self):
    #     # only goals are considered for now

    #     # create an empty entry
    #     unit = self.max_len * [0]
    #     states = list(map(lambda x : self.encode(x[1]), self.history))
    #     # pad with empty entries
    #     states = (states + self.max_goals * [unit])[:self.max_goals]
    #     # return torch.tensor(states, dtype=torch.float)
    #     return torch.FloatTensor(states)

    def get_states(self):
        # this always assumes that history is not empty
        # get the first goal-assumptions pair in history
        
        # create an empty entry
        unit = self.max_len * [0]
        
        # first 0: take the first goal-assumption pair
        # second 0: take the polished form
        # [(polished form,(assumptions, goal)), ...]
        target = self.fringe[0][0]
        state = [self.encode(target[1])]
        for i in range(self.max_assumptions):
            if i < len(target[0]):
                state.append(self.encode(target[0][i]))
            else:
                state.append(unit)
        # returns the polished goal and the encoding of the goal-assumptions pair
        return (target[1], torch.FloatTensor(state))

    def get_polish(self, raw_goal):
        goal = self.construct_goal(raw_goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("top_goals();".encode("utf-8"))
        self.process.expect("val it =")
        self.process.expect([": goal list", ":\r\n +goal list"])
        polished_raw = self.process.before.decode("utf-8")
        polished_subgoals = re.sub("“|”","\"", polished_raw)
        polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

        # print("content:{}".format(subgoals))
        # exit()
        pd = eval(polished_subgoals)
        
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        return list(zip(pd, [([], raw_goal)]))
    
    def query(self, raw_goal, tac):
        # print("content1:{}".format(self.process.before.decode("utf-8")))
        # print("goal is: {}".format(goal))
        # print("tac is: {}".format(tac))
        
        goal = self.construct_goal(raw_goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")
        
        # bug1 = self.process.before.decode("utf-8")
        # print("bug1: {}".format(bug1))
        
        tactic = self.construct_tactic(tac)
        self.process.sendline(tactic.encode("utf-8"))
        
        # bug2 = self.process.before.decode("utf-8")
        # print("bug2: {}".format(bug2))

        # Note we may see "metis: proof translation error: trying again with types."]
        try: 
            i = self.process.expect(["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof" , "Exception", "error"])
        except:
            print("Exception: {} to {} to be debugged".format(tac, raw_goal))
            exit()
            
        # print("i is {}".format(i))
        
        # bug3 = self.process.before.decode("utf-8")
        # print("bug3: {}".format(bug3))
        # exit()

        # workaround
        if i == 0:
            i = self.process.expect(["metis: proof translation error", "Initial goal proved", ": proof" , "Exception", "error"])
            # print("i is {}".format(i))
        
        if i == 2 or i == 3:
            # bug4 = self.process.before.decode("utf-8")
            # print("bug4: {}".format(bug4))

            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            # bug4 = self.process.before.decode("utf-8")
            # print("bug4: {}".format(bug4))

            try:
                self.process.expect("val it =")
            except:
                print("Exception: {} to {} returned no goals".format(tac, raw_goal))
                exit()
            
            # this (:\r\n) doesn't seem robust
            self.process.expect([": goal list", ":\r\n +goal list"])
            raw = self.process.before.decode("utf-8")
            
            # print("sub: {}".format(raw))
            subgoals = re.sub("“|”","\"", raw)
            subgoals = re.sub("\r\n +"," ", subgoals)

            # get Polished version
            self.process.expect("\r\n>")
            self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            self.process.expect("val it =")
            self.process.expect([": goal list", ":\r\n +goal list"])
            polished_raw = self.process.before.decode("utf-8")         
            # print("sub: {}".format(raw))
            polished_subgoals = re.sub("“|”","\"", polished_raw)
            polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

            # print("content:{}".format(subgoals))
            # exit()
            pd = eval(polished_subgoals)
            d = eval(subgoals)
            data = list(zip(pd, d))
            # data = (pd, d)
            # data = eval(subgoals)
        elif i == 1:
            data = []
        elif i == 4:
            j = self.process.expect(["Time", pexpect.TIMEOUT], timeout=0.01)
            if j == 0:
                data = "timeout"
            else:
                # print("pexpect timeout")
                data = "exception"
        else:
            if PRINT_EXCEPTION:
                print("Exception: {} to {}.".format(tac, raw_goal))
            data = "exception"
        
        # clear stack and consume the remaining stdout
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        return data

    def revert(self, assumptions, goal):
        # order shouldn't matter
        for i in assumptions: 
            goal = "(" + i + ")" + " ==> " + "(" + goal + ")"
        return goal

    # def encode(self, s):
    #     r = []
    #     for c in s:
    #         if c not in dictionary:
    #             dictionary[c] = len(dictionary) + 1
    #         r.append(dictionary[c])
    #     # pad r with 0's
    #     r = (r + self.max_len * [0])[:self.max_len]
    #     return r

    def encode(self, s):
        # s is a string
        # print("Encoding: {}".format(s))
        s = s.split()
        r = []
        for c in s:
            if c not in dictionary:
                dictionary[c] = len(dictionary) + 1
            r.append(dictionary[c])
        # pad r with 0's
        r = (r + self.max_len * [0])[:self.max_len]

        # a list whose length is max_len
        return r

    
    def random_play_no_back(self):
        self.reset()
        steps = 0
        while self.fringe:
            pre_target = sample(self.fringe, 1)[0]
            
            # take the natural representation
            target = pre_target[1]
            
            tac = sample(tactic_pool, 1)[0]
            
            if tac == "Induct_on":
                g = pre_target[0][1]
                tokens = g.split()
                term = sample(tokens, 1)[0]
                term = term[1:]
                if term:
                    args = [term]
                    action = "Induct_on `{}`".format(term)
                else:
                    action = "Induct_on"
            else:
                args = []
                for i in range(5):
                    arg = sample(fact_pool, 1)
                    args.extend(arg)
                action = self.assemble_tactic(tac, args)

            print("Target: {}\nTactic: {}".format(target, action))
            if target[0]:
                # if there are assumptions
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
            else:
                goal = target[1]
                d = self.query(goal, action)
                
            # if d == "metis_timeout":
            #     print("Failed due to metis timeout.")
            #     return steps
            
            if d != "exception" and d != "timeout":
                self.fringe.remove(pre_target)
                self.fringe.extend(d)

            steps += 1
            if steps == 50:
                print("Failed")
                return steps
        print("Proof found in {} steps.".format(steps))
        return steps

    def step(self, action):
        # action = self.action_pool[action]
        if self.fringe:
            pre_target = self.fringe[0]
            target = pre_target[1]
            if target[0]:
                # there are assumptions
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
                script = ("rpt strip_tac >> " + action, goal)
            else:
                # no assumptions
                goal = target[1]
                d = self.query(goal, action)
                script = (action, goal)
                
            # if d == "metis_timeout":
            #     reward = -10
            #     return None, reward, d
                
            if d != "exception" and d != "timeout":
                # print("origin: {}".format(pre_target))
                # print("new: {}".format(d))
                # print([pre_target]==d)
                
                # progress has been made
                if [pre_target] != d:
                    self.fringe.remove(pre_target)
                    self.fringe.extend(d)
                    self.scripts.append(script)
                    reward = 0
                else:
                    # nothing changed
                    reward = -2
            else:
                if d == "timeout":
                    reward = -1
                else:
                    # not applicable
                    reward = -2

            if not self.fringe:
                # take the current goal
                new = self.goal
                # get its name
                name = thms[new]
                p = self.get_polish(new)
                entry = {p[0][0][1]: name}
                new_facts.update(entry)
                fact_pool.append(p[0][0][1])
                
                return None, None, 10, True
            else:
                ng, next_state = self.get_states()
                return ng, next_state, reward, False
        else:
            return None, None, None, True
