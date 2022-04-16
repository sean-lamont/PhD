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
import json

HOLPATH = "/home/minchao/HOL/bin/hol"
dictionary = {}


fact_pool = list(defs.keys())

with open("def_dict.json") as f:
    defs = json.load(f)
    # print(list(dictionary.keys()))

class HolEnv():
    def __init__(self, goals):
        # Each state is a max_len * (max_assumptions + 1) * max_goals matrix
        # In a simpler case where we don't consider other goals,
        # each state is a max_len * (max_assumptions + 1) matrix

        # maximal length of a goal
        self.max_len = 64

        # maximal assumptions to consider
        self.max_assumptions = 3
        # maximal goals to consider
        self.max_goals = 4
        
        self.goals = goals

        self.tactic_pool = ["simp", "fs", "strip_tac", "metis_tac", "irule", "drule", "Induct_on"]

        self.action_pool = ["simp[]", "gen_tac", "conv_tac", "metis_tac[]", "strip_tac[]", "Induct_on `m`", "Induct_on `n`", "strip_tac"]

        self.theories = ["listTheory", "bossLib"]
        self.process = pexpect.spawn(HOLPATH)

        # import theories
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        for i in self.theories:
            self.process.sendline("open {};".format(i).encode("utf-8"))
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
            
        # consumes hol4 head
        self.process.expect('\r\n>')
        
        # setup the goal
        self.goal = sample(goals, 1)[0]
        self.history = [([], self.goal)]
        # self.history = [(["Induct_on `m`","conv_tac"], self.goal)]
        self.scripts = []
        # goal = self.construct_goal(self.goal)
        # self.process.sendline(goal.encode("utf-8"))        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))

    def construct_goal(self, goal):
        s = "g " + "`" + goal + "`;"
        return s

    def construct_tactic(self, tac):
        s = "e " + "(" + tac + ");"
        return s

    def reset(self):
        self.__init__(self.goals)

    def get_states(self):
        # only goals are considered for now

        # create an empty entry
        unit = self.max_len * [0]
        states = list(map(lambda x : self.encode(x[1]), self.history))
        # pad with empty entries
        states = (states + self.max_goals * [unit])[:self.max_goals]
        # return torch.tensor(states, dtype=torch.float)
        return torch.FloatTensor(states)

    # def get_states(self):
    #     # this always assumes that history is not empty
    #     # get the first goal-assumptions pair in history
        
    #     # create an empty entry
    #     unit = self.max_len * [0]
    #     target = self.history[0]
    #     state = [self.encode(target[1])]
    #     for i in range(self.max_assumptions):
    #         if i < len(target[0]):
    #             state.append(self.encode(target[0][i]))
    #         else:
    #             state.append(unit)

    #     return torch.FloatTensor(state)


    def query(self, goal, tac):
        goal = self.construct_goal(goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")
        tactic = self.construct_tactic(tac)
        self.process.sendline(tactic.encode("utf-8"))
        i = self.process.expect([": proof", "Initial goal proved", "Exception", "error"])
        if i == 0:
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            # print(self.process.readline())            
            self.process.expect("val it =")
            
            # this doesn't seem robust
            self.process.expect([": goal list", ":\r\n"])
            raw = self.process.before.decode("utf-8")
            subgoals = re.sub("“|”","\"", raw)
            subgoals = re.sub("\r\n +"," ", subgoals)
            data = eval(subgoals)
        elif i == 1:
            data = []
        else:
            # print("Exception raised when applying tactic {} to {}.".format(tac, goal))
            data = None
        
        # clear stack and consume the remaining stdout
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        return data

    def revert(self, assumptions, goal):
        # order shouldn't matter
        for i in assumptions: 
            goal = "(" + i + ")" + "==>" + goal
        return goal

    def encode(self, s):
        r = []
        for c in s:
            if c not in dictionary:
                dictionary[c] = len(dictionary) + 1
            r.append(dictionary[c])
        # pad r with 0's
        r = (r + self.max_len * [0])[:self.max_len]
        return r
    
    def random_play_no_back(self):
        steps = 0
        while self.history:
            target = sample(self.history, 1)[0]
            action = sample(self.action_pool, 1)[0]
            print("Target: {}\nTactic: {}".format(target, action))
            if target[0]:
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
            else:
                goal = target[1]
                d = self.query(goal, action)
            if d != None:
                self.history.remove(target)
                self.history.extend(d)

            steps += 1
            if steps == 50:
                print("Failed")
                return steps
        print("Proof found in {} steps.".format(steps))
        return steps

    def step(self, action):
        action = self.action_pool[action]
        if self.history:
            target = self.history[0]
            if target[0]:
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
                script = ("rpt strip_tac >> " + action, goal)
            else:
                goal = target[1]
                d = self.query(goal, action)
                script = (action, goal)
            if d != None:
                self.history.remove(target)
                self.history.extend(d)
                reward = 0
                self.scripts.append(script)
            else:
                reward = -1

            if not self.history:
                return None, 1, True
            else:
                next_state = self.get_states()
                return next_state, reward, False
        else:
            return None, None, True
