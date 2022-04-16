# import psutil
import subprocess
from random import sample
from random import shuffle
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
import os
from exp_config import *

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def get_process(pstring):
    pids = []
    fd = os.popen("ps ax | grep " + pstring + " | grep -v grep")
    for line in fd:
        fields = line.split()
        pid = fields[0]
        pids.append(pid)
    fd.close()
    return pids

# def get_process(pstring):
#     p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
#     out, err = p.communicate()
#     pids = []
#     for line in out.splitlines():
#         if b'hol' in line:
#             pid = int(line.split(None, 1)[0])
#             pids.append(pid)
#     return pids


class HolEnv():
    def __init__(self, goals, start=0):
        # Each state is a max_len * (max_assumptions + 1) * max_goals matrix
        # In a simpler case where we don't consider other goals,
        # each state is a max_len * (max_assumptions + 1) matrix

        # maximal length of a goal
        self.max_len = MAX_LEN

        # maximal assumptions to consider
        self.max_assumptions = MAX_ASSUMPTIONS
        # maximal goals to consider
        self.max_contexts = MAX_CONTEXTS
        
        self.goals = goals
        self.counter = start
        self.handling = None
        self.using = None
        
        self.theories = ["listTheory", "bossLib"]
        self.process = pexpect.spawn(HOLPATH)

        # experimental feature
        self.process.delaybeforesend = None
        # import theories
        print("Importing theories...")
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        for i in self.theories:
            self.process.sendline("open {};".format(i).encode("utf-8"))

        # remove built-in simp lemmas
        print("Removing simp lemmas...")
        # self.process.sendline("delsimps [\"HD\", \"EL_restricted\", \"EL_simp_restricted\"];")
        self.process.sendline("delsimps {};".format(dels))
        self.process.sendline("delsimps {};".format(dels2))
        # self.process.sendline("delsimps {};".format(dels3))
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
        # self.goal = sample(self.goals, 1)[0]
        self.goal = self.goals[self.counter%(len(self.goals))]
        # self.history = [([], self.goal)]
        # self.history = [(["Induct_on `m`","conv_tac"], self.goal)]
        
        # a pair of polished goal list and original goal list
        self.fringe = self.get_polish(self.goal)
        
        self.scripts = []
        # goal = self.construct_goal(self.goal)
        # self.process.sendline(goal.encode("utf-8"))        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))
        self.counter += 1

    def get_names(self, exps):
        # look up the names of exps
        names = []
        for e in exps:
            try:
                i = defs[e]
            except:
                # i = new_facts[e]
                # i = pthms[e]
                i = pthms[e][0]
            names.append(i)
        return names
        
    def assemble_tactic(self, tac, args):
        # args is a list of strings
        if tac in thms_tactic:
            names = self.get_names(args)
            action = tac + re.sub("'", "", str(names))
        elif tac in thm_tactic:
            names = self.get_names(args)
            action = tac + " " + names[0]
        else:
            # term tactics will be already assembled
            # no_arg_tactic are handled as is
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
        
        # self.goal = sample(self.goals, 1)[0]
        # if self.counter%(len(self.goals)) == 0:
        #     shuffle(self.goals)

        # TODO: record the previous goal
        
        self.goal = self.goals[self.counter%(len(self.goals))]

        # self.history = [([], self.goal)]
        self.fringe = self.get_polish(self.goal)
        
        self.scripts = []
        self.counter += 1        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))

    def close(self):    
        pids = get_process("hol")
        pidsh = get_process("buildheap")
        print("Found HOL pids: {}".format(pids))
        for pid in (pids+pidsh):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except:                 
                pass                             
            print("Tried closing {}".format(pid))
            
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
        unit_channel = [unit for i in range(self.max_assumptions+1)]
        states = []
        for j in range(self.max_contexts):
            # first 0: take the first goal-assumption pair
            # second 0: take the polished form
            # [(polished form,(assumptions, goal)), ...]
            if j < len(self.fringe):                
                target = self.fringe[j][0]
                if j == 0:
                    p = target[1]
                context = [self.encode(target[1])]
                for i in range(self.max_assumptions):
                    if i < len(target[0]):
                        context.append(self.encode(target[0][i]))
                    else:
                        context.append(unit)
                states.append(context)
            else:
                states.append(unit_channel)
        
        # returns the first polished goal and the encoding of the goal-assumptions pair
        return (p, torch.FloatTensor(states))

    def get_polish(self, raw_goal):
        # raw_goal = ansi_escape.sub('', raw_goal)
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
        # raw_goal = ansi_escape.sub('', raw_goal)
        self.handling = raw_goal
        self.using = tac
        
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
            i = -1


        if i == -1:
            data = "unexpected"
            return data
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
            
            # escape colored characters
            polished_subgoals = ansi_escape.sub('', polished_subgoals)
            subgoals = ansi_escape.sub('', subgoals)
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
                raw = self.process.before.decode("utf-8")
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
        with open("fact_pool.json") as f:
            facts = json.load(f)
        steps = 0
        while self.fringe:
            pre_target = sample(self.fringe, 1)[0]
            
            # take the natural representation
            target = pre_target[1]
            
            tac = sample(tactic_pool, 1)[0]
            
            if tac == "Induct_on":
                g = pre_target[0][1]
                tokens = g.split()
                tokens = list(dict.fromkeys(tokens))
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
                    arg = sample(facts, 1)
                    args.extend(arg)
                action = self.assemble_tactic(tac, args)

            # print("Target: {}\nTactic: {}".format(target, action))
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
                self.reset()
                return "Failed"
        print("Proof found in {} steps.".format(steps))
        self.reset()
        return "Success"

    def random_test(self, num):
        random_proved = 0
        for i in range(num):
            r = self.random_play_no_back()
            if r == "Success":
                random_proved += 1
            print("Proved so far: {}.".format(random_proved))

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
            if d == "unexpected":
                reward = UNEXPECTED_REWARD
                
            elif d != "exception" and d != "timeout":
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
                fact = p[0][0][1]
                # print(fact not in fact_pool)
                if ALLOW_NEW_FACTS:
                    if fact not in fact_pool:
                        fact_pool.append(fact)
                        # print(len(fact_pool))
                
                return None, None, 100, True
            else:
                ng, next_state = self.get_states()
                return ng, next_state, reward, False
        else:
            return None, None, None, True
