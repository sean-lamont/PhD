import subprocess
# import psutil
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
from copy import deepcopy
from igraph import *
import plotly.graph_objects as go


ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def revert_assumptions(assumptions, goal):
    # order shouldn't matter
    for i in assumptions: 
        goal = "(" + i + ")" + " ==> " + "(" + goal + ")"
    return goal    

def get_process(pstring):
    pids = []
    fd = os.popen("ps ax | grep " + pstring + " | grep -v grep")
    for line in fd:
        fields = line.split()
        pid = fields[0]
        pids.append(pid)
    fd.close()
    return pids


class HolEnv():
    def __init__(self, goals, start=0):
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
        
        self.fringe = {"content": self.get_polish(self.goal),
                       "parent": None,
                       "goal": None,
                       "by_tactic":""}

        # a fringe is a list of the form
        # [((polished assumptions, polished goal),
        #   (assumptions, goal)),
        #   ...]
        self.history = [self.fringe]
        self.action_history = [] # list of tuples (id, id, tactic)
        
        print("Initialization done. Main goal is:\n{}.".format(self.goal))
        self.counter += 1

    def get_names(self, exps):
        # look up the names of exps
        # TODO: rewrite this
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
        # TODO: record the previous goal
                
        self.goal = self.goals[self.counter%(len(self.goals))]
        self.fringe = {"content": self.get_polish(self.goal),
                       "parent": None,
                       "goal": None,
                       "by_tactic":""}

        self.history = [self.fringe]
        self.action_history = []
        
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

        data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                 "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data # list(zip(pd, [([], raw_goal)]))
    
    def query(self, raw_goal, tac):
        # print("content1:{}".format(self.process.before.decode("utf-8")))
        # print("goal is: {}".format(goal))
        # print("tac is: {}".format(tac))
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
            # data = list(zip(pd, d))
            data = zip(pd, d)
            data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                     "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                    for e in data]
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

    def step(self, action):
        if action in self.action_history:
            return -2, False
        fringe_id, goal_id, tactic = action
        target_fringe = self.history[fringe_id]
        pre_target = target_fringe["content"][goal_id]
        target = pre_target["plain"]
        if target["assumptions"]:
            # there are assumptions
            goal = revert_assumptions(target["assumptions"], target["goal"])
            d = self.query(goal, "rpt strip_tac >> " + tactic)

        else:
            # no assumptions
            goal = target["goal"]
            d = self.query(goal, tactic)

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

                for f in self.history:
                    # if results in something occurred before 
                    if new_content == f["content"]:
                        return -0.5, False
                    
                new_fringe = {"content": new_content,
                              "parent": fringe_id,
                              "goal": goal_id,
                              "by_tactic": tactic}

                self.history.append(new_fringe)
                self.action_history.append(action)
                reward = 0

                if new_content == []:
                    new = self.goal
                    # get its name
                    name = thms[new]
                    p = self.get_polish(new)
                    entry = {p[0]["polished"]["goal"]: name}
                    new_facts.update(entry)
                    fact = p[0]["polished"]["goal"]
                    # print(fact not in fact_pool)
                    if ALLOW_NEW_FACTS:
                        if fact not in fact_pool:
                            fact_pool.append(fact)
                            # print(len(fact_pool))

                    return 100, True

            else:
                # nothing changed
                reward = -2
        else:
            if d == "timeout":
                reward = -1
            else:
                # not applicable
                reward = -2

        return reward, False
        

def extract_proof(history):
    qed = history[-1]
    path = []
    proof = []
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    for _ in count():
        parent_fringe = history[parent_fringe_id]
        path.append(history[parent_fringe_id])
        parent_goal = parent_fringe["content"][parent_goal_id]
        plain_text_goal = parent_goal["plain"]["goal"]
        proof.append((plain_text_goal, tactic))
        
        if parent_fringe_id == 0:
            proof.reverse()
            return proof
        
        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]


def extract_path_id(history):
    qed = history[-1]
    # print(qed)
    path = [-1]
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    for _ in count():
        parent_fringe = history[parent_fringe_id]
        path.append(parent_fringe_id)
        
        if parent_fringe_id == 0:
            return path
        
        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]


def get_text(fringe):
    # [((polished assumptions, polished goal),(assumptions, goal)),...]
    # t = [p[1] for p in fringe]
    # texts = []
    if not fringe:
        return "QED"
    
    text = ""
    for i,p in enumerate(fringe):
        text += "{}: {}<br>".format(i,p["plain"])
        
    return text[:-4]
    

def make_tree(history):
    es = []
    for i in history:
        p = i["parent"]
        if p != None: # p can be 0
            es.append(((p, history.index(i)), # edge
                       (i["goal"], i["by_tactic"]))) # label

    return es

    
def draw_tree(history, output_graph=False):
    nv = len(history)
    eslb = make_tree(history)
    es = [i[0] for i in eslb]
    g = Graph(nv, es, True)
    
    g.vs["goals"] = [get_text(i["content"]) for i in history]
    g.es["by applying tactic on"] = [i[1] for i in eslb]
    g.vs["label"] = g.vs["goals"]
    g.es["label"] = g.es["by applying tactic on"]
    # g.add_vertices(nv)
    # g.add_edges(es)
    
    # if output_graph:
    #     layout = g.layout("rt")
    #     plot(g, layout = layout, bbox = (1024, 1024))
    # print(g)
    # return summary(g)

    lay = g.layout('rt')

    position = {k: lay[k] for k in range(nv)}
    Y = [lay[k][1] for k in range(nv)]
    M = max(Y)
    
    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []    
    for edge in es:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    # fs = [get_text(i["content"]) for i in history]
    
    vlabels = [get_text(i["content"]) for i in history]
    elabels = g.es["by applying tactic on"]
    
    # calculate the middle points for edge labels
    Xel = []
    Yel = []
    for edge in es:
        Xel+=[0.5*(position[edge[0]][0]+position[edge[1]][0])]
        Yel+=[0.5*(2*M-position[edge[0]][1]+2*M-position[edge[1]][1])]
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines+text',
                             name='Actions',
                             line=dict(color='rgb(0,0,0)', width=1),
                             # text=elabels,
                             # hoverinfo='text',
                             opacity=0.8
                             # hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='Dead',
                             marker=dict(symbol='circle-dot',
                                         size=18,
                                         color='#FF0000',    #'#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                             ),
                             text=vlabels,
                             hoverinfo='text',
                             opacity=0.8
    ))
    fig.add_trace(go.Scatter(x=Xel,
                             y=Yel,
                             mode='markers',
                             name='Action labels',
                             marker=dict(color='rgb(210,210,210)',
                                         size=1),
                             text=elabels,
                             hoverinfo='text'
    ))
    if "QED" in g.vs["goals"]:
        path = extract_path_id(history)
        pathXn = [Xn[i] for i in path]
        pathYn = [Yn[i] for i in path]
        plabels = [vlabels[i] for i in path]

        fig.add_trace(go.Scatter(x=pathXn,
                                 y=pathYn,
                                 mode='markers',
                                 name='Path',
                                 marker=dict(symbol='circle-dot',
                                             size=18,
                                             color='#6175c1',    #'#DB4551',
                                             line=dict(color='rgb(50,50,50)', width=1)
                                 ),
                                 text=plabels,
                                 hoverinfo='text',
                                 opacity=0.8
        ))

    fig.show()


def encode(s):
    # s is a string
    # print("Encoding: {}".format(s))
    s = s.split()
    r = []
    for c in s:
        if c not in dictionary:
            dictionary[c] = len(dictionary) + 1
        r.append(dictionary[c])
    # pad r with 0's
    r = (r + MAX_LEN * [0])[:MAX_LEN]

    # a list whose length is max_len
    return r

# def encode_goal(goal): # goal is an (assumption, goal) pair
#     # create an empty entry
#     unit = MAX_LEN * [0]
#     target = goal["polished"] # the polished form
#     polished_goal = target["goal"]
#     context = [encode(polished_goal)]
#     for i in range(MAX_ASSUMPTIONS):
#         if i < len(target["assumptions"]):
#             context.append(encode(target["assumptions"][i]))
#         else:
#             context.append(unit)

#     # returns the first polished goal and the encoding of the goal-assumptions pair
#     return (polished_goal, torch.FloatTensor(context))


def context_encoder(context): # a context is an (assumption, goal) pair
    # create an empty entry
    unit = MAX_LEN * [0]
    target = context["polished"] # the polished form
    polished_goal = target["goal"]
    context = [encode(polished_goal)]
    for i in range(MAX_ASSUMPTIONS):
        if i < len(target["assumptions"]):
            context.append(encode(target["assumptions"][i]))
        else:
            context.append(unit)

    # returns the first polished goal and the encoding of the goal-assumptions pair
    return torch.FloatTensor(context)


def context_seq2seq_encoder(context): # a context is an (assumption, goal) pair
    # create an empty entry
    unit = MAX_LEN * [0]
    target = context["polished"] # the polished form
    polished_goal = target["goal"]
    context = [encode(polished_goal)]
    for i in range(MAX_ASSUMPTIONS):
        if i < len(target["assumptions"]):
            context.append(encode(target["assumptions"][i]))
        else:
            context.append(unit)

    # returns the first polished goal and the encoding of the goal-assumptions pair
    return torch.FloatTensor(context)


# def gather_goals(history):
#     fringe_sizes = []
#     goals = []
#     for i in history:
#         c = i["content"]
#         goals.extend(c)
#         fringe_sizes.append(len(c))
#     return goals, fringe_sizes


# def gather_encoded_content(history):
#     fringe_sizes = []
#     goals = []
#     for i in history:
#         c = i["content"]
#         goals.extend(c)
#         fringe_sizes.append(len(c))
#     representations = []
#     for g in goals:
#         _, encoded = encode_goal(g)
#         representations.append(encoded.unsqueeze(0))
#     return torch.stack(representations), goals,fringe_sizes

def gather_encoded_content(history, encoder):
    fringe_sizes = []
    contexts = []
    representations = []
    # representations_dict = {}
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        encoded = encoder(e)
        # representations_dict[e] = (encoded.unsqueeze(0))
        representations.append(encoded.unsqueeze(0))
    return torch.stack(representations), contexts, fringe_sizes


def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs



