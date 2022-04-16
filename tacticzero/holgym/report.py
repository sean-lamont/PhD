import os
import re
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#with open("latest.out","r") as f:
#    s = f.read()

# with open("good_models/10000-1e-5-1e-4-1e-5.out","r") as f:
#     s = f.read()

# with open("log/random/random_seq2seq_42534105.log","r") as f:
#     s = f.read()
with open("log/train/fixed_sigmoid.log","r") as f:
    s = f.read()

l0 = re.findall("Total games: (.*?)\n", s)

iter_num = float(l0[0])

l1 = re.findall("Total: (.*?)\n", s)

all_reward = []

for i in l1:
    k = float(i)
    all_reward.append(k)

print("Average total reward: {}".format(np.mean(all_reward)))

l2 = re.findall("Proved in (.*?) steps.", s)

all_steps = []

for i in l2:
    k = float(i)
    all_steps.append(k)


print("Average steps taken for a proof: {}".format(np.mean(all_steps)))

l3 = re.findall("Total: (.*?)\nProof", s)

proved_reward = []

for i in l3:
    k = float(i)
    proved_reward.append(k)


l4 = re.findall("Proved so far: (.*?)\n.*\nModels saved.\nDictionary", s)

l5 = re.findall("Proved so far: (.*?)\n", s)

proved = []

for i in l5:
    k = float(i)
    proved.append(k)

sl5 = np.array_split(proved, len(proved)//iter_num)

proved_iter = []

for i in sl5:
    proved_iter.append(i[-1])

proved_plot = []

for i,e in enumerate(proved_iter):
    if i == 0:
        proved_plot.append(e)
    else:
        p = proved_iter[i-1]
        proved_plot.append(e-p)

proved_percentage = [i*10 for i in proved_plot]


sl6 = np.array_split(all_reward, len(proved)//iter_num)

sl6 = [np.mean(i) for i in sl6]

sl7 = np.array_split(all_steps, len(proved)//iter_num)

sl7 = [np.mean(i) for i in sl7]

# for (i, e) in enumerate(sl5):
#     if i == 0:
#         proved_plot.append(e[-1])
#     else:
#         p = proved_plot[-1]
#         proved_plot.append(e[-1]-p)
    

print("Average proved reward: {}".format(np.mean(proved_reward)))

print("Average recent 2000 rewards: {}".format(np.mean(all_reward[-2000:])))

print("Average recent 1000 rewards: {}".format(np.mean(all_reward[-1000:])))

print("Average recent 500 rewards: {}".format(np.mean(all_reward[-500:])))

print("Average recent 200 rewards: {}".format(np.mean(all_reward[-200:])))

print("Average recent 100 rewards: {}".format(np.mean(all_reward[-100:])))

print("Average recent 50 rewards: {}".format(np.mean(all_reward[-50:])))

print("Number of proved theorems: {}".format(l4[0]))




# print("50 episode-wise rewards: {}".format(av))

# all_mean = np.mean(all_reward)
# all_std = np.std(all_reward)
# for i in range(len(all_reward)):
#     all_reward[i] = (all_reward[i] - all_mean) / (all_std + np.finfo(np.float32).eps) # eps is important, otherwise we get divided by 0

# sl1 = np.array_split(all_reward,len(all_reward)//500)

# ar = []

# for i in sl1:
#     a = np.mean(i)
#     ar.append(a)

# sl2 = np.array_split(all_steps,len(all_steps)//50)

# ast = []

# for i in sl2:
#     a = np.mean(i)
#     ast.append(a)

def plot_iterations():
    plt.figure(1)
    # plt.clf()
    durations_t = torch.tensor(sl6, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Iteration')
    plt.ylabel('Average rewards')
    # plt.plot(durations_t.numpy())
    # astplot = torch.tensor(ast, dtype=torch.float)
    plt.plot(durations_t, label='Trained')
    plt.legend(loc='lower right')
    # # Take 100 episode averages and plot them too
    # if len(durations_t) >= 10:
    #     means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(9), means))
    #     plt.plot(means.numpy())
    #     plt.legend(loc='lower right')
    # plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_success():
    plt.figure(1)
    # plt.clf()
    durations_t = torch.tensor(proved_plot, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episodes')
    plt.ylabel('Proved')
    # plt.plot(durations_t.numpy())
    # astplot = torch.tensor(ast, dtype=torch.float)
    plt.plot(durations_t)

    # # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 1000, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(999), means))
    #     plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


        
# sl1 = np.array_split(all_reward,len(all_reward)//500)

# av = []

# for i in sl1:
#     a = np.mean(i)
#     av.append(a)

# print("500 episode-wise rewards: {}".format(av))

# sl2 = np.array_split(all_steps,len(all_steps)//100)

# av = []

# for i in sl2:
#     a = np.mean(i)
#     av.append(a)

# print("100 episode-wise proved steps: {}".format(av))

# sl3 = np.array_split(proved_reward,len(proved_reward)//100)

# av = []

# for i in sl3:
#     a = np.mean(i)
#     av.append(a)

# print("100 episode-wise proved rewards: {}".format(av))
