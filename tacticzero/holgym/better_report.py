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
# stats_2020_05_18_14_16_59.json
# stats_2020_05_19_22_27_56.json
# 2021_01_13_13_38_57_checkpoint
with open("/scratch1/wu099/temp/2021_01_13_13_38_57_checkpoint/statistics.json") as f:
    statistics = json.load(f)
    
training_size = sum([len(l) for l in statistics["train_episodic_rewards"]])
training_iteration_average = [np.mean(l) for l in statistics["train_episodic_rewards"]]
training_average = np.mean(training_iteration_average)
training_proved = sum(statistics["train_proved_per_iteration"])

print("Valid training size: {}".format(training_size))
print("Average total reward of training: {}".format(training_average))
print("Successful proofs of training: {}".format(training_proved))
print("Training success rate: {}".format(training_proved/training_size))

validation_size = sum([len(l) for l in statistics["validation_episodic_rewards"]])
validation_iteration_average = [np.mean(l) for l in statistics["validation_episodic_rewards"]]
validation_average = np.mean(validation_iteration_average)
validation_proved = sum(statistics["validation_proved_per_iteration"])

print("Valid validation size: {}".format(validation_size))
print("Average total reward of validation: {}".format(validation_average))
print("Successful proofs of validation: {}".format(validation_proved))
print("Validation success rate: {}".format(validation_proved/validation_size))

training_all_episodes_rewards = sum(statistics["train_episodic_rewards"], [])
validation_all_episodes_rewards = sum(statistics["validation_episodic_rewards"], [])

def plot_iteration_rewards(mode):
    plt.figure(1)
    # plt.clf()
    if mode == "training":
        durations_t = torch.tensor(training_iteration_average, dtype=torch.float)
        plt.title('Training')
        label = "Train"
    if mode == "validation":
        durations_t = torch.tensor(validation_iteration_average, dtype=torch.float)
        plt.title('Validation')
        label = "Validation"
    
    plt.xlabel('Iterations')
    plt.ylabel('Average total rewards')
    # plt.plot(durations_t.numpy())
    # astplot = torch.tensor(ast, dtype=torch.float)
    plt.plot(durations_t, label=label)
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


def plot_moving_rewards(mode, window_size):
    plt.figure(1)
    # plt.clf()

    if mode == "training":
        i = 0
        moving_averages = []
        while i < len(training_all_episodes_rewards) - window_size + 1:
            this_window = training_all_episodes_rewards[i : i + window_size]
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += 1
        durations_t = torch.tensor(moving_averages, dtype=torch.float)
        plt.title('Training')
        label = "Train"
    if mode == "validation":
        i = 0
        moving_averages = []
        while i < len(validation_all_episodes_rewards) - window_size + 1:
            this_window = validation_all_episodes_rewards[i : i + window_size]
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
            i += 1
        durations_t = torch.tensor(moving_averages, dtype=torch.float)
        plt.title('Validation')
        label = "Validation"
    
    plt.xlabel('Episodes')
    plt.ylabel('Average total rewards')
    # plt.plot(durations_t.numpy())
    # astplot = torch.tensor(ast, dtype=torch.float)
    plt.plot(durations_t, label=label)
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


def plot_success(mode):
    plt.figure(1)
    # plt.clf()
    if mode == "training":
        durations_t = torch.tensor(statistics["train_proved_per_iteration"], dtype=torch.float)
        plt.title('Training', fontsize=24)
        label = "Train"
    if mode == "validation":
        durations_t = torch.tensor(statistics["validation_proved_per_iteration"], dtype=torch.float)
        plt.title('Validation')
        label = "Validation"

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    # ax.set_yticklabels(y_ticks, rotation=0, fontsize=8)
    plt.xlabel('Iterations', fontsize=24)
    plt.ylabel('Proved',fontsize=24)
    # plt.plot(durations_t.numpy())
    # astplot = torch.tensor(ast, dtype=torch.float)
    plt.plot(durations_t, label="updating all")
    plt.legend(loc='lower right', prop={'size': 22})
    # # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 1000, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(999), means))
    #     plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# statistics["train_proved_per_iteration"] = statistics["train_proved_per_iteration"][:724]

# stats_json = json.dumps(statistics)
# with open("/scratch1/wu099/temp/2021_01_27_22_08_42_checkpoint/statistics.json","w") as f:
#     f.write(stats_json)

# print("Statistics saved.")
