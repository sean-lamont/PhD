# from subprocess import Popen, PIPE
from random import sample
import pexpect
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from sys import exit
from exp_env import *
from exp_model import *
from exp_config import *
import json
import timeit
# from guppy import hpy
import json

e = HolEnv(GOALS)

start = timeit.default_timer()

# a = e.query("∀l. ZIP (UNZIP l) = l", "Induct_on `l`")
a = e.query("∀l. ZIP (UNZIP l) = l", "simp[]")

stop = timeit.default_timer()

print('Time: {}  '.format(stop - start))
print(a)
