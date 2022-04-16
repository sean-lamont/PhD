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

with open("validation_data.json") as f:
    VALIDATION_SET = json.load(f)
print("Validation data loaded.")

termlist = []
for i in VALIDATION_SET:
    # e = re.sub('\"', "``", i) # remove useless information
    e = "``" + i + "``"
    termlist.append(e)

# print(str(termlist))
# e = re.sub('\'``|``\'|\"``|``\"', "``", str(termlist)) # remove useless information

# e = re.sub('\'|\"', "", str(termlist)) # remove useless information

e = re.sub('\'``|``\'', "``", str(termlist)) # remove useless information

e = re.sub('\"', "", e) # remove useless information

print(e)

# with open(VALIDATION_SET,"w") as f:
#     f.write(termlist)
