#!/usr/bin/env python

import os
import re
import sys
import json
from exp_config import *

AUGMENT = True
s = []
LEN_FILTER = 300

fn = "hammer_stats/vampire_ps_b_128_256.txt"

with open(fn,"r") as f:
    s = f.read()


proved_fn = "proved_theorems_tacticzero.json"

with open(proved_fn) as f:
    proved_by_tacticzero = json.load(f)




match = re.findall(r'Failed theorem:\n(.*?)\nEnd printing', s, re.DOTALL)

thms = [re.sub("\n +| +", " ", i) for i in match]

thms = [re.sub("\n", " ", i) for i in thms]

thms = [re.sub('⧺', "++", i) for i in thms] # replace chars

overlap = []
for i, t in enumerate(thms):
    if t not in plain_database.keys():
        print("Not found in database")
        print(t)
        print(i)
        break
    else:
        proved_by_hammer = [h for h in plain_database.keys() if h not in thms]


for t in proved_by_hammer:
    if t in proved_by_tacticzero:
        overlap.append(t)

print(len(overlap))
# print(match)
# ss = s.split("End printing")
# sss = []

# for i in ss:
#     e = re.sub(' +', " ", i) shrink spaces
#     if e:
#         sss.append(e)

# print(len(sss))
# dicts = {}

# for i in sss:
#     try:
#         l = i.split("大")
#         if len(l[2].split()) > LEN_FILTER:
#             continue
#         dicts[l[2]] = (l[0], l[1], l[3], l[4], l[5])
#     except:
#         print(l)
#         exit()

# dict_json = json.dumps(dicts)
# print("{} theorems stored.".format(len(list(dicts.keys()))))

# if AUGMENT:
#     with open("augmented_database.json","w") as f:
#         f.write(dict_json)
# else:
#     with open("database.json","w") as f:
#         f.write(dict_json)
