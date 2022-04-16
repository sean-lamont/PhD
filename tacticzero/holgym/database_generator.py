import os
import re
import sys
import json

AUGMENT = True
s = []
LEN_FILTER = 300

fn = "raw_data.txt"

with open(fn,"r") as f:
    s = f.read()
            
ss = s.split("小")
sss = []

for i in ss:
    e = re.sub('\n|⊢ ', "", i) # remove useless information
    e = re.sub(' +', " ", e) # shrink spaces
    e = re.sub('⧺', "++", e) # replace chars
    if AUGMENT:
        d = re.sub('C\$min\$ ==>','D$min$ ==>', e) # data augmentation (also to distinguish assumptions and implications)    
    if e:
        sss.append(e)
        if AUGMENT:
            sss.append(d)

# print(len(sss))
dicts = {}

for i in sss:
    try:
        l = i.split("大")
        if len(l[2].split()) > LEN_FILTER:
            continue
        dicts[l[2]] = (l[0], l[1], l[3], l[4], l[5])
    except:
        print(l)
        exit()

dict_json = json.dumps(dicts)
print("{} theorems stored.".format(len(list(dicts.keys()))))

if AUGMENT:
    with open("augmented_database.json","w") as f:
        f.write(dict_json)
else:
    with open("database.json","w") as f:
        f.write(dict_json)
