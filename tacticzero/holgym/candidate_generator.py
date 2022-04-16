import os
import re
import sys
import json

MODE = "thms"
POLISHED = True
s = []
LEN_FILTER = 300

fn = "raw_data.txt"

with open(fn,"r") as f:
    s = f.read()
            
ss = s.split(";")
sss = []

for i in ss:
    e = re.sub('\n|⊢ ', "", i) # remove useless information
    e = re.sub(' +', " ", e) # shrink spaces
    e = re.sub('⧺', "++", e) # replace chars
    if e:
        sss.append(e)
        
# print(len(sss))
dicts = {}

for i in sss:
    l = i.split("大")
    dicts[l[2]] = (l[0], l[1], l[3], l[4])
    
# print(sss)
# print(list(def_dict.keys()))
# print(len(dicts))

dict_json = json.dumps(dicts)
print("{} theorems stored.".format(len(list(dicts.keys()))))

with open("database.json","w") as f:
    f.write(dict_json)

# if MODE == "defs":
#     if POLISHED:
#         with open("polished_def_dict.json","w") as f:
#             f.write(dict_json)
#         print("Polished definitions saved.")
#     else:
#         with open("def_dict.json","w") as f:
#             f.write(dict_json)
#         print("Definitions saved.")
# elif MODE == "thms":
#     if POLISHED:
#         with open("polished_thm_dict_sorted.json","w") as f:
#             f.write(dict_json)
#         print("Polished theorems saved.")
#     else:
#         with open("thm_dict_sorted.json","w") as f:
#             f.write(dict_json)
#         print("Theorems saved.")


    # print(def_dict["∀P. FIND P = OPTION_MAP SND ∘ INDEX_FIND 0 P"])

# with open("thm_dict.json") as f:
#     dictionary = json.load(f)
#     print(list(dictionary.keys()))
