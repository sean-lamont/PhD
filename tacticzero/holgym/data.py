import os
import re
import sys
import json

MODE = "thms"
POLISHED = True
s = []
LEN_FILTER = 300


if MODE == "thms":
    if POLISHED:
        fn = "polished_core_theories_defs.txt"
    else:
        fn = "core_theories_defs.txt"
else:
    if POLISHED:
        fn = "polished_core_theories_thms.txt"
    else:
        fn = "core_theories_thms.txt"

fn = "core_theories.txt"

# if MODE == "defs":
#     if POLISHED:
#         with open("polished_defs.txt","r") as f:
#             s = f.read()
#     else:
#         with open("defs.txt","r") as f:
#             s = f.read()
# elif MODE == "thms":
#     if POLISHED:
#         with open("polished_thms.txt","r") as f:
#             s = f.read()
#     else:
#         with open("thms.txt","r") as f:
#             s = f.read()    

# if MODE == "defs":
#     if POLISHED:
#         with open("polished_defs.txt","r") as f:
#             s = f.read()
#     else:
#         with open("defs.txt","r") as f:
#             s = f.read()
# elif MODE == "thms":
#     if POLISHED:
#         with open("polished_thms_sorted.txt","r") as f:
#             s = f.read()
#     else:
#         with open("thms_sorted.txt","r") as f:
#             s = f.read()    


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

dicts = {}

for i in sss:
    l = i.split("大")
    if len(l[1].split()) > LEN_FILTER:
        continue
    dicts[l[1]] = (l[0], l[2])
    
# print(sss)
# print(list(def_dict.keys()))
# print(len(dicts))

dict_json = json.dumps(dicts)
print("{} theorems stored.".format(len(list(dicts.keys()))))

with open("core_theories_dict_sorted.json","w") as f:
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
