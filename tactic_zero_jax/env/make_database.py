import json

with open("include_probability.txt") as fp:
    x = fp.readlines()

y = "".join(x)

y = y.replace("\n","")

y = y.replace("  "," ")
y = y.replace("  "," ")

z = y.split("|||")


ret = []
buf = []
j = 1
i = 0
while i < len(z):
    #at end of entry
    cur = z[i]
    if j % 6 == 0:
        buf.append(cur)
        j = 1
        ret.append(buf)
        buf = []
    elif cur == 'thm':
        buf.append(cur)
    else:
        buf.append(cur)
        j += 1
    i+=1

#new database mapping from theory-number to values (much smaller key than polished goal)

#1. mapping theory to dependencies (e.g. list-25 : [list-24, bool-23, bool-2])
#2. mapping theory/def name to values 

dep_dict = {}
db_dict = {}

for term in ret:
    #if thm
    if len(term) == 7:
        dep_dict[str(term[0]) + "-" + str(term[3])] = term[5].split(", ")
    
    db_dict[str(term[0]) + "-" + str(term[3])] = [term[0], term[1], term[2][2:], term[3], term[4], term[-1][2:]] 

# #number of examples for training conjecture model
# count = 0
# for i, v in enumerate(dep_dict):
#     count += len(dep_dict[v])
    
# print (count)


with open("dep_data.json", "w") as f:
    json.dump(dep_dict, f)

with open("new_db.json", "w") as f:
    json.dump(db_dict, f)


