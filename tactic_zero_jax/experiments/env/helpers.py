#TODO Move to another file 

import torch
def get_polish(raw_goal):
        goal = construct_goal(raw_goal)
        process.sendline(goal.encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("top_goals();".encode("utf-8"))
        process.expect("val it =")
        process.expect([": goal list", ":\r\n +goal list"])

        polished_raw = process.before.decode("utf-8")
        polished_subgoals = re.sub("“|”","\"", polished_raw)
        polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

        pd = eval(polished_subgoals)
        
        process.expect("\r\n>")
        process.sendline("drop();".encode("utf-8"))
        process.expect("\r\n>")
        process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        process.expect("\r\n>")

        data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                 "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data 
    
def construct_goal(goal):
    s = "g " + "`" + goal + "`;"
    return s

def gather_encoded_content_(history, encoder):
    fringe_sizes = []
    contexts = []
    reverted = []
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        g = revert_with_polish(e)
        reverted.append(g.strip().split())
    out = []
    sizes = []
    for goal in reverted:
        out_, sizes_ = encoder.encode([goal])
        out.append(torch.cat(out_.split(1), dim=2).squeeze(0))
        sizes.append(sizes_)

    representations = out

    return representations, contexts, fringe_sizes

def parse_theory(pg):
    theories = re.findall(r'C\$(\w+)\$ ', pg)
    theories = set(theories)
    for th in EXCLUDED_THEORIES:
        theories.discard(th)
    return list(theories)

def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions): 
        #goal = "@ @ D$min$==> {} {}".format(i, goal)
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)

    return goal 

def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs
