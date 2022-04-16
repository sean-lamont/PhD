from new_env import *

# t = [('∀xs ys. MAP f1 xs ++ MAP g1 ys = MAP f2 xs ++ MAP g2 ys ⇔ MAP f1 xs = MAP f2 xs ∧ MAP g1 ys = MAP g2 ys', 'rw[listTheory.ALL_DISTINCT_EL_IMP, listTheory.DROP_def, listTheory.LAST_CONS_cond, listTheory.MEM_ZIP, listTheory.MAP]'), ('MAP f1 xs ⧺ MAP g1 ys = MAP f2 xs ⧺ MAP g2 ys ⇔ MAP f1 xs = MAP f2 xs ∧ MAP g1 ys = MAP g2 ys', 'Induct_on `xs`'), ('MAP f1 [] ⧺ MAP g1 ys = MAP f2 [] ⧺ MAP g2 ys ⇔ MAP f1 [] = MAP f2 [] ∧ MAP g1 ys = MAP g2 ys', 'Induct_on `ys`'), ('∀h. MAP f1 (h::xs) ⧺ MAP g1 ys = MAP f2 (h::xs) ⧺ MAP g2 ys ⇔ MAP f1 (h::xs) = MAP f2 (h::xs) ∧ MAP g1 ys = MAP g2 ys', 'Induct_on `xs`'), ('MAP f1 [] ⧺ MAP g1 [] = MAP f2 [] ⧺ MAP g2 [] ⇔ MAP f1 [] = MAP f2 [] ∧ MAP g1 [] = MAP g2 []', 'fs[listTheory.FLAT, listTheory.FLAT, listTheory.MAP, listTheory.MAP, listTheory.LIST_BIND_APPEND]'), ('∀h. MAP f1 [] ⧺ MAP g1 (h::ys) = MAP f2 [] ⧺ MAP g2 (h::ys) ⇔ MAP f1 [] = MAP f2 [] ∧ MAP g1 (h::ys) = MAP g2 (h::ys)', 'fs[listTheory.GENLIST, listTheory.MAP, listTheory.FLAT, listTheory.MAP, listTheory.LENGTH_EQ_NUM]'), ('(MAP f1 [] ⧺ MAP g1 ys = MAP f2 [] ⧺ MAP g2 ys ⇔ MAP f1 [] = MAP f2 [] ∧ MAP g1 ys = MAP g2 ys) ⇒ (MAP f1 [h] ⧺ MAP g1 ys = MAP f2 [h] ⧺ MAP g2 ys ⇔ MAP f1 [h] = MAP f2 [h] ∧ MAP g1 ys = MAP g2 ys)', 'rw[listTheory.SHORTLEX_def, listTheory.EXISTS_DEF, listTheory.MAP, listTheory.MAP, listTheory.MAP]'), ("∀h'. (MAP f1 (h'::xs) ⧺ MAP g1 ys = MAP f2 (h'::xs) ⧺ MAP g2 ys ⇔ MAP f1 (h'::xs) = MAP f2 (h'::xs) ∧ MAP g1 ys = MAP g2 ys) ⇒ (MAP f1 (h::h'::xs) ⧺ MAP g1 ys = MAP f2 (h::h'::xs) ⧺ MAP g2 ys ⇔ MAP f1 (h::h'::xs) = MAP f2 (h::h'::xs) ∧ MAP g1 ys = MAP g2 ys)", 'fs[listTheory.LENGTH, listTheory.MAP, listTheory.LIST_REL_O, listTheory.LLEX_def, listTheory.MAP]'), ("(f1 h = f2 h ∧ f1 h' = f2 h') ∧ MAP f1 xs ⧺ MAP g1 ys = MAP f2 xs ⧺ MAP g2 ys ⇔ (f1 h = f2 h ∧ f1 h' = f2 h' ∧ MAP f1 xs = MAP f2 xs) ∧ MAP g1 ys = MAP g2 ys", 'metis_tac[listTheory.UNZIP_THM, listTheory.FILTER, listTheory.MAP, listTheory.FILTER, listTheory.LIST_TO_SET_MAP]')]


# with open("bug_history2.txt", "r") as f:
#     bug_history = f.read()
#     bug_history = eval(bug_history)
# print("Bug loaded.")

# # print(bug_history[0]["content"][0]["plain"]["goal"])
# # m = construct_map(bug_history)
# # s = reconstruct_proof(bug_history)
# e = HolEnv("T")
# if check_proof(e, bug_history):
#     print("check passed")
# else:
#     print("check failed")



# saved_replays = "replays.json"
# try:
#     with open(saved_replays) as f:
#         replays = json.load(f)
#     print("Replays loaded.")
# except:
#     replays = {}
#     print("No initial replays.")


# e = HolEnv("T")
# bugs = []
# for i,r in enumerate(replays.values()):
#     if check_proof(e, r[0]):
#         print("check passed")
#     else:
#         print("check failed")
#         bugs.append(i)

# print("Failures: {}".format(bugs))

#51 123

# bug = list(replays.values())[2]

# print(check_proof(e, bug[0]))
# print(bug[0][0]["content"][0]["plain"]["goal"])
# print(extract_proof(bug[0]))
# print(reconstruct_proof(bug[0]))



saved_failures = "failures_without_arith_cache.json"
try:
    with open(saved_failures) as f:
        failures = json.load(f)
    print("Failures loaded.")
except:
    failures = {}
    print("No initial replays.")

# failures = failures[1:]
e = HolEnv("T")
bugs = []
for i,r in enumerate(failures):
    if check_proof(e, r):
        print("check passed")
    else:
        print("check failed")
        bugs.append(i)

print("Failures: {}".format(bugs))

# print(failures[6][0]["content"][0]["plain"]["goal"])
# p = extract_proof(failures[6])
# draw_tree(failures[0])
# # m = construct_map(failures[10])
# s = reconstruct_proof(failures[0])
