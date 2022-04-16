import json
import re
import random
import pickle
random.seed(0)

HOLPATH = "/home/wu099/temp/HOL/bin/hol --maxheap=256"

CACHE_PATH = "/scratch1/wu099/temp/HOL_cache_normalized.pkl"

try: 
    with open("dict.json") as f:
        dictionary = json.load(f)
except:
    dictionary = {}

try: 
    with open("provables.json") as f:
        provables = json.load(f)
except:
    provables = []

provables = [t[0] for t in provables]
    
try:
    with open("bigger_new_facts.json") as f:
        new_facts = json.load(f)
except:
    new_facts = {}

try: 
    with open(CACHE_PATH, "rb") as f:
        HOL_cache = pickle.load(f)
except:
    HOL_cache = {}


MAX_LEN = 128
MAX_ASSUMPTIONS = 3
MAX_CONTEXTS = 8
PRINT_EXCEPTION = False
UNEXPECTED_REWARD = -1000
TARGET_THEORIES = ["list"]
EXCLUDED_THEORIES = ["min", "bool"]
ARG_LEN = 5
# MODE = "train"
CONTINUE = False
ALLOW_NEW_FACTS = False #True
MORE_TACTICS = True

# DELSIMPS = ["HD", "TL_DEF", "APPEND", "FLAT", "LENGTH", "MAP", "LIST_TO_SET_DEF", "FILTER", "EVERY_DEF", "EXISTS_DEF", "MAP2_DEF", "APPEND_NIL", "LENGTH_APPEND", "MAP_ID", "FLAT_APPEND", "EL_restricted", "EL_simp_restricted", "LIST_REL_def", "LIST_REL_NIL", "REVERSE_DEF", "REVERSE_REVERSE", "REVERSE_11", "MEM_REVERSE", "LENGTH_REVERSE", "REVERSE_EQ_NIL", "REVERSE_EQ_SING", "LAST_CONS", "FRONT_CONS", "LENGTH_FRONT_CONS", "FRONT_CONS_EQ_NIL", "LAST_APPEND_CONS", "TAKE_nil", "TAKE_cons", "DROP_nil", "DROP_cons", "TAKE_0", "TAKE_LENGTH_ID", "LENGTH_TAKE", "DROP_0", "TAKE_DROP", "LENGTH_DROP", "ALL_DISTINCT", "LIST_TO_SET_APPEND", "LIST_TO_SET_EQ_EMPTY", "FINITE_LIST_TO_SET", "LIST_TO_SET_REVERSE", "SET_TO_LIST_EMPTY", "SET_TO_LIST_SING", "ALL_DISTINCT_SET_TO_LIST", "isPREFIX", "isPREFIX_THM", "SNOC", "LENGTH_SNOC", "LAST_SNOC", "FRONT_SNOC", "SNOC_11", "LENGTH_GENLIST", "GENLIST_AUX_compute", "EL_GENLIST", "GENLIST_NUMERALS", "INFINITE_LIST_UNIV", "LENGTH_LUPDATE", "LIST_BIND_THM", "SINGL_LIST_APPLY_L", "SINGL_SINGL_APPLY", "dropWhile_def", "APPEND_11"]

# DELSIMPS = ["HD", "TL_DEF", "APPEND", "FLAT", "LENGTH", "MAP", "LIST_TO_SET_DEF", "FILTER", "EVERY_DEF", "EXISTS_DEF", "MAP2_DEF", "APPEND_NIL", "LENGTH_APPEND", "MAP_ID", "FLAT_APPEND", "EL_restricted", "EL_simp_restricted", "LIST_REL_def", "LIST_REL_NIL", "REVERSE_DEF", "REVERSE_REVERSE", "REVERSE_11", "MEM_REVERSE", "LENGTH_REVERSE", "REVERSE_EQ_NIL", "REVERSE_EQ_SING", "LAST_CONS", "FRONT_CONS", "LENGTH_FRONT_CONS", "FRONT_CONS_EQ_NIL", "LAST_APPEND_CONS", "TAKE_nil", "TAKE_cons", "DROP_nil", "DROP_cons", "TAKE_0", "TAKE_LENGTH_ID", "LENGTH_TAKE", "DROP_0", "TAKE_DROP", "LENGTH_DROP", "ALL_DISTINCT", "LIST_TO_SET_APPEND", "LIST_TO_SET_EQ_EMPTY", "FINITE_LIST_TO_SET", "LIST_TO_SET_REVERSE", "SET_TO_LIST_EMPTY", "SET_TO_LIST_SING", "ALL_DISTINCT_SET_TO_LIST", "isPREFIX", "isPREFIX_THM", "SNOC", "LENGTH_SNOC", "LAST_SNOC", "FRONT_SNOC", "SNOC_11", "LENGTH_GENLIST", "GENLIST_AUX_compute", "EL_GENLIST", "GENLIST_NUMERALS", "INFINITE_LIST_UNIV", "LENGTH_LUPDATE", "LIST_BIND_THM", "SINGL_LIST_APPLY_L", "SINGL_SINGL_APPLY", "dropWhile_def", "APPEND_11", "MAP2_NIL", "LENGTH_MAP2", "MAP_EQ_NIL", "MAP_EQ_SING", "LENGTH_NIL", "LENGTH_NIL_SYM", "ALL_DISTINCT_REVERSE", "ALL_DISTINCT_FLAT_REVERSE", "isPREFIX_NILR", "LUPDATE_NIL", "SHORTLEX_THM", "SHORTLEX_NIL2", "WF_SHORTLEX", "LLEX_THM", "LLEX_NIL2", "nub_set", "EVERY2_THM", "oHD_thm", "LIST_TO_SET", "LENGTH_MAP", "MEM_APPEND", "SING_HD", "APPEND_eq_NIL", "NULL_APPEND", "FILTER_F", "FILTER_T", "MEM", "LENGTH_ZIP_MIN", "LAST_MAP", "TAKE_EQ_NIL", "MEM_SET_TO_LIST", "MEM_SNOC", "LIST_REL_eq"]

DELSIMPS = ["ALL_DISTINCT.1", "ALL_DISTINCT.2", "ALL_DISTINCT_FLAT_REVERSE.1", "ALL_DISTINCT_REVERSE.1", "ALL_DISTINCT_SET_TO_LIST.1", "APPEND.1", "APPEND.2", "APPEND_11.1", "APPEND_11.2", "APPEND_ASSOC.1", "APPEND_NIL.1", "APPEND_eq_NIL.1", "APPEND_eq_NIL.2", "APPEND_eq_NIL.3", "APPEND_eq_NIL.4", "CONS.1", "CONS_11.1", "CONS_11.2", "CONS_ACYCLIC.1", "CONS_ACYCLIC.2", "CONS_ACYCLIC.3", "CONS_ACYCLIC.4", "DROP_0.1", "DROP_cons.1", "DROP_nil.1", "EL_GENLIST.1", "EL_restricted.1", "EL_restricted.2", "EL_simp_restricted.1", "EL_simp_restricted.2", "EVERY2_THM.1", "EVERY2_THM.2", "EVERY2_THM.3", "EVERY2_THM.4", "EVERY_APPEND.1", "EVERY_DEF.1", "EVERY_DEF.2", "EVERY_SIMP.1", "EXISTS_APPEND.1", "EXISTS_DEF.1", "EXISTS_DEF.2", "EXISTS_SIMP.1", "FILTER.1", "FILTER.2", "FILTER_F.1", "FILTER_T.1", "FINITE_LIST_TO_SET.1", "FLAT.1", "FLAT.2", "FLAT_APPEND.1", "FOLDL.1", "FOLDL.2", "FOLDL2_def.1", "FOLDL2_def.2", "FOLDL2_def.3", "FOLDL_ZIP_SAME.1", "FOLDR.1", "FOLDR.2", "FRONT_CONS.1", "FRONT_CONS.2", "FRONT_CONS_EQ_NIL.1", "FRONT_CONS_EQ_NIL.2", "FRONT_CONS_EQ_NIL.3", "FRONT_SNOC.1", "GENLIST_AUX_compute.1", "GENLIST_AUX_compute.2", "GENLIST_AUX_compute.3", "GENLIST_NUMERALS.1", "GENLIST_NUMERALS.2", "HD.1", "INFINITE_LIST_UNIV.1", "LAST_APPEND_CONS.1", "LAST_CONS.1", "LAST_CONS.2", "LAST_MAP.1", "LAST_SNOC.1", "LENGTH.1", "LENGTH.2", "LENGTH_APPEND.1", "LENGTH_DROP.1", "LENGTH_FRONT_CONS.1", "LENGTH_GENLIST.1", "LENGTH_LUPDATE.1", "LENGTH_MAP.1", "LENGTH_MAP2.1", "LENGTH_NIL.1", "LENGTH_NIL_SYM.1", "LENGTH_REVERSE.1", "LENGTH_SNOC.1", "LENGTH_TAKE.1", "LENGTH_UNZIP.1", "LENGTH_UNZIP.2", "LENGTH_ZIP.1", "LENGTH_ZIP.2", "LENGTH_ZIP_MIN.1", "LIST_BIND_THM.1", "LIST_BIND_THM.2", "LIST_REL_NIL.1", "LIST_REL_NIL.2", "LIST_REL_def.1", "LIST_REL_def.2", "LIST_REL_def.3", "LIST_REL_def.4", "LIST_REL_eq.1", "LIST_TO_SET.1", "LIST_TO_SET.2", "LIST_TO_SET_APPEND.1", "LIST_TO_SET_DEF.1", "LIST_TO_SET_DEF.2", "LIST_TO_SET_EQ_EMPTY.1", "LIST_TO_SET_EQ_EMPTY.2", "LIST_TO_SET_REVERSE.1", "LLEX_NIL2.1", "LLEX_THM.1", "LLEX_THM.2", "LLEX_THM.3", "LLEX_THM.4", "LUPDATE_LENGTH.1", "LUPDATE_NIL.1", "MAP.1", "MAP.2", "MAP2.1", "MAP2.2", "MAP2_DEF.1", "MAP2_DEF.2", "MAP2_DEF.3", "MAP2_NIL.1", "MAP_APPEND.1", "MAP_EQ_NIL.1", "MAP_EQ_NIL.2", "MAP_EQ_SING.1", "MAP_EQ_SING.2", "MAP_ID.1", "MAP_ID.2", "MAP_ZIP_SAME.1", "MEM.1", "MEM.2", "MEM_APPEND.1", "MEM_REVERSE.1", "MEM_SET_TO_LIST.1", "MEM_SNOC.1", "NOT_CONS_NIL.1", "NOT_CONS_NIL.2", "NOT_EVERY.1", "NOT_EXISTS.1", "NOT_NIL_CONS.1", "NOT_NIL_CONS.2", "NULL_APPEND.1", "NULL_DEF.1", "NULL_DEF.2", "REVERSE_11.1", "REVERSE_DEF.1", "REVERSE_DEF.2", "REVERSE_EQ_NIL.1", "REVERSE_EQ_SING.1", "REVERSE_REVERSE.1", "SET_TO_LIST_EMPTY.1", "SET_TO_LIST_SING.1", "SHORTLEX_NIL2.1", "SHORTLEX_THM.1", "SHORTLEX_THM.2", "SHORTLEX_THM.3", "SHORTLEX_THM.4", "SINGL_LIST_APPLY_L.1", "SINGL_SINGL_APPLY.1", "SING_HD.1", "SING_HD.2", "SNOC.1", "SNOC.2", "SNOC_11.1", "SNOC_11.2", "SUM.1", "SUM.2", "TAKE_0.1", "TAKE_DROP.1", "TAKE_EQ_NIL.1", "TAKE_EQ_NIL.2", "TAKE_LENGTH_ID.1", "TAKE_cons.1", "TAKE_nil.1", "TL_DEF.1", "TL_DEF.2", "UNZIP.1", "UNZIP.2", "UNZIP_ZIP.1", "WF_SHORTLEX.1", "ZIP.1", "ZIP.2", "ZIP_UNZIP.1", "dropWhile_def.1", "dropWhile_def.2", "isPREFIX.1", "isPREFIX.2", "isPREFIX_NILR.1", "isPREFIX_THM.1", "isPREFIX_THM.2", "isPREFIX_THM.3", "list_case_def.1", "list_case_def.2", "nub_set.1", "oHD_thm.1", "oHD_thm.2", "FINITE_common_prefixes.1", "IS_PREFIX_APPEND3.1", "IS_PREFIX_APPENDS.1", "IS_PREFIX_REFL.1", "IS_SUFFIX_REFL.1", "LENGTH_FLAT_REPLICATE.1", "LENGTH_REPLICATE.1", "LIST_REL_APPEND_SING.1", "LIST_REL_REVERSE_EQ.1", "NIL_IN_common_prefixes.1", "REPLICATE.1", "REPLICATE.2", "REVERSE_REPLICATE.1", "SUM_REPLICATE.1", "common_prefixes_NONEMPTY.1", "common_prefixes_NONEMPTY.2", "common_prefixes_PAIR.1", "common_prefixes_PAIR.2", "common_prefixes_PAIR.3", "longest_prefix_EMPTY.1", "longest_prefix_SING.1"]

DELSIMPS2 = ["FINITE_common_prefixes.1", "IS_PREFIX_APPEND3.1", "IS_PREFIX_APPENDS.1", "IS_PREFIX_REFL.1", "IS_SUFFIX_REFL.1", "LENGTH_FLAT_REPLICATE.1", "LENGTH_REPLICATE.1", "LIST_REL_APPEND_SING.1", "LIST_REL_REVERSE_EQ.1", "NIL_IN_common_prefixes.1", "REPLICATE.1", "REPLICATE.2", "REVERSE_REPLICATE.1", "SUM_REPLICATE.1", "common_prefixes_NONEMPTY.1", "common_prefixes_NONEMPTY.2", "common_prefixes_PAIR.1", "common_prefixes_PAIR.2", "common_prefixes_PAIR.3", "longest_prefix_EMPTY.1", "longest_prefix_SING.1"]

DELSIMPS3 = ["LIST_EQ_SIMP_CONV"]

# DELSIMPS3 = ["list$list simpl.rewrite: 1.1","list$list simpl.rewrite: 1.2", "list$list simpl.rewrite: 2.1","list$list simpl.rewrite: 2.2", "list$list simpl.rewrite: 3.1","list$list simpl.rewrite: 3.2", "list$list simpl.rewrite: 4.1","list$list simpl.rewrite: 4.2"]

dels = re.sub("\'","\"",str(DELSIMPS))
dels2 = re.sub("\'","\"",str(DELSIMPS2))
dels3 = re.sub("\'","\"",str(DELSIMPS3))

with open("polished_def_dict.json") as f:
    defs = json.load(f)
    # print(list(dictionary.keys()))
# with open("polished_thm_dict_sorted.json") as f:
#     pthms = json.load(f)
with open("database.json") as f:
    database = json.load(f)

if CONTINUE:
    with open("fact_pool.json") as f:
        fact_pool = json.load(f)
else:
    fact_pool = list(defs.keys())

reverse_database = {(value[0], value[1]) : key for key, value in database.items()}

PROVABLES = [value[4] for key, value in database.items() if value[0] == "list" and value[1] in provables]

if not MORE_TACTICS:
    thms_tactic = ["simp", "fs", "metis_tac"]
    thm_tactic = ["irule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac"]
else:
    thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    thm_tactic = ["irule", "drule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac", "EQ_TAC"]

tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic

GOALS = [value[4] for key, value in database.items() if value[3] == "thm" and value[0] in TARGET_THEORIES]

# TEST_GOALS = [GOALS[5]]
SMALL = ["∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c",
             "REVERSE l = [] ⇔ l = []",
             "∀l. l = [] ∨ ∃h t. l = h::t",
             "∀l1 l2 l3. l1 ++ (l2 ++ l3) = l1 ++ l2 ++ l3",
             "∀M M' v f. M = M' ∧ (M' = [] ⇒ v = v') ∧ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
             "l1 ++ l2 = [e] ⇔ l1 = [e] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [e]",
             "LAST (h::t) = if t = [] then h else LAST t",
             "0 = LENGTH l ⇔ l = []",
             "¬SHORTLEX R l []",
             "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃a l. x = a::l ∧ f a l = v'"]

# LARGER = PROVABLES + ["∀l. ZIP (UNZIP l) = l",
#                       "ZIP ([],[]) = [] ∧ ∀x1 l1 x2 l2. ZIP (x1::l1,x2::l2) = (x1,x2)::ZIP (l1,l2)",
#                       "∀l1 l2. LENGTH l1 = LENGTH l2 ⇒ UNZIP (ZIP (l1,l2)) = (l1,l2)",
#                       "UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))",
#                       "∀f n. TL (GENLIST f (SUC n)) = GENLIST (f ∘ SUC) n",
#                       "∀m n. TAKE n (TAKE m l) = TAKE (MIN n m) l",
#                       "∀l n. LENGTH l ≤ n ⇒ TAKE n l = l",
#                       "TAKE n (GENLIST f m) = GENLIST f (MIN n m)",
#                       "∀l. ALL_DISTINCT l ⇔ ∀x. MEM x l ⇒ FILTER ($= x) l = [x]",
#                       "∀xs. ALL_DISTINCT (FLAT (REVERSE xs)) ⇔ ALL_DISTINCT (FLAT xs)",
#                       "∀l. ¬NULL l ⇒ HD l::TL l = l",
#                       "∀l1 x l2. l1 ++ SNOC x l2 = SNOC x (l1 ++ l2)",
#                       "∀l n. LENGTH l ≤ n ⇒ DROP n l = []",
#                       "∀ls n. DROP n ls = [] ⇔ n ≥ LENGTH ls",
#                       "∀f n x. x < n ⇒ EL x (GENLIST f n) = f x",
#                       "∀n. EL n l = if n = 0 then HD l else EL (PRE n) (TL l)",
#                       "EL 0 = HD ∧ EL (SUC n) (l::ls) = EL n ls",
#                       "∀n x l. x < n ⇒ EL x (TAKE n l) = EL x l",
#                       "∀R l1 l2. LIST_REL R l1 l2 ⇒ LIST_REL R (REVERSE l1) (REVERSE l2)",
#                       "(∀x. MEM x ls ⇒ R x x) ⇒ LIST_REL R ls ls"]

# PROVABLES = ["REVERSE [] = [] ∧ ∀x l. REVERSE (x::l) = SNOC x (REVERSE l)",
#               "∀l1 l2. REVERSE (l1 ++ l2) = REVERSE l2 ++ REVERSE l1",
#               "REVERSE l = [] ⇔ l = []",
#               "∀f x l. MAP f (SNOC x l) = SNOC (f x) (MAP f l)",
#               "UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))",
#               "∀h1 h2. h1 ≠ h2 ⇒ ∀l1 l2. h1::l1 ≠ h2::l2",
#               "∀n f. NULL (GENLIST f n) ⇔ n = 0",
#               "¬LLEX R l []",
#               "∀P l. EVERY P l ⇔ ¬EXISTS (λx. ¬P x) l",
#               "LIST_BIND [x] f = f x"]

# MAYBE = ["LENGTH (FST ps) = LENGTH (SND ps) ∧ MEM p (ZIP ps) ⇒ MEM (FST p) (FST ps) ∧ MEM (SND p) (SND ps)",
#          "∀l f x. MEM x (MAP f l) ⇔ ∃y. x = f y ∧ MEM y l",
#          "∀x ls n. MEM x (DROP n ls) ⇔ ∃m. m + n < LENGTH ls ∧ x = EL (m + n) ls",
#          "∀ls n. ALL_DISTINCT ls ⇒ ALL_DISTINCT (DROP n ls)",
#          "BIGUNION (IMAGE f (set ls)) ⊆ s ⇔ ∀x. MEM x ls ⇒ f x ⊆ s",
#          "DATATYPE (list [] CONS)",
#          "∀n l. n < LENGTH l ⇒ ∀x. EL n (SNOC x l) = EL n l",
#          "∀P l1 l2. EXISTS P (l1 ++ l2) ⇔ EXISTS P l1 ∨ EXISTS P l2",
#          "∀x l. FRONT (SNOC x l) = l",
#          "∀f l1 l2. INJ f (set l1 ∪ set l2) 𝕌(:β) ⇒ (MAP f l1 = MAP f l2 ⇔ l1 = l2)"]

TEST_GOALS = PROVABLES
# TEST_GOALS = SMALL

random.shuffle(TEST_GOALS)
random.shuffle(GOALS)
TEST = GOALS[:(len(GOALS)//4)]
TRAIN = GOALS[(len(GOALS)//4):]
