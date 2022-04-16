open listTheory
use "helper.sml";

(* DELSIMPS = ["ALL_DISTINCT.1", "ALL_DISTINCT.2", "ALL_DISTINCT_FLAT_REVERSE.1", "ALL_DISTINCT_REVERSE.1", "ALL_DISTINCT_SET_TO_LIST.1", "APPEND.1", "APPEND.2", "APPEND_11.1", "APPEND_11.2", "APPEND_ASSOC.1", "APPEND_NIL.1", "APPEND_eq_NIL.1", "APPEND_eq_NIL.2", "APPEND_eq_NIL.3", "APPEND_eq_NIL.4", "CONS.1", "CONS_11.1", "CONS_11.2", "CONS_ACYCLIC.1", "CONS_ACYCLIC.2", "CONS_ACYCLIC.3", "CONS_ACYCLIC.4", "DROP_0.1", "DROP_cons.1", "DROP_nil.1", "EL_GENLIST.1", "EL_restricted.1", "EL_restricted.2", "EL_simp_restricted.1", "EL_simp_restricted.2", "EVERY2_THM.1", "EVERY2_THM.2", "EVERY2_THM.3", "EVERY2_THM.4", "EVERY_APPEND.1", "EVERY_DEF.1", "EVERY_DEF.2", "EVERY_SIMP.1", "EXISTS_APPEND.1", "EXISTS_DEF.1", "EXISTS_DEF.2", "EXISTS_SIMP.1", "FILTER.1", "FILTER.2", "FILTER_F.1", "FILTER_T.1", "FINITE_LIST_TO_SET.1", "FLAT.1", "FLAT.2", "FLAT_APPEND.1", "FOLDL.1", "FOLDL.2", "FOLDL2_def.1", "FOLDL2_def.2", "FOLDL2_def.3", "FOLDL_ZIP_SAME.1", "FOLDR.1", "FOLDR.2", "FRONT_CONS.1", "FRONT_CONS.2", "FRONT_CONS_EQ_NIL.1", "FRONT_CONS_EQ_NIL.2", "FRONT_CONS_EQ_NIL.3", "FRONT_SNOC.1", "GENLIST_AUX_compute.1", "GENLIST_AUX_compute.2", "GENLIST_AUX_compute.3", "GENLIST_NUMERALS.1", "GENLIST_NUMERALS.2", "HD.1", "INFINITE_LIST_UNIV.1", "LAST_APPEND_CONS.1", "LAST_CONS.1", "LAST_CONS.2", "LAST_MAP.1", "LAST_SNOC.1", "LENGTH.1", "LENGTH.2", "LENGTH_APPEND.1", "LENGTH_DROP.1", "LENGTH_FRONT_CONS.1", "LENGTH_GENLIST.1", "LENGTH_LUPDATE.1", "LENGTH_MAP.1", "LENGTH_MAP2.1", "LENGTH_NIL.1", "LENGTH_NIL_SYM.1", "LENGTH_REVERSE.1", "LENGTH_SNOC.1", "LENGTH_TAKE.1", "LENGTH_UNZIP.1", "LENGTH_UNZIP.2", "LENGTH_ZIP.1", "LENGTH_ZIP.2", "LENGTH_ZIP_MIN.1", "LIST_BIND_THM.1", "LIST_BIND_THM.2", "LIST_REL_NIL.1", "LIST_REL_NIL.2", "LIST_REL_def.1", "LIST_REL_def.2", "LIST_REL_def.3", "LIST_REL_def.4", "LIST_REL_eq.1", "LIST_TO_SET.1", "LIST_TO_SET.2", "LIST_TO_SET_APPEND.1", "LIST_TO_SET_DEF.1", "LIST_TO_SET_DEF.2", "LIST_TO_SET_EQ_EMPTY.1", "LIST_TO_SET_EQ_EMPTY.2", "LIST_TO_SET_REVERSE.1", "LLEX_NIL2.1", "LLEX_THM.1", "LLEX_THM.2", "LLEX_THM.3", "LLEX_THM.4", "LUPDATE_LENGTH.1", "LUPDATE_NIL.1", "MAP.1", "MAP.2", "MAP2.1", "MAP2.2", "MAP2_DEF.1", "MAP2_DEF.2", "MAP2_DEF.3", "MAP2_NIL.1", "MAP_APPEND.1", "MAP_EQ_NIL.1", "MAP_EQ_NIL.2", "MAP_EQ_SING.1", "MAP_EQ_SING.2", "MAP_ID.1", "MAP_ID.2", "MAP_ZIP_SAME.1", "MEM.1", "MEM.2", "MEM_APPEND.1", "MEM_REVERSE.1", "MEM_SET_TO_LIST.1", "MEM_SNOC.1", "NOT_CONS_NIL.1", "NOT_CONS_NIL.2", "NOT_EVERY.1", "NOT_EXISTS.1", "NOT_NIL_CONS.1", "NOT_NIL_CONS.2", "NULL_APPEND.1", "NULL_DEF.1", "NULL_DEF.2", "REVERSE_11.1", "REVERSE_DEF.1", "REVERSE_DEF.2", "REVERSE_EQ_NIL.1", "REVERSE_EQ_SING.1", "REVERSE_REVERSE.1", "SET_TO_LIST_EMPTY.1", "SET_TO_LIST_SING.1", "SHORTLEX_NIL2.1", "SHORTLEX_THM.1", "SHORTLEX_THM.2", "SHORTLEX_THM.3", "SHORTLEX_THM.4", "SINGL_LIST_APPLY_L.1", "SINGL_SINGL_APPLY.1", "SING_HD.1", "SING_HD.2", "SNOC.1", "SNOC.2", "SNOC_11.1", "SNOC_11.2", "SUM.1", "SUM.2", "TAKE_0.1", "TAKE_DROP.1", "TAKE_EQ_NIL.1", "TAKE_EQ_NIL.2", "TAKE_LENGTH_ID.1", "TAKE_cons.1", "TAKE_nil.1", "TL_DEF.1", "TL_DEF.2", "UNZIP.1", "UNZIP.2", "UNZIP_ZIP.1", "WF_SHORTLEX.1", "ZIP.1", "ZIP.2", "ZIP_UNZIP.1", "dropWhile_def.1", "dropWhile_def.2", "isPREFIX.1", "isPREFIX.2", "isPREFIX_NILR.1", "isPREFIX_THM.1", "isPREFIX_THM.2", "isPREFIX_THM.3", "list_case_def.1", "list_case_def.2", "nub_set.1", "oHD_thm.1", "oHD_thm.2", "FINITE_common_prefixes.1", "IS_PREFIX_APPEND3.1", "IS_PREFIX_APPENDS.1", "IS_PREFIX_REFL.1", "IS_SUFFIX_REFL.1", "LENGTH_FLAT_REPLICATE.1", "LENGTH_REPLICATE.1", "LIST_REL_APPEND_SING.1", "LIST_REL_REVERSE_EQ.1", "NIL_IN_common_prefixes.1", "REPLICATE.1", "REPLICATE.2", "REVERSE_REPLICATE.1", "SUM_REPLICATE.1", "common_prefixes_NONEMPTY.1", "common_prefixes_NONEMPTY.2", "common_prefixes_PAIR.1", "common_prefixes_PAIR.2", "common_prefixes_PAIR.3", "longest_prefix_EMPTY.1", "longest_prefix_SING.1"] *)

[('simp[LAST_DEF, LEN_DEF, LEN_DEF, LAST_DEF, FLAT]', 'LIST_REL $= = $=')]

(* Proof trace: [('fs[LRC_def, nub_def, LIST_BIND_def, LIST_TO_SET_DEF, SUM]', '∀l m. m = LENGTH l ⇒ TAKE m l = l'), ('Induct_on `l`', '∀l. TAKE (LENGTH l) l = l'), ('simp[EXISTS_DEF, oEL_def, TAKE_0, REV_DEF, TAKE_EQ_NIL]', 'TAKE (LENGTH []) [] = []'), ('rpt strip_tac >> simp[LEN_DEF, PAD_RIGHT, LENGTH, REVERSE_DEF, nub_def]', '(TAKE (LENGTH l) l = l) ==> (∀h. TAKE (LENGTH (h::l)) (h::l) = h::l)'), ('rpt strip_tac >> simp[TAKE_def, SHORTLEX_NIL2, EL, NULL_DEF, EVERYi_def]', '(TAKE (LENGTH l) l = l) ==> (TAKE (SUC (LENGTH l)) (h::l) = h::l)')] *)
(* Time: 2.296069447998889   *)

(* Game: 471 *)
(* Facts: 64 *)
(* Initialization done. Main goal is: *)
(* ∀s. FINITE s ⇒ set (SET_TO_LIST s) = s. *)
(* Count: 472 *)
(* Failed. *)
(* Rewards: [-2, 0, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -2, -2, -2, -1, -2, -2, -1, -2, -2] *)
(* Tactics: ['simp', 'Induct_on', 'simp', 'fs', 'Induct_on', 'Induct_on', 'simp', 'Induct_on', 'simp', 'simp', 'simp', 'metis_tac', 'Induct_on', 'strip_tac', 'strip_tac', 'metis_tac', 'fs', 'simp', 'metis_tac', 'simp', 'simp'] *)
(* Total: -37 *)
(* Time: 1.4448557710020395   *)
(* 2865008 *)
(* Induct args: ['Vs', 'Vs', 'Ve', 'Ve', 'Ve'] *)
(* Preferences: tensor([[0.4161, 0.1674, 0.0709, 0.1831, 0.0661, 0.0965]], device='cuda:0') *)
(* Proved so far: 32 *)
Theorem ex23 : ∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c
Proof
Induct_on `l`
strip_tac
fs[list_size_def, list_TY_DEF, EXISTS_DEF, oEL_def, LIST_GUARD_def]
fs[ALL_DISTINCT, LIST_IGNORE_BIND_def, EXISTS_DEF, EL, nub_def]
metis_tac[list_case_def, PAD_RIGHT, LAST_DEF, UNZIP, FILTER]
QED

Theorem ex23 : ¬SHORTLEX R l []
Proof
Induct_on `l`
fs[oEL_def, SHORTLEX_THM, REVERSE_SNOC_DEF, LIST_BIND_ID, EXISTS_SIMP]
rpt strip_tac >> irule NOT_CONS_NIL
fs[TAKE_def, list_case_cong, SHORTLEX_THM, MAP_EQ_NIL,MAP_ID]
QED

Theorem ex23 : ∀M M' v f. M = M' ∧ (M' = [] ⇒ v = v') ∧ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'
Proof
simp[EL_restricted, LENGTH, LUPDATE_def, LAST_DEF, FRONT_DEF]

Induct_on `M'`

fs[APPEND_EQ_SELF, LRC_def, LIST_BIND_def, LAST_compute, GENLIST_CONS]

rpt strip_tac >> metis_tac[ALL_DISTINCT, list_case_eq, TAKE_LENGTH_ID_rwt, list_case_compute, MAP_EQ_NIL]
QED

Theorem ex22 : ∀x l. SNOC x l = l ++ [x]
Proof
simp[]
QED

Theorem ex21 : ∀h t. TL (h::t) = t
Proof
strip_tac
strip_tac
fs[LEN_DEF, HD, TL_DEF, LIST_LIFT2_def, LIST_TO_SET_DEF]

QED

Theorem ex20 : UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))
Proof
strip_tac
metis_tac[PAD_LEFT, PAD_RIGHT, UNZIP, OPT_MMAP_def, FILTER]
fs[LIST_IGNORE_BIND_def, oHD_def, nub_def, UNZIP, oHD_def]
Induct_on `t`
fs[UNZIP, INDEX_FIND_def, TL_DEF, LIST_APPLY_def, EVERYi_def]
rpt strip_tac >> simp[PAD_LEFT, EXISTS_DEF, DROP_def, nub_def, PAD_RIGHT]
fs[list_size_def, FOLDR, UNZIP, EL, FIND_def]

QED

Theorem ex19 : ZIP ([],[]) = [] ∧ ∀x1 l1 x2 l2. ZIP (x1::l1,x2::l2) = (x1,x2)::ZIP (l1,l2)
Proof
strip_tac
fs[LLEX_def, LIST_APPLY_def, SHORTLEX_def, ZIP_def, LUPDATE_def]
strip_tac
strip_tac
strip_tac
strip_tac
simp[ZIP_def, LEN_DEF, oHD_def, PAD_LEFT, LIST_IGNORE_BIND_def]
QED

Theorem ex18 : ∀x y a b. SNOC x y = SNOC a b ⇔ x = a ∧ y = b
Proof
fs[]
QED


Theorem ex18 : REVERSE l = [] ⇔ l = []
Proof
Induct_on `l`
simp[]
simp[]
QED

Theorem ex17 : REVERSE (REVERSE []) = []
Proof
simp[]
QED

Theorem ex16 : !l:'a list. REVERSE (REVERSE l) = l
Proof
simp[]
QED

Theorem ex15 : REVERSE (l1 ⧺ l2) = REVERSE l2 ⧺ REVERSE l1
Proof
simp[]
QED

Theorem ex14444 : ¬EXISTS P l ⇔ EVERY ($~ ∘ P) l
Proof
simp[OPT_MMAP_def, MAP, splitAtPki_def, oEL_def, LIST_LIFT2_def]
QED

Theorem ex13 : TAKE n [] = []
Proof
simp[]
QED

Theorem ex12 : LIST_REL R (h::t) xs ⇔ ∃h' t'. xs = h'::t' ∧ R h h' ∧ LIST_REL R t t'
Proof
simp[]
QED

Theorem ex11 : FINITE s ⇒ ∀x. MEM x (SET_TO_LIST s) ⇔ x ∈ s
Proof
Induct_on `pred_set$FINITE`
QED

Theorem ex10 : ∀xs ys. LENGTH (MAP2 f xs ys) = MIN (LENGTH xs) (LENGTH ys)
Proof
simp[LIST_TO_SET_DEF, LUPDATE_def, APPEND, ZIP_def, FLAT]
QED

Theorem ex9 : ([] ≼ l ⇔ T) ∧ (h::t ≼ [] ⇔ F) ∧ (h1::t1 ≼ h2::t2 ⇔ h1 = h2 ∧ t1 ≼ t2)
Proof
fs[]
fs[oEL_def, DROP_def, NULL_DEF, TL_DEF, MAP]

fs[UNIQUE_DEF, isPREFIX, FLAT, LEN_DEF, FRONT_DEF]
QED

Theorem ex8 : ∀s. FINITE s ⇒ ∀x. x ∈ s ⇔ MEM x (SET_TO_LIST s)
Proof
fs[LIST_GUARD_def, list_case_def, list_TY_DEF, REV_DEF, SNOC]
QED

Theorem ex7 : EL 0 = HD ∧ EL (SUC n) (l::ls) = EL n ls
Proof
simp[nub_def, LIST_TO_SET_DEF, INDEX_OF_def, EVERYi_def, TAKE_def]
QED

Theorem pb2 : ∀l2 P. LENGTH l1 = LENGTH l2 ⇒ (EVERY (λx. P (SND x)) (ZIP (l1,l2)) ⇔ EVERY P l2)
Proof
metis_tac[PAD_LEFT, EL, isPREFIX, GENLIST, LAST_DEF]

metis_tac[GENLIST, LAST_DEF]

QED

Theorem ex6 : ∀ls f e. FOLDL f e (ZIP (ls,ls)) = FOLDL (λx y. f x (y,y)) e ls
Proof
simp[LIST_LIFT2_def, LIST_LIFT2_def, LENGTH, list_size_def, dropWhile_def]
simp[]
QED


Theorem ex6 : ∀x y. LIST_REL R x y ⇒ LENGTH x = LENGTH y
Proof
Induct_on `x`
strip_tac
simp[FRONT_DEF, FRONT_DEF, OPT_MMAP_def, LIST_APPLY_def, FILTER]
rpt strip_tac >> simp[SUM, LIST_IGNORE_BIND_def, NULL_DEF, LIST_LIFT2_def, LENGTH]
fs[splitAtPki_def, SUM, LEN_DEF, LIST_IGNORE_BIND_def, EVERY_DEF]
QED

Theorem pb1 : (v = [] ⇒ ∃!fn1. fn1 [] = x ∧ ∀h t. fn1 (h::t) = f (fn1 t) h t) ==> (∀h. h::v = [] ⇒ ∃!fn1. fn1 [] = x ∧ ∀h t. fn1 (h::t) = f (fn1 t) h t)
Proof
rpt strip_tac >> metis_tac[HD, FOLDR, SNOC, FOLDL, EL]
QED

Theorem pb : x ≼ [] ⇔ x = []
Proof
simp[]
QED

Theorem ex6 : ∀P. P [] ∧ (∀l. P l ⇒ ∀a. P (a::l)) ⇒ ∀l. P l
Proof
strip_tac
strip_tac
simp[INDEX_OF_def, INDEX_OF_def, HD, list_case_def, LEN_DEF]

Induct_on `l`

simp[GENLIST, EXISTS_DEF, LAST_DEF, PAD_LEFT, ALL_DISTINCT]

simp[LIST_GUARD_def, EXISTS_DEF, LIST_IGNORE_BIND_def, TAKE_def, list_size_def]
QED

Theorem ex5 : ∀l n. LENGTH l ≤ n ⇒ DROP n l = []
Proof
strip_tac

Induct_on `l`

fs[LIST_APPLY_def, LEN_DEF, oHD_def, LIST_BIND_def, REVERSE_DEF, LRC_def, LENGTH, list_case_def, LIST_IGNORE_BIND_def, LLEX_def]

fs[LIST_APPLY_def, FOLDR, TL_DEF, PAD_RIGHT, GENLIST, list_case_def, LIST_TO_SET_DEF, FILTER, TAKE_def, LENGTH]
QED

Theorem ex4 : ∀v l1 x l2 l3. LUPDATE v (LENGTH l1) (l1 ++ [x] ++ l2) = l1 ++ [v] ++ l2
Proof
fs[NULL_DEF, splitAtPki_def, SHORTLEX_def, INDEX_FIND_def, SUM, APPEND, FILTER, PAD_LEFT, GENLIST, FLAT]

strip_tac

Induct_on `l1`

fs[list_TY_DEF, LIST_LIFT2_def, NULL_DEF, EVERY_DEF, EXISTS_DEF, ALL_DISTINCT, oEL_def, APPEND, LIST_IGNORE_BIND_def, FIND_def]

fs[LUPDATE_def, REV_DEF, GENLIST, SHORTLEX_def, FIND_def, HD, SUM, UNZIP, PAD_LEFT, nub_def]

fs[PAD_LEFT, SNOC, EVERYi_def, LIST_GUARD_def, LENGTH, SUM, INDEX_OF_def, list_case_def, LUPDATE_def, ZIP_def]
QED

Theorem exxx : ∀x l. ALL_DISTINCT (SNOC x l) ⇔ ¬MEM x l ∧ ALL_DISTINCT l
Proof
Induct_on `l`

simp[LEN_DEF, LEN_DEF, DROP_def, EVERYi_def, LLEX_def, DROP_def, UNZIP, PAD_RIGHT, FOLDR, isPREFIX]

simp[SNOC, oEL_def, DROP_def, nub_def, LIST_BIND_def, LIST_GUARD_def, SNOC, SNOC, FIND_def, isPREFIX]

fs[nub_def, list_TY_DEF, LUPDATE_def, ZIP_def, EXISTS_DEF, INDEX_OF_def, FIND_def, FOLDL, DROP_def, LAST_DEF]

metis_tac[PAD_RIGHT, REVERSE_DEF, LIST_IGNORE_BIND_def, EL, REVERSE_DEF, INDEX_FIND_def, FOLDL, OPT_MMAP_def, OPT_MMAP_def, LEN_DEF]
QED

Theorem exx : ∀xs x y ys. LUPDATE x (LENGTH xs) (xs ++ y::ys) = xs ++ x::ys
Proof
fs[FILTER, LRC_def, APPEND, REV_DEF, list_case_def, splitAtPki_def, LRC_def, FRONT_DEF, LIST_TO_SET_DEF, dropWhile_def]

Induct_on `ys`

simp[FIND_def, REV_DEF, PAD_RIGHT, list_case_def, LENGTH, FRONT_DEF, PAD_RIGHT, FIND_def, INDEX_FIND_def, SUM_ACC_DEF]

simp[GENLIST_AUX, INDEX_OF_def, SUM_ACC_DEF, FLAT, isPREFIX, TL_DEF, FLAT, LLEX_def, PAD_LEFT, EL]

simp[LEN_DEF, list_TY_DEF, REVERSE_DEF, NULL_DEF, LIST_BIND_def, SHORTLEX_def, ALL_DISTINCT, list_size_def, LENGTH, LIST_LIFT2_def]

Induct_on `xs`

simp[PAD_RIGHT, LIST_BIND_def, list_size_def, TAKE_def, FILTER, REV_DEF, nub_def, SUM_ACC_DEF, FOLDR, dropWhile_def]

fs[oHD_def, LIST_GUARD_def, nub_def, APPEND, GENLIST_AUX, LENGTH, EVERY_DEF, LIST_IGNORE_BIND_def, OPT_MMAP_def, PAD_RIGHT]

simp[SNOC, LEN_DEF, FOLDL, TAKE_def, PAD_RIGHT, TL_DEF, FLAT, dropWhile_def, SUM_ACC_DEF, LUPDATE_def]

fs[PAD_RIGHT, SHORTLEX_def, INDEX_FIND_def, LUPDATE_def, FOLDL, SET_TO_LIST_primitive_def, EL, LEN_DEF, FILTER, SUM_ACC_DEF]
QED

Theorem dc : (∀P ys. LIST_REL P [] ys ⇔ ys = []) ∧ (∀P yys x xs. LIST_REL P (x::xs) yys ⇔ ∃y ys. yys = y::ys ∧ P x y ∧ LIST_REL P xs ys) ∧ (∀P xs. LIST_REL P xs [] ⇔ xs = []) ∧ ∀P xxs y ys. LIST_REL P xxs (y::ys) ⇔ ∃x xs. xxs = x::xs ∧ P x y ∧ LIST_REL P xs ys
Proof
metis_tac[ALL_DISTINCT, LIST_BIND_def, oHD_def, EXISTS_DEF, UNIQUE_DEF, NULL_DEF, HD, FOLDR, SHORTLEX_def, UNZIP]
ED

Theorem eb : oHD (h::t) = SOME h
Proof
fs[listTheory.REV_DEF]
QED

∀ls f e. FOLDL f e (ZIP (ls,ls)) = FOLDL (λx y. f x (y,y)) e ls.
buffer (last 100 chars): b'ssage = "no goals", origin_function = "top_goals", origin_structure =\r\n      "goalStack"} raised\r\n> '
before (last 100 chars): b'ssage = "no goals", origin_function = "top_goals", origin_structure =\r\n      "goalStack"} raised\r\n> '


Initialization done. Main goal is:
∀n. GENLIST f n = GENLIST_AUX f n [].

Initialization done. Main goal is:
∀xs x y ys. LUPDATE x (LENGTH xs) (xs ++ y::ys) = xs ++ x::ys.
Initialization done. Main goal is:
∀l P. FILTER P (REVERSE l) = REVERSE (FILTER P l).
tac is: fs[oEL_def, GENLIST_AUX, dropWhile_def, UNIQUE_DEF, UNZIP, EXISTS_DEF, FRONT_DEF, list_case_def, FOLDL, GENLIST]
content:  @ Cbool$! | Vl @ Cbool$! | VP @ @ Cmin$= @ @ Clist$FILTER VP @ Clist$REVERSE Vl @ Clist$REVERSE @ @ Clist$FILTER VP Vl  : proof
> top_goals();
val it = [([], "@ Cbool$! | Vl @ Cbool$! | VP @ @ Cmin$= @ @ Clist$FILTER VP @ Clist$REVERSE Vl @ Clist$REVERSE @ @ Clist$FILTER VP Vl")]

Initialization done. Main goal is:
∀ls f. (∀x y. MEM x ls ∧ MEM y ls ∧ f x = f y ⇒ x = y) ∧ ALL_DISTINCT ls ⇒ ALL_DISTINCT (MAP f ls).
Initialization done. Main goal is:
∀l1 l2. nub (l1 ++ l2) = nub (FILTER (λx. ¬MEM x l2) l1) ++ nub l2.
tac is: simp[ALL_DISTINCT, ALL_DISTINCT, TL_DEF, SUM, SUM, LEN_DEF, EXISTS_DEF, SHORTLEX_def, LLEX_def, OPT_MMAP_def]
content:  @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ @ Cmin$= @ Clist$nub @ @ Clist$APPEND Vl1 Vl2 @ @ Clist$APPEND @ Clist$nub @ @ Clist$FILTER | Vx @ Cbool$~ @ @ Cbool$IN Vx @ Clist$LIST_TO_SET Vl2 Vl1 @ Clist$nub Vl2  : proof
> top_goals();
val it = [([], "@ Cbool$! | Vl1 @ Cbool$! | Vl2 @ @ Cmin$= @ Clist$nub @ @ Clist$APPEND Vl1 Vl2 @ @ Clist$APPEND @ Clist$nub @ @ Clist$FILTER | Vx @ Cbool$~ @ @ Cbool$IN Vx @ Clist$LIST_TO_SET Vl2 Vl1 @ Clist$nub Vl2")]

Initialization done. Main goal is:
∀ls f. MAP f (ZIP (ls,ls)) = MAP (λx. f (x,x)) ls.
tac is: strip_tac
content:  @ Cbool$! | Vf @ @ Cmin$= @ @ Clist$MAP Vf @ Clist$ZIP @ @ Cpair$, Vls Vls @ @ Clist$MAP | Vx @ Vf @ @ Cpair$, Vx Vx Vls  : proof
> top_goals();
val it = [([], "@ Cbool$! | Vf @ @ Cmin$= @ @ Clist$MAP Vf @ Clist$ZIP @ @ Cpair$, Vls Vls @ @ Clist$MAP | Vx @ Vf @ @ Cpair$, Vx Vx Vls")]

Initialization done. Main goal is:
∀x l. FRONT (SNOC x l) = l.
tac is: simp[GENLIST_AUX, MAP, FRONT_DEF, MAP, LIST_IGNORE_BIND_def, REV_DEF, FIND_def, PAD_RIGHT, TAKE_def, EXISTS_DEF]
content:  @ Cbool$! | Vx @ Cbool$! | Vl @ @ Cmin$= @ Clist$FRONT @ @ Clist$APPEND Vl @ @ Clist$CONS Vx Clist$NIL Vl  : proof
> top_goals();
val it = [([], "@ Cbool$! | Vx @ Cbool$! | Vl @ @ Cmin$= @ Clist$FRONT @ @ Clist$APPEND Vl @ @ Clist$CONS Vx Clist$NIL Vl")]

Initialization done. Main goal is:
∀n l. n < LENGTH l ⇒ EL n (REVERSE l) = EL (PRE (LENGTH l − n)) l.
Initialization done. Main goal is:
transitive R ⇒ transitive (LLEX R).
tac is: strip_tac
content:  0.  @ Crelation$transitive VR ------------------------------------ @ Crelation$transitive @ Clist$LLEX VR  : proof
> top_goals();
val it = [(["@ Crelation$transitive VR"], "@ Crelation$transitive @ Clist$LLEX VR")]


∀l1 l2 f. LIST_REL f l1 l2 ⇔ LENGTH l1 = LENGTH l2 ∧ EVERY (UNCURRY f) (ZIP (l1,l2))

Theorem ea : transitive R ⇒ transitive (LLEX R)
Proof
strip_tac
QED

Theorem ee : ∀x xs. LENGTH (FRONT (x::xs)) = LENGTH xs
Proof

QED

Theorem ex : (∀f cs c bs b a. FOLDL2 f a (b::bs) (c::cs) = FOLDL2 f (f a b c) bs cs) ∧ (∀f cs a. FOLDL2 f a [] cs = a) ∧ ∀v7 v6 f a. FOLDL2 f a (v6::v7) [] = a
Proof
(* works without arguments *)
strip_tac
strip_tac
fs[GENLIST_AUX]
strip_tac
simp[TL_DEF]
fs[EVERYi_def]
QED

Theorem ex2 : ∀a0 a1 a0' a1'. a0::a1 = a0'::a1' ⇔ a0 = a0' ∧ a1 = a1'
Proof
Induct_on `ab`
simp[]
QED

Theorem ex3 : ∀n l. LENGTH (DROP n l) = LENGTH l − n
Proof
simp[]
QED

Theorem frc : LIST_REL R1 l1 l2 ==> ∀x y. R1 x y ⇒ R2 x y ==> LIST_REL R2 l1 l2
Proof
metis_tac[listTheory.SUM]
QED


Theorem foo : !n m. ~(2*n = 2*m + 1)
Proof
strip_tac
rw[]
strip_tac
simp[]
Induct_on `n`
rw[]
Induct_on `m`
(rpt strip_tac) >> Induct_on `m`
strip_tac
Induct_on `m`

rw[]
rw[]
QED

Theorem bar : 2 * n ≠ 2 * m + 1 ⇒ 2 * n = 2 * SUC m + 1 ⇒ F==>2 * SUC n ≠ 2 * m + 1 ⇒ 2 * SUC n = 2 * SUC m + 1 ⇒ F
Proof
rpt strip_tac
QED

(* [('strip_tac', '!n m. ~(2*n = 2*m + 1)'), ('rw[]', '∀m. 2 * n ≠ 2 * m + 1'), ('strip_tac', '2 * n ≠ 2 * m + 1'), ('rpt strip_tac >> simp[]', '2 * n = 2 * m + 1==>F'), ('rpt strip_tac >> Induct_on `n`', '2 * n = 2 * m + 1==>F'), ('rw[]', '2 * 0 = 2 * m + 1 ⇒ F'), ('rpt strip_tac >> Induct_on `m`', '2 * n = 2 * m + 1 ⇒ F==>2 * SUC n = 2 * m + 1 ⇒ F')] *)

Theorem faz : (MAP (\x. x) l = l) /\ (MAP I l = l)
Proof
(* simp[Excl "MAP_ID"] *)
(* strip_tac *)
(* simp[listTheory.MAP] *)
QED
