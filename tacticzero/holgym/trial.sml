open listTheory
open probabilityTheory
use "helper.sml";

val frags = diminish_srw_ss ["arithmetic", "list"];
val _ = augment_srw_ss ["list"];

delsimps ["ALL_DISTINCT.1", "ALL_DISTINCT.2", "ALL_DISTINCT_FLAT_REVERSE.1", "ALL_DISTINCT_REVERSE.1", "ALL_DISTINCT_SET_TO_LIST.1", "APPEND.1", "APPEND.2", "APPEND_11.1", "APPEND_11.2", "APPEND_ASSOC.1", "APPEND_NIL.1", "APPEND_eq_NIL.1", "APPEND_eq_NIL.2", "APPEND_eq_NIL.3", "APPEND_eq_NIL.4", "CONS.1", "CONS_11.1", "CONS_11.2", "CONS_ACYCLIC.1", "CONS_ACYCLIC.2", "CONS_ACYCLIC.3", "CONS_ACYCLIC.4", "DROP_0.1", "DROP_cons.1", "DROP_nil.1", "EL_GENLIST.1", "EL_restricted.1", "EL_restricted.2", "EL_simp_restricted.1", "EL_simp_restricted.2", "EVERY2_THM.1", "EVERY2_THM.2", "EVERY2_THM.3", "EVERY2_THM.4", "EVERY_APPEND.1", "EVERY_DEF.1", "EVERY_DEF.2", "EVERY_SIMP.1", "EXISTS_APPEND.1", "EXISTS_DEF.1", "EXISTS_DEF.2", "EXISTS_SIMP.1", "FILTER.1", "FILTER.2", "FILTER_F.1", "FILTER_T.1", "FINITE_LIST_TO_SET.1", "FLAT.1", "FLAT.2", "FLAT_APPEND.1", "FOLDL.1", "FOLDL.2", "FOLDL2_def.1", "FOLDL2_def.2", "FOLDL2_def.3", "FOLDL_ZIP_SAME.1", "FOLDR.1", "FOLDR.2", "FRONT_CONS.1", "FRONT_CONS.2", "FRONT_CONS_EQ_NIL.1", "FRONT_CONS_EQ_NIL.2", "FRONT_CONS_EQ_NIL.3", "FRONT_SNOC.1", "GENLIST_AUX_compute.1", "GENLIST_AUX_compute.2", "GENLIST_AUX_compute.3", "GENLIST_NUMERALS.1", "GENLIST_NUMERALS.2", "HD.1", "INFINITE_LIST_UNIV.1", "LAST_APPEND_CONS.1", "LAST_CONS.1", "LAST_CONS.2", "LAST_MAP.1", "LAST_SNOC.1", "LENGTH.1", "LENGTH.2", "LENGTH_APPEND.1", "LENGTH_DROP.1", "LENGTH_FRONT_CONS.1", "LENGTH_GENLIST.1", "LENGTH_LUPDATE.1", "LENGTH_MAP.1", "LENGTH_MAP2.1", "LENGTH_NIL.1", "LENGTH_NIL_SYM.1", "LENGTH_REVERSE.1", "LENGTH_SNOC.1", "LENGTH_TAKE.1", "LENGTH_UNZIP.1", "LENGTH_UNZIP.2", "LENGTH_ZIP.1", "LENGTH_ZIP.2", "LENGTH_ZIP_MIN.1", "LIST_BIND_THM.1", "LIST_BIND_THM.2", "LIST_REL_NIL.1", "LIST_REL_NIL.2", "LIST_REL_def.1", "LIST_REL_def.2", "LIST_REL_def.3", "LIST_REL_def.4", "LIST_REL_eq.1", "LIST_TO_SET.1", "LIST_TO_SET.2", "LIST_TO_SET_APPEND.1", "LIST_TO_SET_DEF.1", "LIST_TO_SET_DEF.2", "LIST_TO_SET_EQ_EMPTY.1", "LIST_TO_SET_EQ_EMPTY.2", "LIST_TO_SET_REVERSE.1", "LLEX_NIL2.1", "LLEX_THM.1", "LLEX_THM.2", "LLEX_THM.3", "LLEX_THM.4", "LUPDATE_LENGTH.1", "LUPDATE_NIL.1", "MAP.1", "MAP.2", "MAP2.1", "MAP2.2", "MAP2_DEF.1", "MAP2_DEF.2", "MAP2_DEF.3", "MAP2_NIL.1", "MAP_APPEND.1", "MAP_EQ_NIL.1", "MAP_EQ_NIL.2", "MAP_EQ_SING.1", "MAP_EQ_SING.2", "MAP_ID.1", "MAP_ID.2", "MAP_ZIP_SAME.1", "MEM.1", "MEM.2", "MEM_APPEND.1", "MEM_REVERSE.1", "MEM_SET_TO_LIST.1", "MEM_SNOC.1", "NOT_CONS_NIL.1", "NOT_CONS_NIL.2", "NOT_EVERY.1", "NOT_EXISTS.1", "NOT_NIL_CONS.1", "NOT_NIL_CONS.2", "NULL_APPEND.1", "NULL_DEF.1", "NULL_DEF.2", "REVERSE_11.1", "REVERSE_DEF.1", "REVERSE_DEF.2", "REVERSE_EQ_NIL.1", "REVERSE_EQ_SING.1", "REVERSE_REVERSE.1", "SET_TO_LIST_EMPTY.1", "SET_TO_LIST_SING.1", "SHORTLEX_NIL2.1", "SHORTLEX_THM.1", "SHORTLEX_THM.2", "SHORTLEX_THM.3", "SHORTLEX_THM.4", "SINGL_LIST_APPLY_L.1", "SINGL_SINGL_APPLY.1", "SING_HD.1", "SING_HD.2", "SNOC.1", "SNOC.2", "SNOC_11.1", "SNOC_11.2", "SUM.1", "SUM.2", "TAKE_0.1", "TAKE_DROP.1", "TAKE_EQ_NIL.1", "TAKE_EQ_NIL.2", "TAKE_LENGTH_ID.1", "TAKE_cons.1", "TAKE_nil.1", "TL_DEF.1", "TL_DEF.2", "UNZIP.1", "UNZIP.2", "UNZIP_ZIP.1", "WF_SHORTLEX.1", "ZIP.1", "ZIP.2", "ZIP_UNZIP.1", "dropWhile_def.1", "dropWhile_def.2", "isPREFIX.1", "isPREFIX.2", "isPREFIX_NILR.1", "isPREFIX_THM.1", "isPREFIX_THM.2", "isPREFIX_THM.3", "list_case_def.1", "list_case_def.2", "nub_set.1", "oHD_thm.1", "oHD_thm.2", "FINITE_common_prefixes.1", "IS_PREFIX_APPEND3.1", "IS_PREFIX_APPENDS.1", "IS_PREFIX_REFL.1", "IS_SUFFIX_REFL.1", "LENGTH_FLAT_REPLICATE.1", "LENGTH_REPLICATE.1", "LIST_REL_APPEND_SING.1", "LIST_REL_REVERSE_EQ.1", "NIL_IN_common_prefixes.1", "REPLICATE.1", "REPLICATE.2", "REVERSE_REPLICATE.1", "SUM_REPLICATE.1", "common_prefixes_NONEMPTY.1", "common_prefixes_NONEMPTY.2", "common_prefixes_PAIR.1", "common_prefixes_PAIR.2", "common_prefixes_PAIR.3", "longest_prefix_EMPTY.1", "longest_prefix_SING.1"];

Theorem shortlex_validation:
FINITE (∅ :α -> bool)
Proof
fs[pred_setTheory.FINITE_DEF, pred_setTheory.DIFF_DEF, pred_setTheory.SUBSET_applied, pred_setTheory.BIJ_IMP_11, pred_setTheory.INJ_INSERT]
QED

Theorem DBG: 
(s :real -> bool) ⊆ (t :real -> bool) ∧ subspace t ⇒ span s ⊆ t
Proof
fs[pred_setTheory.DISJOINT_INSERT, pred_setTheory.FORALL_IN_INSERT, boolTheory.IMP_DISJ_THM, pred_setTheory.UNION_SUBSET, pred_setTheory.FINITE_INDUCT]
QED

Theorem debugp1:
WF ($<< :α -> α -> bool) ⇔ ¬∃(s :num -> α). ∀(n :num). s (SUC n) ≪ s n
Proof

QED

Theorem REAL_LE_INV_EQ[simp]:
  !x. 0 <= inv x <=> 0 <= x
Proof
  REWRITE_TAC[REAL_LE_LT, REAL_LT_INV_EQ, REAL_INV_EQ_0] THEN
  MESON_TAC[REAL_INV_EQ_0]
QED

Theorem IMAGE_SURJ:
!f:'a->'b. !s t. SURJ f s t = ((IMAGE f s) = t)
Proof

QED

Theorem IMAGE_SURJ:
!f:'a->'b. !s t. SURJ f s t = ((IMAGE f s) = t)
Proof
strip_tac >> strip_tac >> rw[pred_setTheory.SURJ_DEF] >> fs[] >> fs[pred_setTheory.EXTENSION] >> fs[pred_setTheory.SPECIFICATION] >> fs[] >> fs[pred_setTheory.IMAGE_applied] >> fs[pred_setTheory.IN_APP, boolTheory.RES_EXISTS_THM] >> metis_tac[]
QED

Theorem INJ_DELETE:
!f s t. INJ f s t ==> !e. e IN s ==> INJ f (s DELETE e) (t DELETE (f e))
Proof
strip_tac >> strip_tac >> strip_tac 
>> fs[] >> rw[] 
>> (fs[pred_setTheory.INJ_DEF] >> (strip_tac >> fs[pred_setTheory.IN_DELETE, boolTheory.IMP_DISJ_THM] 
>- (metis_tac[pred_setTheory.IN_APP]) 
>- (fs[] >> (fs[] >> (fs[] >> (metis_tac[]))))))
QED

Theorem DISJOINT_INSERT:
!(x:'a) s t. DISJOINT (x INSERT s) t <=> DISJOINT s t /\ x NOTIN t
Proof
strip_tac >> strip_tac >> strip_tac 
>> fs[pred_setTheory.IN_INSERT, pred_setTheory.INSERT_DEF, pred_setTheory.IN_DISJOINT] 
>> metis_tac[]
QED


Theorem INSERT_INTER:
!x:'a. !s t. (x INSERT s) INTER t = (if x IN t then x INSERT (s INTER t) else s INTER t)
Proof
strip_tac >> strip_tac >> 
rw[pred_setTheory.INSERT_DEF, pred_setTheory.SPECIFICATION, pred_setTheory.INTER_DEF] 
>- (rw[pred_setTheory.GSPEC_ETA] >> metis_tac[])
>- (rw[] >> (rw[pred_setTheory.GSPEC_ETA] >> metis_tac[]))
QED


Theorem ABSORPTION:
!x:'a. !s. (x IN s) <=> (x INSERT s = s)
Proof
strip_tac 
>> rw[pred_setTheory.INSERT_DEF] 
>> fs[pred_setTheory.GSPEC_ETA, pred_setTheory.INSERT_DEF] 
>> metis_tac[pred_setTheory.SPECIFICATION]
QED


Theorem SET_MINIMUM:
!s:'a -> bool. !M. (?x. x IN s) <=> ?x. x IN s /\ !y. y IN s ==> M x <= M y
Proof
rw[] 
>> fs[boolTheory.IMP_CONG, boolTheory.EQ_TRANS, boolTheory.EQ_IMP_THM] 
>> rw[arithmeticTheory.WOP_measure, boolTheory.COND_ABS] 
>> metis_tac[boolTheory.ONE_ONE_THM]
QED

Theorem FILTER_COMM:
!f1 f2 l. FILTER f1 (FILTER f2 l) = FILTER f2 (FILTER f1 l)
Proof
Induct_on `l` 
>- rw[]
>- rw[]
QED

Theorem REVERSE_11:
!l1 l2:'a list. (REVERSE l1 = REVERSE l2) <=> (l1 = l2)
Proof
strip_tac >> strip_tac >> metis_tac[listTheory.REVERSE_REVERSE]
QED

Theorem MEM_MAP_f:
!f l a. MEM a l ==> MEM (f a) (MAP f l)
Proof
strip_tac >> strip_tac >> strip_tac >> rw[listTheory.MEM_MAP] >> (metis_tac[listTheory.MEM_MAP])
QED

Theorem FLAT_APPEND:
!l1 l2. FLAT (APPEND l1 l2) = APPEND (FLAT l1) (FLAT l2)
Proof
strip_tac >> strip_tac >> 
Induct_on `l1` 
>- (rw[listTheory.APPEND, listTheory.FLAT]) 
>- (fs[listTheory.APPEND] >> fs[listTheory.APPEND_ASSOC, listTheory.FLAT])
QED


Theorem MONO_EXISTS:
(!x. P x ==> Q x) ==> (EXISTS P l ==> EXISTS Q l)
Proof
rw[listTheory.EXISTS_MEM] >> metis_tac[]
QED


Theorem EVERY_CONJ:
!P Q l. EVERY (\(x:'a). (P x) /\ (Q x)) l = (EVERY P l /\ EVERY Q l)
Proof
rw[] >> Induct_on `l` 
>- (rw[listTheory.EVERY_DEF])
>- (rw[listTheory.EVERY_DEF] >> metis_tac[])
QED

(* A minimal reconstruction of the proof found by Tactic-Zero *)
Theorem EVERY2_DROP:
!R l1 l2 n.
      EVERY2 R l1 l2 ==> EVERY2 R (DROP n l1) (DROP n l2)
Proof
Induct_on ‘n’ 
>- (strip_tac >> fs[])
>- (Induct_on ‘l1’
   >- (fs[])
   >- (rpt strip_tac >> fs[]))
QED

(* Human proof in the library *)
val EVERY2_DROP = Q.store_thm("EVERY2_DROP",
   `!R l1 l2 n.
      EVERY2 R l1 l2 ==> EVERY2 R (DROP n l1) (DROP n l2)`,
  REPEAT STRIP_TAC THEN IMP_RES_TAC LIST_REL_LENGTH
  THEN Q.PAT_ASSUM `LIST_REL P xs ys` MP_TAC
  THEN ONCE_REWRITE_TAC [GSYM TAKE_DROP] THEN REPEAT STRIP_TAC
  THEN ONCE_REWRITE_TAC [TAKE_DROP]
  THEN Cases_on `n <= LENGTH l1`
  THEN1 (METIS_TAC [EVERY2_APPEND,LENGTH_DROP,LENGTH_TAKE])
  THEN fs [GSYM NOT_LESS] THEN `LENGTH l1 <= n` by numLib.DECIDE_TAC
  THEN fs [DROP_LENGTH_TOO_LONG]
  THEN rfs [DROP_LENGTH_TOO_LONG]);

Theorem EVERY2_DROP:
!R l1 l2 n.
      EVERY2 R l1 l2 ==> EVERY2 R (DROP n l1) (DROP n l2)
Proof
Induct_on ‘n’ 
>- (strip_tac >> fs[])
>- (Induct_on ‘l1’
   >- (fs[])
   >- (rpt strip_tac >> fs[]))
QED

Theorem example3:
∀(R :α -> β -> bool) (l1 :α list) (l2 :β list) (n :num). LIST_REL R l1 l2 ⇒ LIST_REL R (DROP n l1) (DROP n l2)
Proof
Induct_on `n` >- (strip_tac >> fs[listTheory.MAP_SNOC, listTheory.LIST_REL_O, listTheory.LAST_MAP, listTheory.SNOC_CASES, listTheory.LENGTH_FILTER_LEQ_MONO]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `l1` >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.NOT_CONS_NIL, listTheory.LIST_APPLY_o, listTheory.MAP_LIST_BIND, listTheory.GENLIST_CONS, listTheory.list_size_def]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.EVERY_GENLIST, listTheory.GENLIST_CONS, listTheory.EL_APPEND_EQN, listTheory.LENGTH_APPEND, listTheory.TAKE_APPEND2]))

Induct_on ‘n’ 
>- (strip_tac >> fs[])
>- (Induct_on `l1`
   >- (fs[])
   >- (rpt strip_tac >> fs[]))


Induct_on ‘n’ 
>- (strip_tac >> fs[])
>- (Induct_on `l1`
   >- (fs[])
   >- (rpt strip_tac >> fs[]))
QED

Theorem example2:
∀(n :num) (l1 :α list) (l2 :α list). TAKE n (l1 ++ l2) = TAKE n l1 ++ TAKE (n − LENGTH l1) l2
Proof
Induct_on `n` >- (Induct_on `l1` >- (rw[boolTheory.DISJ_EQ_IMP, listTheory.EL_MAP, arithmeticTheory.EXP, listTheory.NULL, listTheory.oEL_def]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> rw[listTheory.GENLIST_AUX_compute, arithmeticTheory.MAX_LT, arithmeticTheory.ZERO_LESS_EXP, listTheory.ALL_DISTINCT_ZIP_SWAP, listTheory.ZIP_MAP])) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `l1` >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.ADD_DIV_ADD_DIV, listTheory.MEM_APPEND, listTheory.MAP2_MAP, arithmeticTheory.MAX_ASSOC, arithmeticTheory.LE]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.LESS_EQ_LESS_EQ_MONO, listTheory.LUPDATE_def, listTheory.FOLDR, arithmeticTheory.EXP_1, arithmeticTheory.LESS_EQ]))

Induct_on `n`
simp[]
Induct_on `l1`
simp[]
simp[]

QED

Theorem example:
p ∧ q ⇒ p ∧ q
Proof
strip_tac >> 
(* Induct_on ‘p’ *)
simp[] >> simp[]
QED

Theorem ex201 : 
∀(y :α) (l :α list). MEM y l ⇔ FOLDR (λ(x :α) (l' :bool). y = x ∨ l') F l
Proof
Induct_on ‘l’
rw[listTheory.EVERY_FLAT, listTheory.LIST_REL_MAP_inv_image, listTheory.DROP_EQ_NIL, listTheory.LIST_APPLY_def, listTheory.EVERY_MEM]
rw[listTheory.LENGTH_TAKE_EQ, listTheory.LUPDATE_NIL, listTheory.FIND_def, listTheory.DROP_GENLIST, listTheory.REVERSE_SNOC]
rw[]
QED

Theorem ex201 : 
∀(l1 :α list) (l2 :α list). TAKE (LENGTH l1) (l1 ++ l2) = l1
Proof
rw[listTheory.TAKE_APPEND1]
QED


Theorem ex20 : 
(∀(x :α). x ∈ (s :α -> bool) ⇒ (INL x :α + β) ∈ (t :α + β -> bool)) ⇒ INJ (INL :α -> α + β) s t
Proof
fs[]
QED

Theorem failure_pred_set: 
∀s t. countable s ∨ countable t ⇒ countable (s ∩ t)
Proof
simp[boolTheory.DISJ_IMP_THM, pred_setTheory.PHP, pred_setTheory.SUBSET_MAX_SET, pred_setTheory.TC_SUBSET_THM, pred_setTheory.IMAGE_IN] >> simp[pred_setTheory.REST_DEF, pred_setTheory.COUNTABLE_ALT, pred_setTheory.COUNT_ZERO, pred_setTheory.COUNT_NOT_EMPTY, boolTheory.F_DEF] >> strip_tac >> rw[pred_setTheory.BIJ_INSERT, boolTheory.EQ_SYM, pred_setTheory.IN_INTER, boolTheory.DISJ_EQ_IMP, pred_setTheory.NOT_EMPTY_SING] >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> metis_tac[pred_setTheory.COUNTABLE_COUNT, pred_setTheory.COUNT_SUC, pred_setTheory.IN_GSPEC_IFF, pred_setTheory.FINITE_BIJ_COUNT_EQ, pred_setTheory.SPECIFICATION]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> metis_tac[pred_setTheory.SCHROEDER_CLOSED, pred_setTheory.IN_ABS, pred_setTheory.BIJ_TRANS, pred_setTheory.IN_BIGUNION, boolTheory.literal_case_DEF])

fs[pred_setTheory.COMPL_CLAUSES, boolTheory.FORALL_DEF, pred_setTheory.FINITE_PSUBSET_INFINITE, pred_setTheory.CARD_SING, pred_setTheory.SUBSET_DEF] >> rw[pred_setTheory.IN_INTER, pred_setTheory.IN_GSPEC_IFF, pred_setTheory.FINITE_INSERT, pred_setTheory.EXTENSION, pred_setTheory.SUBSET_DEF] >> fs[pred_setTheory.IN_APP, pred_setTheory.INJ_CARD, boolTheory.EQ_SYM, pred_setTheory.BIGINTER, boolTheory.BOTH_EXISTS_AND_THM] >> fs[pred_setTheory.INTER_DEF, pred_setTheory.POW_DEF, pred_setTheory.SUBSET_UNION_ABSORPTION, boolTheory.literal_case_CONG] >> rw[pred_setTheory.DELETE_applied, pred_setTheory.COUNTABLE_ALT, pred_setTheory.INJ_IMAGE_SUBSET, pred_setTheory.MAX_SET_ELIM, pred_setTheory.IN_GSPEC_IFF] >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> metis_tac[pred_setTheory.SPECIFICATION, pred_setTheory.DELETE_SUBSET, pred_setTheory.COMPL_applied, pred_setTheory.GSPEC_ETA, pred_setTheory.IN_ABS]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> metis_tac[pred_setTheory.IN_APP, pred_setTheory.SPECIFICATION, pred_setTheory.IN_COMPL, pred_setTheory.IMAGE_IN, boolTheory.RES_FORALL_DEF])

simp[boolTheory.DISJ_IMP_THM, pred_setTheory.PHP, pred_setTheory.SUBSET_MAX_SET, pred_setTheory.TC_SUBSET_THM, pred_setTheory.IMAGE_IN]

simp[pred_setTheory.REST_DEF, pred_setTheory.COUNTABLE_ALT, pred_setTheory.COUNT_ZERO, pred_setTheory.COUNT_NOT_EMPTY, boolTheory.F_DEF]

strip_tac

rw[pred_setTheory.BIJ_INSERT, boolTheory.EQ_SYM, pred_setTheory.IN_INTER, boolTheory.DISJ_EQ_IMP, pred_setTheory.NOT_EMPTY_SING]

metis_tac[pred_setTheory.COUNTABLE_COUNT, pred_setTheory.COUNT_SUC, pred_setTheory.IN_GSPEC_IFF, pred_setTheory.FINITE_BIJ_COUNT_EQ, pred_setTheory.SPECIFICATION]

metis_tac[pred_setTheory.SCHROEDER_CLOSED, pred_setTheory.IN_ABS, pred_setTheory.BIJ_TRANS, pred_setTheory.IN_BIGUNION, boolTheory.literal_case_DEF]
QED

Theorem polish_timout: 
∀p q. q ≠ 0 ⇒ ABS (p quot q * q) ≤ ABS p
Proof

QED

Theorem debug3:
∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)
Proof
Induct_on `b` >- (Induct_on `c` >- (simp[arithmeticTheory.ADD_CLAUSES, arithmeticTheory.SUB, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.ADD]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.LESS_OR, arithmeticTheory.MULT_CLAUSES, arithmeticTheory.EXP])) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (rw[arithmeticTheory.EXP, arithmeticTheory.FUNPOW, arithmeticTheory.LESS_OR, arithmeticTheory.ADD_CLAUSES]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `c` >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.LESS_EQ, arithmeticTheory.LESS_IMP_LESS_OR_EQ, arithmeticTheory.LESS_ADD]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> simp[arithmeticTheory.LEFT_ADD_DISTRIB, arithmeticTheory.EXP_ADD, arithmeticTheory.LESS_OR, arithmeticTheory.ADD_COMM])))
QED

Theorem debug2:
x < b ** x ⇔ 1 < b ∨ x = 0
Proof
Induct_on `x` >- (fs[arithmeticTheory.SUC_ONE_ADD, arithmeticTheory.MULT_CLAUSES, arithmeticTheory.LESS_EQ, arithmeticTheory.EXP, arithmeticTheory.SUB_ELIM_THM_EXISTS]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (simp[arithmeticTheory.ADD_CLAUSES, arithmeticTheory.LESS_OR, arithmeticTheory.PRE_SUC_EQ, arithmeticTheory.EXP, arithmeticTheory.DIV_LT_X]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> simp[arithmeticTheory.ADD_ASSOC, arithmeticTheory.X_LE_X_SQUARED, arithmeticTheory.DIV_LT_X, arithmeticTheory.DIV_EQ_X] >> (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (rw[arithmeticTheory.MULT, arithmeticTheory.ADD_CLAUSES, arithmeticTheory.EXP, arithmeticTheory.LESS_EQ_LESS_EQ_MONO, arithmeticTheory.PRE_SUB1]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (fs[arithmeticTheory.EXP_BASE_LEQ_MONO_SUC_IMP, arithmeticTheory.NOT_NUM_EQ, arithmeticTheory.EXP, arithmeticTheory.LESS_EQ_ADD_SUB, arithmeticTheory.X_LT_X_SQUARED]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> simp[arithmeticTheory.X_LT_EXP_X, arithmeticTheory.MULT, arithmeticTheory.LESS_MONO_MULT2, prim_recTheory.WF_PRED, arithmeticTheory.DIV_EQ_X])))))

(* if first do rpt (pop_assum mp_tac) >> simp[arithmeticTheory.X_LT_EXP_X, arithmeticTheory.MULT, arithmeticTheory.LESS_MONO_MULT2, prim_recTheory.WF_PRED, arithmeticTheory.DIV_EQ_X], then  *)

(* rpt (pop_assum mp_tac) >> rpt strip_tac >> simp[arithmeticTheory.X_LT_EXP_X, arithmeticTheory.MULT, arithmeticTheory.LESS_MONO_MULT2, prim_recTheory.WF_PRED, arithmeticTheory.DIV_EQ_X] *)

(* will work [because of cache?] *)
QED

Theorem alt:
(∀(b :num). SUC (c :num) ≤ b ⇒ ∀(a :num). a + b − SUC c = a + (b − SUC c)) ==>
(((∀(b :num). (c :num) ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC c ≤ (b :num) ⇒ (a :num) + b − SUC c = a + (b − SUC c)) ⇒ SUC (SUC c) ≤ b ⇒ a + b − SUC (SUC c) = a + (b − SUC (SUC c))) ==>
((∀(b :num). (c :num) ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC c ≤ SUC (b :num) ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC (SUC c) ≤ SUC b ⇒ a + SUC b − SUC (SUC c) = a + (SUC b − SUC (SUC c))
Proof
rw[arithmeticTheory.LESS_ADD_NONZERO, arithmeticTheory.SUC_SUB, arithmeticTheory.MULT, arithmeticTheory.EXP, arithmeticTheory.MULT_SUC]
QED

Theorem stepwise3:
 (a :num) + SUC (SUC (b :num)) − SUC (c :num) = a + (SUC (SUC b) − SUC c)
Proof
rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]
QED

Theorem stepwise2:
((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ ∀(c :num). c ≤ SUC b ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ==> ((∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ⇒ ∀(c :num). c ≤ SUC (SUC b) ⇒ ∀(a :num). a + SUC (SUC b) − c = a + (SUC (SUC b) − c))
Proof
rpt strip_tac
Induct_on ‘c’
QED

((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC (c :num) ≤ SUC b ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c)

Theorem stepwise1:
((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC (c :num) ≤ SUC b ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c)
Proof
rpt strip_tac
rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]
QED

Theorem debug1:
∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)
Proof
Induct_on `b`
rw[arithmeticTheory.NUMERAL_DEF, arithmeticTheory.MULT, arithmeticTheory.ADD_ASSOC, arithmeticTheory.LESS_SUB_ADD_LESS, arithmeticTheory.nat_elim__magic]
Induct_on `b`
simp[arithmeticTheory.LESS_ADD_1, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.LESS_ADD, arithmeticTheory.MULT_CLAUSES]
rpt strip_tac 
Induct_on `c`
QED

Theorem stepwise5:
   (∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ⇒
   (((∀(c :num). c ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒
     (c :num) ≤ SUC b ⇒ (a :num) + SUC b − c = a + (SUC b − c)) ⇒
    c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − c = a + (SUC (SUC b) − c)) ⇒
   ((∀(c :num). c ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC c ≤ SUC b ⇒
    a + SUC b − SUC c = a + (SUC b − SUC c)) ⇒
   SUC c ≤ SUC (SUC b) ⇒
   a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c)
Proof
rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]
QED

Theorem stepwise4:
(∀c. c ≤ SUC b ⇒ ∀a. a + SUC b − c = a + (SUC b − c)) ==>
(((∀c. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)) ⇒ c ≤ SUC b ⇒ a + SUC b − c = a + (SUC b − c)) ⇒ c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − c = a + (SUC (SUC b) − c)) ==> (((∀c. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)) ⇒ SUC c ≤ SUC b ⇒ a + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c))
Proof
rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]
QED

Theorem stepwise4:
   (∀c. c ≤ SUC b ⇒ ∀a. a + SUC b − c = a + (SUC b − c)) ==>
   (((∀c. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)) ⇒ c ≤ SUC b ⇒
     a + SUC b − c = a + (SUC b − c)) ⇒ c ≤ SUC (SUC b) ⇒
    a + SUC (SUC b) − c = a + (SUC (SUC b) − c)) ==>
   (((∀c. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)) ⇒ SUC c ≤ SUC b ⇒
    a + SUC b − SUC c = a + (SUC b − SUC c)) ⇒
   SUC c ≤ SUC (SUC b) ⇒
   a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c))
Proof
rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]
QED

Theorem stepwise5:
(∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ==> 
(((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ (c :num) ≤ SUC b ⇒ (a :num) + SUC b − c = a + (SUC b − c)) ⇒ c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − c = a + (SUC (SUC b) − c)) ==>
(((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC (c :num) ≤ SUC b ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c))
Proof
rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]

QED

Theorem stepwise6:
(∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ==> (((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ (c :num) ≤ SUC b ⇒ (a :num) + SUC b − c = a + (SUC b − c)) ⇒ c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − c = a + (SUC (SUC b) − c)) ==> (((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC (c :num) ≤ SUC b ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c))
Proof
rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]

QED

(* Proof trace: [('∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)', 'Induct_on `c`'), ('∀(b :num). (0 :num) ≤ b ⇒ ∀(a :num). a + b − (0 :num) = a + (b − (0 :num))', 'fs[arithmeticTheory.MULT_CLAUSES, arithmeticTheory.nat_elim__magic, arithmeticTheory.LESS_OR, arithmeticTheory.LESS_EQ_ADD, arithmeticTheory.LESS_EQ_ADD]'), ('∀(b :num). SUC (c :num) ≤ b ⇒ ∀(a :num). a + b − SUC c = a + (b − SUC c)', 'Induct_on `c`'), ('(∀(b :num). (0 :num) ≤ b ⇒ ∀(a :num). a + b − (0 :num) = a + (b − (0 :num))) ⇒ SUC (0 :num) ≤ (b :num) ⇒ (a :num) + b − SUC (0 :num) = a + (b − SUC (0 :num))', 'simp[arithmeticTheory.PRE_SUB1, arithmeticTheory.ADD, arithmeticTheory.PRE_SUB1, arithmeticTheory.EVEN, arithmeticTheory.ADD_CLAUSES]'), ('(∀(b :num). SUC (c :num) ≤ b ⇒ ∀(a :num). a + b − SUC c = a + (b − SUC c)) ⇒ SUC (SUC c) ≤ (b :num) ⇒ (a :num) + b − SUC (SUC c) = a + (b − SUC (SUC c))', 'Induct_on `b`'), ('((∀(b :num). (c :num) ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC c ≤ (0 :num) ⇒ (a :num) + (0 :num) − SUC c = a + ((0 :num) − SUC c)) ⇒ SUC (SUC c) ≤ (0 :num) ⇒ a + (0 :num) − SUC (SUC c) = a + ((0 :num) − SUC (SUC c))', 'fs[arithmeticTheory.MULT_CLAUSES, arithmeticTheory.SUC_SUB1, arithmeticTheory.EXP, arithmeticTheory.SUC_SUB, arithmeticTheory.NUMERAL_DEF]'), ('((∀(b :num). (c :num) ≤ b ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC c ≤ SUC (b :num) ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC (SUC c) ≤ SUC b ⇒ a + SUC b − SUC (SUC c) = a + (SUC b − SUC (SUC c))', 'rw[arithmeticTheory.LESS_ADD_NONZERO, arithmeticTheory.SUC_SUB, arithmeticTheory.MULT, arithmeticTheory.EXP, arithmeticTheory.MULT_SUC]')]                             *)
(* Exception: Induct_on `c` >- (fs[arithmeticTheory.MULT_CLAUSES, arithmeticTheory.nat_elim__magic, arithmeticTheory.LESS_OR, arithmeticTheory.LESS_EQ_ADD]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `c` >- (simp[arithmeticTheory.PRE_SUB1, arithmeticTheory.ADD, arithmeticTheory.EVEN, arithmeticTheory.ADD_CLAUSES]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.MULT_CLAUSES, arithmeticTheory.SUC_SUB1, arithmeticTheory.EXP, arithmeticTheory.SUC_SUB, arithmeticTheory.NUMERAL_DEF]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> rw[arithmeticTheory.LESS_ADD_NONZERO, arithmeticTheory.SUC_SUB, arithmeticTheory.MULT, arithmeticTheory.EXP, arithmeticTheory.MULT_SUC]))) to ∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c) to be debugged  *)

(* Proof trace: [('∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)', 'Induct_on `b`'), ('∀(c :num). c ≤ (0 :num) ⇒ ∀(a :num). a + (0 :num) − c = a + ((0 :num) − c)', 'rw[arithmeticTheory.NUMERAL_DEF, arithmeticTheory.MULT, arithmeticTheory.ADD_ASSOC, arithmeticTheory.LESS_SUB_ADD_LESS, arithmeticTheory.nat_elim__magic]'), ('∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)', 'Induct_on `b`'), ('(∀(c :num). c ≤ (0 :num) ⇒ ∀(a :num). a + (0 :num) − c = a + ((0 :num) − c)) ⇒ (c :num) ≤ SUC (0 :num) ⇒ (a :num) + SUC (0 :num) − c = a + (SUC (0 :num) − c)', 'simp[arithmeticTheory.LESS_ADD_1, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.LESS_ADD, arithmeticTheory.MULT_CLAUSES]'), ('(∀(c :num). c ≤ SUC (b :num) ⇒ ∀(a :num). a + SUC b − c = a + (SUC b − c)) ⇒ (c :num) ≤ SUC (SUC b) ⇒ (a :num) + SUC (SUC b) − c = a + (SUC (SUC b) − c)', 'Induct_on `c`'), ('((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ (0 :num) ≤ SUC b ⇒ (a :num) + SUC b − (0 :num) = a + (SUC b − (0 :num))) ⇒ (0 :num) ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − (0 :num) = a + (SUC (SUC b) − (0 :num))', 'fs[arithmeticTheory.ADD1, arithmeticTheory.EXP, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.ADD_INV_0_EQ, arithmeticTheory.LESS_OR]'), ('((∀(c :num). c ≤ (b :num) ⇒ ∀(a :num). a + b − c = a + (b − c)) ⇒ SUC (c :num) ≤ SUC b ⇒ (a :num) + SUC b − SUC c = a + (SUC b − SUC c)) ⇒ SUC c ≤ SUC (SUC b) ⇒ a + SUC (SUC b) − SUC c = a + (SUC (SUC b) − SUC c)', 'rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]')] *)

Theorem debug1:
∀c b. c ≤ b ⇒ ∀a. a + b − c = a + (b − c)
Proof
Induct_on `b` >- (rw[arithmeticTheory.NUMERAL_DEF, arithmeticTheory.MULT, arithmeticTheory.ADD_ASSOC, arithmeticTheory.LESS_SUB_ADD_LESS, arithmeticTheory.nat_elim__magic]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `b` >- (simp[arithmeticTheory.LESS_ADD_1, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.LESS_ADD, arithmeticTheory.MULT_CLAUSES]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `c` >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[arithmeticTheory.ADD1, arithmeticTheory.EXP, arithmeticTheory.NUMERAL_DEF, arithmeticTheory.ADD_INV_0_EQ, arithmeticTheory.LESS_OR]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> rw[arithmeticTheory.EXP, arithmeticTheory.EXP, arithmeticTheory.MULT_RIGHT_1, arithmeticTheory.SUB_LESS_OR, arithmeticTheory.RIGHT_SUB_DISTRIB]))) 
QED

Theorem not_found:
!l1 l2 n x. LENGTH l1 <= n ==> (LUPDATE x n (l1 ++ l2) = l1 ++ (LUPDATE x (n-LENGTH l1) l2))
Proof
Induct_on ‘l1’
>-(rw[] >> fs[listTheory.LUPDATE_def,listTheory.LENGTH])
>-(Induct_on ‘n’
   >-(fs[listTheory.LUPDATE_def,listTheory.LENGTH])
   >-(fs[listTheory.LUPDATE_def,listTheory.LENGTH]))
QED

Theorem e2:
MEM a (b::l) ==> MEM b l ==> MEM a l
Proof
rw[listTheory.MEM_APPEND, listTheory.EVERY_DEF, listTheory.list_size_def, listTheory.LIST_TO_SET_APPEND] >> (rpt (pop_assum mp_tac) >> rpt strip_tac >> Induct_on `l` >- (rw[listTheory.lazy_list_case_compute, listTheory.DROP_compute, listTheory.FLAT, listTheory.list_size_def, listTheory.LIST_TO_SET]) >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.LIST_TO_SET, listTheory.NOT_NIL_EQ_LENGTH_NOT_0, listTheory.FLAT, listTheory.MAP, listTheory.MEM]))
QED

Theorem talk:
p ∧ q ⇒ p ∧ q
Proof
strip_tac
QED

(* need to record normalized version in history *)
Theorem recon1:
∀P l1 l2. EXISTS P (l1 ++ l2) ⇔ EXISTS P l1 ∨ EXISTS P l2
Proof
fs[listTheory.DROP_def, listTheory.MEM_FLAT, listTheory.MAP_EQ_CONS, listTheory.EVERY_APPEND, listTheory.ALL_DISTINCT_SET_TO_LIST] >> fs[listTheory.EXISTS_NOT_EVERY, listTheory.EXISTS_MEM, listTheory.LIST_TO_SET_APPEND, listTheory.ITSET_eq_FOLDL_SET_TO_LIST, listTheory.REVERSE_DEF] >> rw[listTheory.LIST_TO_SET, listTheory.SUM, listTheory.LIST_IGNORE_BIND_def, listTheory.mem_exists_set, listTheory.CONS] >> metis_tac[listTheory.FLAT, listTheory.LIST_TO_SET, listTheory.EVERY_DEF, listTheory.MEM_GENLIST, listTheory.MAP]
QED

Theorem failure4:
NULL (l1 ++ l2) ⇔ NULL l1 ∧ NULL l2
Proof
fs[listTheory.LENGTH_EQ_NIL, listTheory.NULL_EQ, listTheory.NULL_LENGTH, listTheory.SING_HD]

rw[listTheory.NULL_EQ, listTheory.NULL_LENGTH, listTheory.SING_HD, listTheory.list_distinct, listTheory.LENGTH_EQ_NIL]
(* rw[listTheory.NULL_EQ] *)
rw[]
QED

Theorem bar:
MAP f [x] = l ==> LENGTH l = 1
Proof
rw[]
rw[MAP, LENGTH]
QED

Theorem foo_failure5:
(∀P.LENGTH (dropWhile P ls) = LENGTH [] ∧ (∀x. x < LENGTH (dropWhile P ls) ⇒ EL x (dropWhile P ls) = EL x []) ⇔ EVERY P ls) ==> (LENGTH (dropWhile P (h::ls)) = LENGTH [] ∧ (∀x.x < LENGTH (dropWhile P (h::ls)) ⇒ EL x (dropWhile P (h::ls)) = EL x []) ⇔ P h ∧ EVERY P ls)
Proof
rpt strip_tac >> rw[listTheory.MEM, listTheory.EXISTS_DEF, listTheory.ALL_DISTINCT, listTheory.dropWhile_def, listTheory.MAP]
QED

Theorem failure5:
∀P ls. dropWhile P ls = [] ⇔ EVERY P ls
Proof
Induct_on `ls`
>- (rw[listTheory.LENGTH_DROP, listTheory.MAP, listTheory.LIST_APPLY_def, listTheory.EL_MAP2, listTheory.EVERY_DEF]>> (simp[listTheory.LLEX_def, listTheory.APPEND, listTheory.dropWhile_def, listTheory.EVERY_DEF, listTheory.splitAtPki_EQN]))
>- (rpt (pop_assum mp_tac) >> rpt strip_tac >> (fs[listTheory.MAP, listTheory.TAKE_def, listTheory.isPREFIX, listTheory.EXISTS_DEF, listTheory.EXISTS_DEF]>> (rpt (pop_assum mp_tac) >> rpt strip_tac >> (fs[listTheory.SUM_ACC_SUM_LEM, listTheory.EVERY_DEF, listTheory.APPEND_eq_NIL, listTheory.EXISTS_DEF, listTheory.LIST_EQ_REWRITE]>> (rpt (pop_assum mp_tac) >> rpt strip_tac >> (rw[listTheory.MEM, listTheory.EXISTS_DEF, listTheory.ALL_DISTINCT, listTheory.dropWhile_def, listTheory.MAP]>- (rpt (pop_assum mp_tac) >> rpt strip_tac >> (rw[listTheory.dropWhile_def, listTheory.LIST_BIND_THM, listTheory.LENGTH, listTheory.EXISTS_DEF, listTheory.MEM]>> (rpt (pop_assum mp_tac) >> rpt strip_tac >> (fs[listTheory.LENGTH, listTheory.LENGTH, listTheory.EVERY_DEF, listTheory.EVERY_DEF, listTheory.dropWhile_def]))))>- (rpt (pop_assum mp_tac) >> rpt strip_tac >> (Induct_on `ls`>- (rpt (pop_assum mp_tac) >> rpt strip_tac >> (fs[listTheory.MAP, listTheory.MAP, listTheory.MAP, listTheory.MAP, listTheory.MAP]>> (rpt (pop_assum mp_tac) >> rpt strip_tac >> (rw[listTheory.dropWhile_def, listTheory.EXISTS_LIST, listTheory.dropWhile_def, listTheory.LENGTH, listTheory.LENGTH]))))>- (rpt (pop_assum mp_tac) >> rpt strip_tac >> (fs[listTheory.ALL_DISTINCT, listTheory.MEM, listTheory.UNZIP_THM, listTheory.ZIP, listTheory.MAP]>> (rpt (pop_assum mp_tac) >> rpt strip_tac >> (rw[listTheory.dropWhile_def, listTheory.LENGTH, listTheory.LENGTH, listTheory.EL_MAP2, listTheory.EXISTS_DEF]))))))))))))
QED



Theorem failure4:
LIST_BIND [] f = [] ∧ LIST_BIND (h::t) f = f h ++ LIST_BIND t f
Proof
fs[listTheory.EXISTS_SNOC, listTheory.MAP_EQ_APPEND, listTheory.NULL_DEF, listTheory.FILTER_REVERSE, listTheory.LIST_BIND_def]>>(fs[listTheory.MAP, listTheory.UNION_APPEND, listTheory.splitAtPki_MAP, listTheory.REVERSE_DEF, listTheory.TAKE_nil]>>(Induct_on `t`>-(rw[listTheory.EL_REVERSE, listTheory.FLAT, listTheory.LENGTH, listTheory.GENLIST, listTheory.SET_TO_LIST_EMPTY])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.EXISTS_DEF, listTheory.MAP, listTheory.EXISTS_DEF, listTheory.EXISTS_DEF, listTheory.MAP]>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.LIST_TO_SET, listTheory.FLAT, listTheory.GENLIST, listTheory.FLAT, listTheory.EVERY_DEF])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.GENLIST, listTheory.MAP, listTheory.FLAT, listTheory.TL, listTheory.FILTER_F]))))
QED


Theorem failure3:
FLAT [] = [] ∧ FLAT ([]::t) = FLAT t ∧ FLAT ((h::t1)::t2) = h::FLAT (t1::t2)
Proof
Induct_on `t2`>-(rw[listTheory.UNZIP_THM, listTheory.LAST_DEF, listTheory.EXISTS_MEM, listTheory.LENGTH, listTheory.FLAT])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.MAP, listTheory.SHORTLEX_def, listTheory.LIST_TO_SET, listTheory.dropWhile_def, listTheory.NULL_APPEND]>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.MAP, listTheory.MEM, listTheory.TAKE_def, listTheory.FLAT, listTheory.FLAT])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.GENLIST_GENLIST_AUX, listTheory.LIST_TO_SET, listTheory.FLAT, listTheory.EXISTS_DEF, listTheory.NULL_LENGTH]))
QED

Theorem foo_check_fix:
UNZIP ([] :(α # β) list ) = (([] :α list ),([] :β list )) ==> UNZIP ([] :(α # β) list ) = (([] :α list ),([] :β list ))
Proof
rpt strip_tac >> fs[]
QED

Theorem foo_check_failure2:
(UNZIP [] = ([],[])) ==> (UNZIP [] = ([],[]))
Proof
rpt strip_tac >> fs[]
QED

Theorem foo_check_failure1:
(UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))) ==> (∀h. UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::h::t) = (let (L1,L2) = UNZIP (h::t) in (x::L1,y::L2)))
Proof
rpt strip_tac >> fs[listTheory.MEM, listTheory.MEM, listTheory.MEM, listTheory.MEM, listTheory.MEM]
QED

Theorem check_failure1:
UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))
Proof
Induct_on `t`>-(rw[listTheory.UNZIP, listTheory.MEM, listTheory.MEM, listTheory.MEM, listTheory.EL_simp])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.MEM, listTheory.MEM, listTheory.MEM, listTheory.MEM, listTheory.MEM]>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.list_size_cong, listTheory.LIST_REL_CONS1, listTheory.LIST_REL_CONJ, listTheory.LIST_REL_CONS1, listTheory.UNZIP])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.MEM_SPLIT, listTheory.MEM, listTheory.ZIP_def, listTheory.MEM, listTheory.ZIP]>>(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.NULL_FILTER, listTheory.ZIP, listTheory.UNZIP, listTheory.MEM, listTheory.LENGTH_NIL])))
QED

Theorem failure_script_generation1:
(LIST_REL R [] y ⇔ y = []) ∧ (LIST_REL R x [] ⇔ x = [])
Proof
Induct_on ‘x’
>-(rw[listTheory.FILTER_NEQ_ID, listTheory.MAP_EQ_APPEND, listTheory.MEM, listTheory.LENGTH_NIL, listTheory.APPEND_eq_NIL]>-(Induct_on ‘y’>-(rw[listTheory.LIST_REL_def, listTheory.FILTER_APPEND_DISTRIB, listTheory.LIST_REL_def, listTheory.LIST_EQ_REWRITE, listTheory.MEM])>-(fs[listTheory.FILTER_COND_REWRITE, listTheory.LIST_REL_def, listTheory.list_case_eq, listTheory.list_case_eq, listTheory.list_case_eq]))>-(rw[listTheory.LIST_REL_def, listTheory.FILTER_APPEND_DISTRIB, listTheory.LIST_REL_def, listTheory.LIST_EQ_REWRITE, listTheory.MEM]))
>-(fs[listTheory.list_case_eq, listTheory.LIST_REL_def, listTheory.LIST_EQ_REWRITE, listTheory.list_case_eq, listTheory.LIST_REL_EL_EQN])
QED

Theorem foo_failure_timeout1:
((∃x0 t0. l = x0::t0 ∧ x = f x0 ∧ MAP f t0 = []) ⇔ ∃x0. l = [x0] ∧ x = f x0) ==> (x = f h ∧ MAP f l = [] ⇔ l = [] ∧ x = f h)
Proof

QED

Theorem failure_timeout1:
MAP f l = [x] ⇔ ∃x0. l = [x0] ∧ x = f x0
Proof
rw[listTheory.MAP_EQ_CONS, listTheory.LENGTH_SNOC, listTheory.GENLIST, listTheory.GENLIST, listTheory.HD_DROP]>>(Induct_on `l`>-(EQ_TAC>-(rw[listTheory.SWAP_REVERSE_SYM, listTheory.FILTER_NEQ_ID, listTheory.FRONT_DEF, listTheory.MAP, listTheory.MAP_APPEND])>-(rw[listTheory.EL_MAP2, listTheory.LENGTH, listTheory.GENLIST_PLUS_APPEND, listTheory.ZIP, listTheory.REVERSE_11]))>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> rw[listTheory.mem_exists_set, listTheory.SWAP_REVERSE, listTheory.NULL_LENGTH, listTheory.MAP, listTheory.FRONT_SNOC]>>(rpt (pop_assum mp_tac) >> rpt strip_tac >> metis_tac[listTheory.FLAT, listTheory.EVERY2_REVERSE, listTheory.MAP, listTheory.MAP_EQ_APPEND, listTheory.LIST_TO_SET_FILTER]>>(Induct_on `l`>-(metis_tac[listTheory.TAKE_APPEND1, listTheory.LIST_TO_SET_MAP, listTheory.FRONT_DEF, listTheory.MAP, listTheory.CARD_LIST_TO_SET_ALL_DISTINCT])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> fs[listTheory.MAP, listTheory.MAP_EQ_CONS, listTheory.GENLIST, listTheory.EXISTS_DEF, listTheory.mem_exists_set])))))
QED

Theorem foo2:
LIST_REL (λa b. P a b ∧ Q a b) l1 l2 ⇔ LIST_REL (λa b. P a b) l1 l2 ∧ LIST_REL (λa b. Q a b) l1 l2
Proof
Induct_on `l1`>-(Induct_on `l2`>-(rw[listTheory.LIST_REL_EL_EQN, listTheory.MEM, listTheory.LIST_REL_def, listTheory.LENGTH_EQ_NUM, listTheory.LIST_REL_EL_EQN])>-(rpt strip_tac >> fs[listTheory.LIST_REL_CONS2, listTheory.LIST_REL_EL_EQN, listTheory.LIST_REL_CONS1, listTheory.NULL_EQ, listTheory.LIST_REL_EL_EQN]))>-(rpt strip_tac >> fs[listTheory.LIST_REL_EL_EQN, listTheory.LIST_REL_EL_EQN, listTheory.LIST_EQ_REWRITE, listTheory.APPEND_11, listTheory.LIST_REL_EL_EQN]>>(rpt strip_tac >> metis_tac[listTheory.LIST_REL_def, listTheory.APPEND_11, listTheory.MEM, listTheory.APPEND_11, listTheory.APPEND_11]))
QED

(* timeout example *)
Theorem foo152:
BIGUNION (IMAGE f (set ls)) ⊆ s ⇔ ∀x. MEM x ls ⇒ f x ⊆ s
Proof
Induct_on `ls`>-(fs[listTheory.LIST_TO_SET, listTheory.LIST_TO_SET, listTheory.LIST_TO_SET, listTheory.MAP, listTheory.LIST_TO_SET])>-(rpt strip_tac >> fs[listTheory.MAP, listTheory.MAP, listTheory.LIST_TO_SET, listTheory.EXISTS_DEF, listTheory.ALL_DISTINCT]>>(rpt strip_tac >> Induct_on `ls`>-(fs[listTheory.UNZIP_MAP, listTheory.LENGTH, listTheory.LAST_CONS_cond, listTheory.LIST_TO_SET, listTheory.EVERY_DEF])>-(rpt strip_tac >> metis_tac[listTheory.SHORTLEX_def, listTheory.MEM_GENLIST, listTheory.MAP_EQ_APPEND, listTheory.FLAT, listTheory.EVERY_DEF])))
QED

Theorem foo123':
(∃h l'. LENGTH l' = LENGTH l ∧ l ⧺ [x] = h::l') ==> ∀h. ∃h' l'. LENGTH l' = LENGTH (h::l) ∧ h::l ⧺ [x] = h'::l'
Proof
rpt strip_tac >> rw[listTheory.LENGTH, listTheory.FLAT, listTheory.FILTER, listTheory.NULL_FILTER, listTheory.LENGTH]
QED


Theorem foo123:
∀x l. LENGTH (SNOC x l) = SUC (LENGTH l)
Proof
rw[listTheory.LENGTH_EQ_NUM, listTheory.LENGTH_DROP, listTheory.MEM, listTheory.LIST_TO_SET, listTheory.TAKE_LENGTH_ID]>>(Induct_on `l`>-(rw[listTheory.FILTER, listTheory.TAKE_def, listTheory.FILTER, listTheory.MAP, listTheory.MAP]>>(rw[listTheory.MEM, listTheory.LIST_TO_SET, listTheory.LIST_TO_SET, listTheory.LENGTH, listTheory.LIST_TO_SET]))>-(rpt strip_tac >> rw[listTheory.LENGTH, listTheory.FLAT, listTheory.FILTER, listTheory.NULL_FILTER, listTheory.LENGTH]))

QED

Theorem baa:
(ll = [] ∨ ∃x l. ll = l ⧺ [x]) ==> ∀h. h::ll = [] ∨ ∃x l. h::ll = l ⧺ [x]
Proof
rpt strip_tac >> rw[listTheory.LENGTH, listTheory.LIST_TO_SET, listTheory.ZIP, listTheory.MAP2_MAP, listTheory.LIST_TO_SET]
QED

Theorem baz:
∀ll. ll = [] ∨ ∃x l. ll = SNOC x l
Proof
rw[listTheory.LAST_APPEND_CONS, listTheory.EVERY_DEF, listTheory.ALL_DISTINCT, listTheory.EVERY_DEF, listTheory.WF_LIST_PRED]>>(Induct_on `ll`>-(rw[listTheory.isPREFIX_THM, listTheory.MAP, listTheory.NULL_LENGTH, listTheory.isPREFIX, listTheory.ZIP_MAP])>-(rpt (pop_assum mp_tac) >> rpt strip_tac >> rw[listTheory.LENGTH, listTheory.LIST_TO_SET, listTheory.ZIP, listTheory.MAP2_MAP, listTheory.LIST_TO_SET]))
QED

Theorem bar:
REVERSE l = [] ⇔ l = []
Proof
Induct_on `l`>-(fs[EL_simp_restricted, LENGTH_ZIP, SUM_eq_0, LENGTH_EQ_NUM, FILTER_NEQ_NIL])>-(rpt strip_tac >> fs[APPEND_LENGTH_EQ, LIST_REL_rules, FILTER_NEQ_NIL, REVERSE_11, LENGTH_REVERSE])
QED

Theorem foo: 
MAP f l = [x] ⇔ ∃x0. l = [x0] ∧ x = f x0
Proof
rw[listTheory.LIST_TO_SET_FLAT, listTheory.MAP_EQ_CONS, listTheory.APPEND_EQ_APPEND, listTheory.MAP, listTheory.oHD_def]
Induct_on `l`

EQ_TAC
rw[listTheory.FOLDR_CONG, listTheory.SINGL_APPLY_MAP, listTheory.INJ_MAP_EQ_IFF, listTheory.EL_SNOC, listTheory.MAP_EQ_CONS]
strip_tac
rw[listTheory.MAP, listTheory.MAP2, listTheory.ALL_DISTINCT_FLAT_REVERSE, listTheory.EXISTS_DEF, listTheory.TAKE_cons]

strip_tac

Induct_on `l`
fs[listTheory.splitAtPki_APPEND, listTheory.MAP, listTheory.MAP, listTheory.LENGTH_SNOC, listTheory.LLEX_CONG]

rw[listTheory.MAP2_APPEND, listTheory.EL_LENGTH_SNOC, listTheory.FOLDR_CONS, listTheory.APPEND_EQ_APPEND, listTheory.MAP]

QED


Theorem LIST_TO_SET:
  LIST_TO_SET [] = {} /\
  LIST_TO_SET (h::t) = h INSERT LIST_TO_SET t
Proof
rw [FUN_EQ_THM, IN_DEF] >> 

metis_tac[LIST_TO_SET_DEF]
  (* SRW_TAC [] [FUN_EQ_THM, IN_DEF] *)
QED


Theorem MAP2_APPEND:
  !xs ys xs1 ys1 f.
     (LENGTH xs = LENGTH xs1) ==>
     (MAP2 f (xs ++ ys) (xs1 ++ ys1) = MAP2 f xs xs1 ++ MAP2 f ys ys1)
Proof 
(* Induct >> Cases_on ‘xs1’ >> fs [MAP2] *)
Induct_on ‘xs1’
>- (fs[MAP2, LENGTH_NIL, LENGTH])
>- (Induct_on ‘xs’ >> fs[MAP2, LENGTH])
QED

Theorem list_case_compute:
!(l:'a list). list_CASE l (b:'b) f = if NULL l then b else f (HD l) (TL l)
Proof
Induct_on ‘l’ 
>- (rw [list_case_def, HD, TL, NULL_DEF])
>- (rw [list_case_def, HD, TL, NULL_DEF])
QED

Theorem CONS:
  !l : 'a list. ~NULL l ==> HD l :: TL l = l
Proof
Induct_on ‘l’
>- (simp[NULL])
>- (simp[NULL, HD, TL])
QED

Theorem MAP_EQ_CONS:
  MAP (f:'a -> 'b) l = h::t <=> ?x0 t0. l = x0::t0 /\ h = f x0 /\ t = MAP f t0
Proof
  Induct_on ‘l’
  >- (simp[MAP])
  >- (fs[MAP] >> metis_tac[])
QED

Theorem MAP_o:
  !f:'b->'c. !g:'a->'b.  MAP (f o g) = (MAP f) o (MAP g)
Proof
rpt strip_tac >> CONV_TAC FUN_EQ_CONV >> Induct_on ‘’
>- (simp[MAP])
>- (simp[MAP])
QED

Theorem MAP_TL_REVERSE:
∀l f. TL (MAP f l) = MAP f (TL l)
Proof
simp[MAP_TL]
QED

Theorem EL_MAP:
  !n l. n < (LENGTH l) ==> !f:'a->'b. EL n (MAP f l) = f (EL n l)
Proof
Induct_on ‘n’
  >-(Induct_on ‘l’
     >-simp[LENGTH, EL, HD, TL, MAP]
     >-metis_tac[LENGTH, EL, HD, TL, MAP])
  >-(rw[EL] >> rw[GSYM MAP_TL] >> FIRST_ASSUM MATCH_MP_TAC >> fs[LENGTH_TL])
QED

Theorem EVERY_EL:
  !(l:'a list) P. EVERY P l = !n. n < LENGTH l ==> P (EL n l)
Proof
Induct_on ‘l’
 >-(simp[EVERY_DEF, EL, LENGTH])
 >-(simp[EVERY_DEF, EL, LENGTH] >> rpt strip_tac >> 
    EQ_TAC
    >-(strip_tac >> 
       Induct_on ‘n’
       >-(simp[EL, LENGTH, HD])
       >-(strip_tac >> fs[EL, TL]))

    >-(rpt strip_tac
       >-(qpat_assum ‘∀n. n < SUC (LENGTH l) ⇒ P (EL n (h::l))’ (assume_tac o (SPEC (“0”))) >> fs[EL, HD])
       >-(qpat_assum ‘∀n. n < SUC (LENGTH l) ⇒ P (EL n (h::l))’ (assume_tac o (SPEC (“SUC n”))) >> fs[EL, TL])))
QED


Theorem EVERY_CONJ:
!P Q l. EVERY (\(x:'a). (P x) /\ (Q x)) l = (EVERY P l /\ EVERY Q l)
Proof
Induct_on ‘l’
>-(simp[EVERY_DEF])
>-(simp[EVERY_DEF] >> metis_tac[])
QED

Theorem EVERY_MAP:
!P f l:'a list. EVERY P (MAP f l) = EVERY (\x. P (f x)) l
Proof
Induct_on ‘l’
simp[EVERY_DEF, MAP]
simp[EVERY_DEF, MAP]
QED

Theorem MONO_EVERY:
(!x. P x ==> Q x) ==> (EVERY P l ==> EVERY Q l)
Proof
Induct_on ‘l’
simp[EVERY_DEF]
simp[EVERY_DEF]
QED

Theorem EXISTS_SIMP:
!c l:'a list. EXISTS (\x. c) l <=> l <> [] /\ c
Proof
Induct_on ‘l’
simp[EXISTS_DEF]
simp[EXISTS_DEF]
metis_tac[]
QED

Theorem LENGTH_CONS:
!l n. (LENGTH l = SUC n) =
          ?h:'a. ?l'. (LENGTH l' = n) /\ (l = CONS h l')
Proof
Induct_on ‘l’
simp[LENGTH]
simp[LENGTH]
QED

Theorem LENGTH_EQ_CONS:
!P:'a list->bool.
    !n:num.
      (!l. (LENGTH l = SUC n) ==> P l) =
      (!l. (LENGTH l = n) ==> (\l. !x:'a. P (CONS x l)) l)
Proof
rpt strip_tac
EQ_TAC
>-rpt strip_tac >> simp[] >> strip_tac >> qpat_assum ‘∀l. LENGTH l = SUC n ⇒ P l’ MATCH_MP_TAC >> fs[LENGTH]
>-strip_tac >> Induct_on ‘l’
  >-simp[LENGTH]
  >-fs[LENGTH]
QED

Theorem APPEND_eq_NIL:
(!l1 l2:'a list. ([] = APPEND l1 l2) <=> (l1=[]) /\ (l2=[])) /\
  (!l1 l2:'a list. (APPEND l1 l2 = []) <=> (l1=[]) /\ (l2=[]))
Proof
strip_tac
simp[]
simp[]
QED

Theorem EQ_APPEND:
∃ l1 l2. l = l1 ++ l2
Proof
Induct_on ‘l’
>- (simp[])
>- (metis_tac[APPEND_EQ_CONS])
QED

val LIST_INDUCT_TAC = INDUCT_THEN list_INDUCT ASSUME_TAC;

Theorem MAP_EQ_APPEND:
!l1 l2. (MAP (f:'a -> 'b) l = l1 ++ l2) <=>
      ?l10 l20. (l = l10 ++ l20) /\ (l1 = MAP f l10) /\ (l2 = MAP f l20)
Proof
Induct >>
asm_simp_tac bool_ss [MAP, APPEND_eq_NIL, APPEND_EQ_CONS]
>- metis_tac[]>>
rpt strip_tac >> asm_simp_tac bool_ss [EQ_IMP_THM] >> rpt strip_tac >>
asm_simp_tac bool_ss [RIGHT_AND_OVER_OR, EXISTS_OR_THM, MAP, NOT_CONS_NIL, PULL_EXISTS] >>
rw_tac bool_ss [MAP, NOT_CONS_NIL] >>
metis_tac[]
QED


Theorem MAP_EQ_APPEND:
!l1 l2. (MAP (f:'a -> 'b) l = l1 ++ l2) <=>
      ?l10 l20. (l = l10 ++ l20) /\ (l1 = MAP f l10) /\ (l2 = MAP f l20)
Proof
(*   REVERSE EQ_TAC THEN1 SIMP_TAC (srw_ss() ++ boolSimps.DNF_ss) [MAP_APPEND] THEN *)
(*   MAP_EVERY Q.ID_SPEC_TAC [‘l1’, ‘l2’, ‘l’] THEN LIST_INDUCT_TAC THEN *)
(*   SIMP_TAC (srw_ss()) [] THEN MAP_EVERY Q.X_GEN_TAC [‘h’, ‘l2’, ‘l1’] THEN *)
(*   Cases_on ‘l1’ THEN SIMP_TAC (srw_ss() ++ boolSimps.DNF_ss) [MAP_EQ_CONS] THEN *)
(*   METIS_TAC[] *)
Induct >>
asm_simp_tac bool_ss [MAP, APPEND_eq_NIL, APPEND_EQ_CONS]
>- metis_tac[]>>
rpt strip_tac >> asm_simp_tac bool_ss [EQ_IMP_THM] >> rpt strip_tac >>
simp [RIGHT_AND_OVER_OR, EXISTS_OR_THM, MAP, NOT_CONS_NIL, PULL_EXISTS] >>
rw_tac bool_ss [MAP, NOT_CONS_NIL] >>
metis_tac[]

Induct_on ‘l’ >> simp[MAP, APPEND_eq_NIL, APPEND_EQ_CONS]
rpt strip_tac >> asm_simp_tac bool_ss [EQ_IMP_THM]
rpt strip_tac
simp [RIGHT_AND_OVER_OR, EXISTS_OR_THM, MAP, NOT_CONS_NIL, PULL_EXISTS]
rw_tac bool_ss [MAP, NOT_CONS_NIL]
metis_tac[]
(* EQ_TAC *)
(* >-Induct_on ‘l’ *)
(*   >-simp[MAP] *)
(*   >-strip_tac >> strip_tac >> fs[MAP] >> fs[APPEND_EQ_CONS] *)


(* >-Induct_on ‘l1’ *)
(*   >-Induct_on ‘l2’ *)
(*     >-strip_tac >> EXISTS_TAC “[]”>>EXISTS_TAC “[]”>>fs[MAP, APPEND, MAP_EQ_NIL] *)
(*     >-strip_tac >> simp[] >> simp[MAP_EQ_CONS] >> strip_tac >> EXISTS_TAC “[]” >> EXISTS_TAC “x0::t0” >> fs[MAP] *)
(*   >-Induct_on ‘l2’ *)
(*     >- strip_tac >> simp[] >>simp[MAP_EQ_CONS] *)

(* strip_tac >> simp[APPEND] >>simp[MAP_EQ_CONS] >> strip_tac >>  strip_tac >> strip_tac >> EXISTS_TAC “x0::t0” >> EXISTS_TAC “[]” >> fs[MAP] *)

(*  >- strip_tac >> simp[MAP] *)

  (* >-simp[MAP] *)
  (* >-simp[MAP] *)
  (*   simp[APPEND_EQ_CONS] *)
  (*   strip_tac *)
  (*   strip_tac *)
  (*   >-simp[]>>EXISTS_TAC “[]”>>EXISTS_TAC “h::l”>>simp[MAP] *)
  (*   >-fs[]>>simp[MAP_EQ_CONS] *)


>-strip_tac
fs[MAP_APPEND]

QED



Theorem bar:
   EVERY P l ⇔ ∀n. n < LENGTH l ⇒ P (EL n l)
Proof
cheat
QED


Theorem baz:
   [] = [] <=> ∀l. LENGTH l = 0
Proof
cheat
QED

Theorem bak:
   ∀l. LENGTH l = 0
Proof
rw[GSYM baz]
QED




(* qpat_assum ‘∀n. n < SUC (LENGTH l) ⇒ P (EL n (h::l))’ (fn thm => assume_tac (SPEC (“0”) thm)) *)



(* FIRST_ASSUM (fn x => rw[x]) *)

(* qpat_assum ‘∀P. EVERY P l ⇔ ∀n. n < LENGTH l ⇒ P (EL n l)’ (assume_tac o (SPEC (“P:'a->bool”))) *)

(* qpat_assum ‘EVERY P l ⇔ ∀n. n < LENGTH l ⇒ P (EL n l)’ (assume_tac o #1 o EQ_IMP_RULE) *)

(* FIRST_X_ASSUM MATCH_MP_TAC *)

(* FIRST_ASSUM (assume_tac o SPEC “P”) *)

(* FIRST_ASSUM (fn x => MATCH_MP_TAC (#2 (EQ_IMP_RULE x))) *)


(* (fn x => MATCH_MP_TAC (#2 (EQ_IMP_RULE x))) *)

(* rw[] *)


(* rw[GSYM bar] *)

(* fs[] *)

(* FIRST_ASSUM EQ_TAC *)

 (* POP_ASSUM (MP_TAC o (SPEC (“0”))) *)
 (* POP_ASSUM (assume_tac o (SPEC (“0”))) *)

