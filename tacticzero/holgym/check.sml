open listTheory;

Theorem ex1 : (∀l1 l2. l1 ++ l2 = l1 ⇔ l2 = []) ∧ (∀l1 l2. l1 ++ l2 = l2 ⇔ l1 = []) ∧ (∀l1 l2. l1 = l1 ++ l2 ⇔ l2 = []) ∧ ∀l1 l2. l2 = l1 ++ l2 ⇔ l1 = []
Proof
fs[]
strip_tac
fs[]
metis_tac[]
QED
