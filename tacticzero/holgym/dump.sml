(* open integerTheory *)
(* open sortingTheory *)
(* open realTheory *)
(* open bagTheory *)
open probabilityTheory
(* open topologyTheory *)
(* open probabilityTheory *)
val default_pt = get_term_printer();

fun generate_dependencies t =
    case parents t of
    [] => []
   |  hd :: tl => hd :: List.concat (map generate_dependencies (hd::tl))

fun isolate [] = []
  | isolate (x::xs) = x::isolate(List.filter (fn y => y <> x) xs)

fun extract_theories t = t :: isolate (generate_dependencies t)

fun pt t =
    case dest_term t of
        VAR (nm, _) => "V" ^ nm
      | CONST {Name,Thy,...} => "C" ^ "$" ^ Thy ^ "$ " ^ Name
      | COMB(t1,t2) => "@ " ^ pt t1 ^ " " ^ pt t2
      | LAMB(v,b) => "| " ^ pt v ^ " " ^ pt b;

(* val _ = set_term_printer (HOLPP.add_string o pt) *)

(* fun dump (th : string * thm) = print ("\n" ^ thm_to_string (#2 th) ^ "\n"); *)

fun dump (th : string * thm) = print ("\n" ^ (#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ ";" ^ "\n");

fun dump_string (th : string * thm) = ((#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ ";" ^ "\n");

(* fun dump_def_string (th : string * thm) = ((#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ "大" ^ "def" ^ ";" ^ "\n"); *)

(* val _ = set_term_printer default_pt; *)

fun dump_def_string (th : string * thm) = 
    let 
	val _ = set_term_printer (HOLPP.add_string o pt)
	val _ = set_trace "types" 1
	val s1 = (#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ "大" ^ "def" ^ "大" in
	let 
	    val _ = set_term_printer default_pt in
	    s1 ^ thm_to_string (#2 th) ^ "小" ^ "\n" 
	end 
    end;

(* fun dump_thm_string (th : string * thm) = ((#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ "大" ^ "thm" ^ ";" ^ "\n"); *)

fun dump_thm_string (th : string * thm) = 
    let 
	val _ = set_term_printer (HOLPP.add_string o pt)
	val _ = set_trace "types" 1
	val s1 = (#1 th) ^ "大" ^ thm_to_string (#2 th) ^ "大" ^ int_to_string ((Dep.depnumber_of o Dep.depid_of o Tag.dep_of o Thm.tag) (#2 th)) ^ "大" ^ "thm" ^ "大" in
	let 
	    val _ = set_term_printer default_pt in
	    s1 ^ thm_to_string (#2 th) ^ "小" ^ "\n" 
	end 
    end;

(* fun dump_name (th : string * thm) = print ("\n" ^ (#1 th) ^ "\n"); *)

(* List.app dump (definitions "list"); *)

(* min and basicSize are empty *)
val core_theories = ["ConseqConv", "quantHeuristics", "patternMatches", "ind_type", "while", "one", "sum", "option", "pair", "combin", "sat", "normalForms", "relation", "min", "bool", "marker", "num", "prim_rec", "arithmetic", "numeral", "basicSize", "numpair", "pred_set", "list", "rich_list", "indexedLists"];

val up_to_probability = extract_theories "probability";
(* val additional_theories = ["integer", "sorting", "real", "bag"]; *)

fun length_of_theories ts = map (fn x => length (theorems x)) ts

val additional_theories = [];


(* map (fn theory => List.app dump (definitions theory)) core_theories; *)

(* map (fn theory => List.app dump (theorems theory)) core_theories; *)

(* map (fn theory => List.app dump_string (definitions theory @ theorems theory)) (core_theories @ additional_theories); *)

(* val out = map (fn theory => List.map dump_string (definitions theory @ theorems theory)) (core_theories @ additional_theories) *)



(* map (fn theory => List.app dump (theorems theory)) core_theories; *)

(* List.app dump (theorems "list"); *)

(* ~/HOL/bin/hol < dump.sml > defs.txt *)

(* ~/HOL/bin/hol < dump.sml > thms.txt *)

fun writeFile filename content =
    let val fd = TextIO.openOut filename
        val _ = TextIO.output (fd, content) handle e => (TextIO.closeOut fd; raise e)
        val _ = TextIO.closeOut fd
    in () end

fun dump_write filename =
    let val outll = (map (fn theory => List.map (fn s => "\n" ^ theory ^ "大" ^ s) (List.map dump_string (definitions theory @ theorems theory))) (up_to_probability @ additional_theories))
        val outl = List.concat outll
        val out = String.concat outl
    in writeFile filename out end

fun dump_split_write filename =
    let val outll = (map (fn theory => List.map (fn s => "\n" ^ theory ^ "大" ^ s) (List.map dump_def_string (definitions theory) @ List.map dump_thm_string (theorems theory))) (up_to_probability @ additional_theories))
        val outl = List.concat outll
        val out = String.concat outl
    in writeFile filename out end

(* val _ = dump_write "raw_data_defs_thms.txt"; *)

(* val _ = dump_split_write "raw_data.txt"; *)

(* val _ = dump_split_write "typed_raw_data.txt"; *)

val _ = dump_split_write "include_probability.txt";


(* val up_to_probability = *)
(*    ["martingale", "lebesgue", "borel", "real_borel", "sigma_algebra", *)
(*     "util_prob", "fcp", "indexedLists", "rich_list", "list", "pred_set", *)
(*     "numpair", "basicSize", "numeral", "arithmetic", "prim_rec", "num", *)
(*     "marker", "bool", "min", "normalForms", "relation", "sum", "pair", "sat", *)
(*     "while", "bit", "logroot", "transc", "powser", "lim", "seq", "nets", *)
(*     "metric", "real", "realax", "hreal", "hrat", "quotient_sum", "quotient", *)
(*     "res_quan", "product", "iterate", "cardinal", "wellorder", *)
(*     "set_relation", "derivative", "real_topology"]: string list *)
