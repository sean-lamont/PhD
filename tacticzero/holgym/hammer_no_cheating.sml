use "json.sml";
load "holyHammer";
open holyHammer;


val infile = "test_dep_dict.json" ;

fun readjson (infile : string) = let

  val ins = TextIO.openIn infile
in
  case TextIO.inputLine ins of
      SOME line => Json.parse line
    | NONE      => Json.ERROR "Nothing"
end ;


(* fun readfirstkey (infile : string) = let *)
(*     val out = readjson infile *)
(* in *)
(*     case out of *)
(*         Json.OK sth => sth *)
(*       | Json.ERROR s => Json.NULL *)
(* end; *)

fun readfirstkey (infile : string) = let
    val out = readjson infile
in
    case out of
        Json.OK (Json.OBJECT ((s,j)::tl)) => s
     |  Json.ERROR s => ""
     | _ => ""
end;

fun string_to_term (s : string) = Parse.Term [QUOTE s]

fun value_to_term (p : string * Json.json) = string_to_term (#1 p)

fun json_string_to_string js =
    case js of
        Json.STRING s => s
      | _ => "ERROR"

fun value_to_input (p : string * Json.json) =
    case #2 p of
        Json.ARRAY l => (string_to_term (#1 p), map json_string_to_string l)
     | _ => (string_to_term (#1 p), ["ERROR"])

fun allterms (infile : string) = let
    val out = readjson infile
in
    case out of
        Json.OK (Json.OBJECT l) => map value_to_term l
     |  Json.ERROR s => []
     | _ => []
end;

fun firstpair (infile : string) = let
    val out = readjson infile
in
    case out of
        Json.OK (Json.OBJECT l) => [value_to_input (List.nth (l,0))]
     |  Json.ERROR s => []
     | _ => []
end;

fun construct_hh_input (p : term * string list) = hh_pb [holyHammer.Eprover] (#2 p) ([],#1 p)

fun gen_data (infile : string) = let
    val out = readjson infile
in
    case out of
        Json.OK (Json.OBJECT l) => map value_to_input l
      | _ => []
end;

(* fun run_hh_pb (data : (term * string list) list) = *)
(*     case data of *)
(*         [] => hh_pb [holyHammer.Vampire] [] ([], “¬T ⇔ F”) *)
(*       | hd :: tl => *)
(*         let *)
(*             val v = hh_pb [holyHammer.Vampire] (#2 hd) ([], (#1 hd)) *)
(*                     handle HOL_ERR msg => *)
(*                            let val _ = print "proof failed.\n" *)
(*                                val _ = print (#message msg) *)
(*                            in *)
(*                                hh_pb [holyHammer.Vampire] [] ([], “¬T ⇔ F”) *)
(*                            end *)
(*         in *)
(*             run_hh_pb tl *)
(*         end *)

fun premise_selection goal n = mlNearestNeighbor.thmknn_wdep (smlRedirect.hidef mlThmData.create_thmdata ()) n (mlFeature.fea_of_goal true goal)

(* Filter out elements in premises that are not in pool *)
fun premise_filter premises pool =
    case premises of
        [] => []
     | hd::tl => if List.exists (fn x => x = hd) pool then hd::premise_filter tl pool else premise_filter tl pool

fun gen_premises goal n pool = premise_filter (premise_selection goal n) pool

(* Filter out elements of premises that are not in pool until m of them remain *)
fun premise_filter_better premises pool m =
    case premises of
        [] => []
      | hd::tl => if m = 0 then [] else
                  if List.exists (fn x => x = hd) pool then hd::premise_filter_better tl pool (m-1) else
                  premise_filter_better tl pool m

fun gen_premises_better goal n pool m = premise_filter_better (premise_selection goal n) pool m;

fun run_hh_pb (data : (term * string list) list) (prover_list)=
    case data of
        [] => hh_pb prover_list [] ([], “¬T ⇔ F”)
      | hd :: tl =>
        let
            val _ = print_term ((#1 hd))
            val v = hh_pb prover_list (#2 hd) ([], (#1 hd))
                    handle HOL_ERR msg =>
                           let
                               val _ = print "proof failed.\n"
                               val _ = print "Failed theorem:\n"
                               val _ = print_term ((#1 hd))
                               val _ = print "End printing\n"
                               val _ = print (#message msg ^ "\n")
                          in
                               hh_pb prover_list [] ([], “¬T ⇔ F”)
                           end
        in
            run_hh_pb tl prover_list
        end;

holyHammer.dep_flag := true;


(* fun run_hh_pb (data : (term * string list) list) prover_list premise_num = *)
(*     case data of *)
(*         [] => hh_pb prover_list [] ([], “¬T ⇔ F”) *)
(*       | hd :: tl => *)
(*         let *)
(*             val premises = gen_premises ([], (#1 hd)) premise_num (#2 hd) *)
(*             val v = hh_pb prover_list premises ([], (#1 hd)) *)
(*                     handle HOL_ERR msg => *)
(*                            let *)
(*                                val _ = print "proof failed.\n" *)
(*                                val _ = print "Failed theorem:\n" *)
(*                                val _ = print_term ((#1 hd)) *)
(*                                val _ = print "End printing\n" *)
(*                                val _ = print (#message msg ^ "\n") *)
(*                           in *)
(*                                hh_pb prover_list [] ([], “¬T ⇔ F”) *)
(*                            end *)
(*         in *)
(*             run_hh_pb tl prover_list premise_num *)
(*         end *)

fun run_hh_pb_better (data : (term * string list) list) prover_list premise_num remain_num =
    case data of
        [] => hh_pb prover_list [] ([], “¬T ⇔ F”)
      | hd :: tl =>
        let
            val _ = print_term ((#1 hd))
            val premises = gen_premises_better ([], (#1 hd)) premise_num (#2 hd) remain_num
            val v = hh_pb prover_list premises ([], (#1 hd))
                    handle HOL_ERR msg =>
                           let
                               val _ = print "proof failed.\n"
                               val _ = print "Failed theorem:\n"
                               val _ = print_term ((#1 hd))
                               val _ = print "End printing\n"
                               val _ = print (#message msg ^ "\n")
                           in
                               hh_pb prover_list [] ([], “¬T ⇔ F”)
                           end
        in
            run_hh_pb_better tl prover_list premise_num remain_num
        end

(* fun run_hammer data = *)
(*     case data of *)
(*         [] => holyhammer “¬T ⇔ F” *)
(*       | hd :: tl =>  *)
(* let  *)
(* val v = holyhammer hd  *)
(* handle HOL_ERR msg => *)
(* 	    (* print "proof failed.\n" *) *)
(* let val _ = print "proof failed.\n" in *)
(* 	    holyhammer “¬T ⇔ F” *)
(*  end *)
(* in *)
(* run_hammer tl *)
(* end *)
