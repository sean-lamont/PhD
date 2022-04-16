structure Popen :>
	  sig
		  (* Parent wants to write, read stdout, or read stdout + stderr *)
		  datatype pipe_type = PIPE_W | PIPE_R | PIPE_RE
		  val popen : string * pipe_type -> Posix.IO.file_desc
		  val pclose : Posix.IO.file_desc -> Posix.Process.exit_status option
		  val pwait : Posix.IO.file_desc -> Posix.Process.exit_status option

	  end =
struct

datatype pipe_type = PIPE_W | PIPE_R | PIPE_RE

type pinfo = { fd : Posix.ProcEnv.file_desc, pid : Posix.Process.pid }

val pids : pinfo list ref = ref []

(* Implements popen(3) *)
fun popen (cmd, t) =
  let val { infd = readfd, outfd = writefd } = Posix.IO.pipe ()
  in case (Posix.Process.fork (), t)
	  of (NONE, t) => (* Child *)
	 (( case t
		 of PIPE_W => Posix.IO.dup2 { old = readfd, new = Posix.FileSys.stdin }
		  | PIPE_R => Posix.IO.dup2 { old = writefd, new = Posix.FileSys.stdout }
		  | PIPE_RE => ( Posix.IO.dup2 { old = writefd, new = Posix.FileSys.stdout }
			   ; Posix.IO.dup2 { old = writefd, new = Posix.FileSys.stderr })
	  ; Posix.IO.close writefd
	  ; Posix.IO.close readfd
	  ; Posix.Process.execp ("/bin/sh", ["sh", "-c", cmd]))
	  handle OS.SysErr (err, _) =>
		 ( print ("Fatal error in child: " ^ err ^ "\n")
		 ; OS.Process.exit OS.Process.failure ))
	   | (SOME pid, t) => (* Parent *)
	 let val fd = case t of PIPE_W => (Posix.IO.close readfd; writefd)
				  | PIPE_R => (Posix.IO.close writefd; readfd)
				  | PIPE_RE => (Posix.IO.close writefd; readfd)
		 val _ = pids := ({ fd = fd, pid = pid } :: !pids)
	 in fd end
  end

(* Implements pclose(3) *)
fun pclose fd =
  case List.partition (fn { fd = f, pid = _ } => f = fd) (!pids)
   of ([], _) => NONE
	| ([{ fd = _, pid = pid }], pids') =>
	  let val _ = pids := pids'
	  val (_, status) = Posix.Process.waitpid (Posix.Process.W_CHILD pid, [])
	  val _ = Posix.IO.close fd
	  in SOME status end
	| _ => raise Bind (* This should be impossible. *)

fun pwait fd =
  case List.partition (fn { fd = f, pid = _ } => f = fd) (!pids)
   of ([], _) => NONE
	| ([{ fd = _, pid = pid }], pids') =>
	  let val _ = pids := pids'
	  val (_, status) = Posix.Process.waitpid (Posix.Process.W_CHILD pid, [])
	  in SOME status end
	| _ => raise Bind (* This should be impossible. *)

end


(* val f = Popen.popen("ls", Popen.PIPE_R); *)
(* val g = Popen.popen("read line; echo $line>/tmp/foo", Popen.PIPE_W); *)
(* val _ = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "Hello World! I was written by g\n")); *)
(* val h = Popen.popen("cat /tmp/foo", Popen.PIPE_R); *)
(* val i = Popen.popen("echo 'to stderr i' 1>&2", Popen.PIPE_R); *)
(* val j = Popen.popen("echo 'to stderr j' 1>&2", Popen.PIPE_RE); *)
(* val _ = app (fn fd => print (Byte.bytesToString (Posix.IO.readVec (fd, 1000)))) [f, h, i, j]; *)
(* val _ = map Popen.pclose [f, g, h, i, j]; *)
(* val _ = OS.Process.exit OS.Process.success; *)

fun string_of_goal (asm,w) =
  let
    val mem = !show_types
    val _   = show_types := false
    val s   =
      (if null asm
         then "[]"
         else "[" ^ String.concatWith "," (map term_to_string asm) ^
              "]")
    val s1 = "(" ^ s ^ "," ^ (term_to_string w) ^ ")"
  in
    show_types := mem;
    s1
  end

(* fun get_action (state) = *)
(*     let *)
(* 	val t1 = Time.now () *)
(* 	val g = Popen.popen("python temp.py", Popen.PIPE_W) *)
(* 	val r1 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "aaaa\n")) *)
(* 	val r2 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "bbb\n")) *)
(* 	val r3 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "terminate\n")) *)
(* 	(* val _ = Posix.Process.wait () *) *)
(* 	val _ = map Popen.pclose [g];  *)
(* 	(* val _ = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "bbbb\n")) *) *)

(* 	val h = Popen.popen("cat action.txt", Popen.PIPE_R) *)
(*     in *)
(* 	(* app (fn fd => print (Byte.bytesToString (Posix.IO.readVec (fd, 1000)))) [f]; *) *)
(* 	let *)
(* 	    val c = Byte.bytesToString (Posix.IO.readVec (h, 1000)) *)
(* 	    val t2 = Time.now () *)
(* 	in *)
(* 	    r1 *)
(* 	end *)
(*     end *)


fun get_action (state) =
    let
	val t1 = Time.now ()
	val g = Popen.popen("python temp.py", Popen.PIPE_W)
	val r1 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "aaaa\n"))
	val r2 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "bbb\n"))
	val r3 = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "terminate\n"))
	(* val _ = Posix.Process.wait () *)
	val _ = map Popen.pclose [g]; 
	(* val _ = Posix.IO.writeVec (g, Word8VectorSlice.full (Byte.stringToBytes "bbbb\n")) *)

	val h = Popen.popen("cat action.txt", Popen.PIPE_R)
    in
	(* app (fn fd => print (Byte.bytesToString (Posix.IO.readVec (fd, 1000)))) [f]; *)
	let
	    val c = Byte.bytesToString (Posix.IO.readVec (h, 1000))
	    val t2 = Time.now ()
	in
	    c
	end
    end

	
(* fun get_action (state) =  *)
(*     let  *)
(* 	val t1 = Time.now () *)
(* 	val python_path = "/home/minchao/anaconda3/bin/python" *)
(* 	val script_path = ["/home/minchao/proj/RL/station/holgym/sml/test.py"]  *)
(*     in *)
(* 	let *)
(* 	    (* val s = Unix.textInstreamOf (Unix.execute (python_path, script_path)) *) *)
(* 	    (* val ss = TextIO.inputLine s *) *)
(* 	    (* val t2 = Time.now () *) *)
(* 	    val s = Unix.binInstreamOf (Unix.execute (python_path, script_path)) *)
(* 	    val i = BinIO.input s *)
(* 	    val t2 = Time.now () *)

(* 	in *)
(* 	    t2 - t1 *)
(* 	    (* case ss of *) *)
(* 	    (* 	SOME t => t *) *)
(* 	    (*  |  NONE => ""	   *) *)
(* 	end *)
(*     end *)



(* fun bare_readl path = *)
(*   let *)
(*     val python_path = "/home/minchao/anaconda3/bin/python" *)
(*     val script_path = ["/home/minchao/proj/RL/station/holgym/sml/temp.py"]  *)
(*     val s = Unix.execute (python_path, script_path) *)
(*   in *)
(*       s; OS.Process.sleep (Time.fromReal 0.4); *)
(* (* Unix.kill (s, Posix.Signal.int); *) *)
(*       let *)
(* 	  val file = TextIO.openIn path *)

(* 	  fun loop file = case TextIO.inputLine file of *)
(* 	      SOME line => line :: loop file *)
(* 	    | NONE => [] *)
(* 	  val l = loop file *)
(*       in *)
(* 	  (TextIO.closeIn file; l) *)
(*       end *)

(*   end *)


(* fun get_action (state) = *)
(*     let *)
(* 	val t1 = Time.now () *)
(* 	val f = Popen.popen("python test.py", Popen.PIPE_R) *)
(*     in *)
(* 	(* app (fn fd => print (Byte.bytesToString (Posix.IO.readVec (fd, 1000)))) [f]; *) *)
(* 	let *)
(* 	    val c = Byte.bytesToString (Posix.IO.readVec (f, 1000)) *)

(* 	    val t2 = Time.now () *)
(* 	in *)
(* 	    t2 - t1 *)
(* 	end *)
(*     end *)

(* fun get_action (state) = *)
(*     let *)
(* 	val t1 = Time.now () *)
(* 	val f = Popen.popen("python temp.py", Popen.PIPE_R) *)
(*     in *)
(* 	(* app (fn fd => print (Byte.bytesToString (Posix.IO.readVec (fd, 1000)))) [f]; *) *)
(* 	let *)
(* 	    val c = Byte.bytesToString (Posix.IO.readVec (f, 1000)) *)

(* 	    val t2 = Time.now () *)
(* 	in *)
(* 	    t2 - t1 *)
(* 	end *)
(*     end *)
