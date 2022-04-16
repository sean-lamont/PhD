fun mg (goal, tac) = 
    let 
	val _ = g goal
	val r = e tac 
	(* val tg = top_goals(); *)
    in
	r
    end
