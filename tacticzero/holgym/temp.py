    def random_play_no_back(self):
        steps = 0
        while self.history:
            target = sample(self.history, 1)[0]
            action = sample(self.action_pool, 1)[0]
            print("Target: {}\nTactic: {}".format(target, action))
            if target[0]:
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
            else:
                goal = target[1]
                d = self.query(goal, action)
            if d != None:
                self.history.remove(target)
                self.history.extend(d)

            steps += 1
            if steps == 50:
                print("Failed")
                return
        print("Proof found in {} steps.".format(steps))

        
    def random_play(self):
        steps = 0
        self.history = [self.history]
        while self.history:
            state = sample(self.history, 1)[0]
            target = sample(state, 1)[0]
            action = sample(self.action_pool, 1)[0]
            print("State: {}\nTarget: {}\nTactic: {}".format(state, target, action))
            if target[0]:
                goal = self.revert(target[0], target[1])
                d = self.query(goal, "rpt strip_tac >> " + action)
            else:
                goal = target[1]
                d = self.query(goal, action)
            steps += 1

            if d != None:
                state.remove(target)
                state.extend(d)
                # an empty state is reached, meaning
                # a proof is found
                if not state: 
                    break
                self.history.append(state)

            if steps == 50:
                print("Failed")
                return
        print("Proof found in {} steps.".format(steps))

    def query(self, goal, tac):
        goal = self.construct_goal(goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")
        tactic = self.construct_tactic(tac)
        self.process.sendline(tactic.encode("utf-8"))
        i = self.process.expect([": proof", "Initial goal proved", "Exception", "error", pexpect.TIMEOUT], timeout=2)
        if i == 0:
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            # print(self.process.readline())
            # print(self.process.readline())
            # print(self.process.readline())            
            self.process.expect("val it =")
            
            # this doesn't seem robust
            self.process.expect([": goal list", ":\r\n"])
            raw = self.process.before.decode("utf-8")
            subgoals = re.sub("“|”","\"", raw)
            subgoals = re.sub("\r\n +"," ", subgoals)
            print("content:{}".format(subgoals))
            exit()
            data = eval(subgoals)
        elif i == 1:
            data = []
        elif i == 4:
            print("Timeout when applying tactic {} to {}.".format(tac, goal))
            # self.process.kill(signal.SIGINT)
            # print(self.process.isalive())
            # print("terminated.")
            # self.process.sendline("drop();".encode("utf-8"))
            # self.process.expect("\r\n>")
            data = "metis_timeout"
            return data
        else:
            # print("Exception raised when applying tactic {} to {}.".format(tac, goal))
            data = None
        
        # clear stack and consume the remaining stdout
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        return data


    def get_states(self):
        # this always assumes that history is not empty
        # get the first goal-assumptions pair in history
        
        # create an empty entry
        unit = self.max_len * [0]
        target = self.history[0]
        state = [self.encode(target[1])]
        for i in range(self.max_assumptions):
            if i < len(target[0]):
                state.append(self.encode(target[0][i]))
            else:
                state.append(unit)

        return torch.FloatTensor(state)

    def encode(self, s):
        r = []
        for c in s:
            if c not in dictionary:
                dictionary[c] = len(dictionary) + 1
            r.append(dictionary[c])
        # pad r with 0's
        r = (r + self.max_len * [0])[:self.max_len]
        return r



        if tactic_pool[tac] == "Induct_on":
            term_probs = []
            candidates = []
            input = torch.cat([state, tac_tensor])
            tokens = g.split()
            variables = []
            for i in tokens:
                if i[0] == "V":
                    variables.append(i)
                    var_tensor = torch.randn(1,1)
                    var_tensor = var_tensor.new_full((1,MAX_LEN), dictionary[i])
                    input = torch.cat([input, var_tensor])
                    candidates.append(input)
            if not variables:
                action = "Induct_on"
                
            candidates = torch.stack(candidates)
            scores = term_net(candidates)
            term_probs = F.softmax(scores, dim=0)
            term_m = Categorical(term_probs.squeeze())
            term = term_m.sample()
            term_probs.append(term_m.log_prob(term))
            v = variables[term]
            v = v[:1] # remove "V"
