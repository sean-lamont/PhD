


## Policy Gradient:
Basic form is on policy, sample (roll out) policy and collect state, reward information. Can then calculate gradient of policy criterion (sum over log gradientsweighted by rewards, very similar to maximum likelihood loss for e.g. imitation learning). Update policy and repeat. Can be expensive, high variance. Can improve variance with use of baseline (e.g. subtract average of rewards in gradient calculation) and using causality i.e. only count rewards to go rather than including past rewards.

Sampling and taking mean is unbiased estimate 

Off-policy can be done with importance sampling, using another distribution to update parameters of policy network. Good for example with previous demonstration data (could be used with previous proofs for example)

## Actor Critic

Policy gradient doesn't consider expectation in the calculation of reward?

Given true Q value (expectation of future reward under policy and transition distributions) can add this to policy gradient to reduce variance

Q function defined as sum of expected rewards from current time step to end given state and action at, st. Value is expected value of state following policy. Advantage funciton is defined as difference between Q function and value function (notion of how much better/worse an action is on average than the policy in a state). Want to improve algorithm based on advantage. 

In reality, no access to true Q function or value function and so must be approximated. E.g. standard policy gradient taking sample run and averaging reward. 

 Use function approximators (NNs) to approximate Q, V and A (advantage)

Policy evaluation refers to estimating value function/Q function for policy without changing policy.

Monte carlo evaluation: sum reward to go from trajectory in state. Can get better estimate by taking multiple trajectories (in general not possible). Since we are evaluating trajectories from multiple states, over a long time the function approximator should perform well by amoritising the value of similar states. 

Specifically, training data is a set of tuples, with state and trajectory, and sum of rewards from that trajectory. Then minimise square loss as standard for neural network optimisations. Can replace with biased but reduced variance, by taking previous value function of state and replacing that with sum of rewards in original training set (bootstrapped estimate). 

batch Actor critic algorithm:

Sample batch states and actions from policy (run it through)

fit value function to sampled rewards

Evaluate advantage

calculate gradient of optimisation objective (log gradient of policy scaled by advantage) 

Update policy weights

repeat  

Online:

Similar, but one time step at a time and with a discount factor

In practice is done in parallel to collect the transition information, either synchronously or asynch wrt parameter updates 

### Architecture:

Two network design for actor and critic networks. Is simple and stable, but no shared features.

shared network desing uses same network with two outputs for value and action.
Can be unstable as 2 loss functions which are quite different to optimise the network 

### Critic as baseline

For actor critic, use of bootstrapped value function is biased, compared to unbiased but higher variance policy gradient with baseline. Can combine the two by making the baseline for policy gradient the value function for the state, which is unbiased (as it is a baseline) and lower variance from the use of the value function. 

### Control variate 

Action dependent baseline, i.e. Q function rather than Value function 


### Eligibility traces and n- step returns

combines actor critic with monte carlo advantages to contril bias variance tradeoff with how many steps in the future to use MC loss (n steps)

Generalised advantage estimation:

Run n step return for all n, sum the values weighted with an exponential decay function.

## Value function based only 

Given a value function, set a policy as that which takes the arg max action of the advantage given a state

Given a fully observed MDP, there exists an optimal deterministic policy

### Policy iteration

Evaluate advantage, set policy to be arg max of advantage (calculated as reward of action in state plus discounted expected reward in new state minus value of current state

### Dynamic Programming

If transition probabilities are known, construct a tabular representation of state X actions, update value function with reward + discounted expected value of nxt state 

### Fitted value iteration

How to represent V(s)? Table doesn't scale (curse of dimentionality)

Neural net mapping from states to value 

loss function with MSE, with targets being the max Q value for a state 

fitted value iteration: find max action for next value, update network repeat. Not possible often as calculating max Q value needs you to know outcome for different actions. Use Q value network instead.. 


### Fitted Q Iteration 

1. Collect dataset of state, action, next state, reward using *some* policy. Data can be off policy. 

2. Set the targets y as r(s, a) + discount * max a` Q(s`, a`)

3. Update theta for Q network based on target y and Q(s, a) 

Optimises bellman error of Q network   

Q Learning is just online version of fitted Q iteration 


Tabular Q learning converges using contraction with fixed point, but can show fitted value does not converge in general. 



## Practical algorithms for Q learning 

Problems with fitted Q iteration algorithms: 

- Network update is not strictly gradient descent, since the targets are dependend on the value of Q, which is not considered in the equation for updating the parameters

- Samples are correlated since in the outer loop they are redrawn from the updated network from the previous sample.  When running naive Q learning the network tends to overfit local regions with similar states as it progresses. Can be mitigated by parallelising the training, so each worker thread updates parameters based on its own samples, without seeing the others. With enough workers can mitigate the problem.

### Replay buffers

Since Q learning is off policy, can first generate samples (possibly randomly) and add these to a replay buffer. Then draw the samples from the buffer instead of generating them with the updated policy, thereby removing correlation between batches. Also can do multiple samples per batch as opposed to online Q learning giving lower variance gradient.

Need to peridodically feed the buffer, e.g. with an epsilon greedy policy 


### Target network

Attempts to address gradient problem in online q learning. 

Same algorithm as with replay buffer Q learning, but another outer loop step to set 'target' network parameters as those from previous batch. Then the new target network is used to compute the targets in the gradient update step, so the targets aren't changing with each inner loop step (since without target network, every gradient step will be with respect to a slightly different target from the updated network of the previous step) 
This gives rise to classic DQN:

1. take action, observe s,a,s`,r. Add to buffer B
2. sample mini batch from B uniformly
3. compute targets with target network, as reward + discounted max q value of target network
4. update parameters of phi (non target q network) with gradient step
5. update target network with copy of q network every N steps

Can change algorithm to update target network every iteration, but with a dampened update so it moves slowly. This prevents uneven behaviour in updating batches, where later samples are updating based on older parameters than earlier samples. 

Strategies for sampling e.g. prioritised replay sampling, large tends to be better even though recent samples may have more reward. Often FIFO eviction is used.

## Multi step returns

Can, in a similar way to actor critic, implement n step return by summing rewards up to n, then taking discounted Q value estimate. Gives balance and choice for bias variance trade off, standard q learning maximised bias to minimise variance. Problem is that this is done for on policy where the reward sum only makes sense if it is for the current policy. Can ignore (can work well), cut the trace N where actions are off policy (works better when only a few actions). Can also do importance sampling.

### Continuous Actions

Policy usually given by argmax in deterministic case, as with target value. Can use optimisation algorithms to find action, but slow. Instead use stochastic optimisation since action dimensionality usually low. 

### Stochastic optimisation Q learning

sample actions from distribution over actions (e.g. uniform), then take maximum. 
Can more accurately do e.g. cross entropy method, cma-es 

Other option: use function class for Q which is easy to maximise for a. E.g. Normalized advantage functions (NAF). Loses reprenentational power however. 

Option 3:

Learn approximate maximizer, eg. DDPG.

Sometimes referred to as deterministic actor critic but really approximate Q learning.  

Learn network mu which approximates argmax Q(s,a), ie. Q(s, argmax mu(s)) then maximise mu with equivalent argmax mu (Q, mu(s))

DDPG:

1. Take action, observe transition, add to B
2. Sample mini batch from B uniformly
3. Compute target with Q(s, mu(s)) using target nets Q and mu
4. Update Q as usual with gradient step in difference of target and Q value
5. update parameters for mu using chain rule dQ/dmu = dQ/da da/dmu backpropping through Q and mu
6. Update target networks for Q and mu using e.g. polyak averaging or every n steps 

Practical tips:

Q learning takes care to stablise
- Test on easy, reliable tasks to check implementation 

Large replay buffers improve stablility (looks more like fitted Q iteration)

Takes time , may not be better than random for a long time 

Start with high exploration

Advanced tips:

Bellman error gradients can be very large, so gradient clipping is helpful or Huber loss acts similarly 

Double Q learning helps a lot in practice, simple no downsides. (practically involves using target network instead of standard Q network to find arg max when calculating the target

N step return helps a lot but has downsides

Schedule exploration high to low and learning rates high to low works well, as well as Adam (RMSprop in practice too, arguably more principled as momentum may not be as relevant in the RL scenario.

Multiple random seeds 

## Policy gradient cont

Regular gradient ascent has wrong constraint after firsts order approximation 

Taylor expansion of KL divergence gives natural gradient 

Natural policy gradient uses the natural gradient (adds inverse fisher information matrix as product with standard gradient ascent) 

Trust region policy optimization also uses this, with specific step size 

##  Model based RL

Previous methods looked only at model free RL, which do not consider planning under the (true or learned) model of the environment dynamics. 

Either closed or open loop problems. In closed loop, one action taken at a time which gives a state from environment. Open loop takes a sequence of actions for a given state before changing policy. 

Again, objective is to maximise expected reward. Can follow stochastic optimisation (or random shooting where you try a sequence of actions at random and evaluate the outcome, taking the best). Stochastic optimisation would be e.g. cross entropy method which samples from the environment intially based on uniform distribution over actions. Then resamples with a new distribution fitted with best samples from previous iteration. Continue process until convergence, which is guaranteed given (unreasonable) assumptions on the number of samples. Another method which does this with analogue to momentum. 

Taking into account the sequential nature of the problem, can also do MCTS:

1. Find leaf using some tree policy over s1
2. Evaluate the leaf using some RL policy (e.g. random) and collect the reward
3. update values in tree between leaf and s1

Tree policy commonly taken as UCT:

If leaf st not fully expanded, choose new action 
Else choose child with best score(st+1)

Score gives tradeoff between value and visitation count, and visitation count of parent. Each node has value as sum of all paths below, and N as count of expansions at that node. I.e. value Q divided by N gives average reward for paths expanded from the node


For situations where transition function is a linear function of state and action, and cost is quadratic function of state and action we have a linear quadratic problem. These can be solved directly, by expressing the cost of the final state, taking the best action (take gradient wrt action then set to 0), then substituting this action to the cost of the final state. Get the equivalent of a value function for the final state. Can then repeat similar process for state t-1, taking the action which maximises the cost with the value for the final state substituted in. Then repeat iteratively to get closed form expression for the value function. Then from the first state, substitute in the state and take the best action from the closed form expression, and continue to the final state, giving a plan. 

For non linear functions, can simply take the first order taylor approximation for the transition, and second order for the cost and apply LQ optimisation as before. Similar to newtons method, and can be made exactly the same by taking second order approximation for both cost and transition. In practice, only standard LQ is done due to performance considerations. Can be used in settings such as MuJoCu with differentiable costs and transitions, and works very well considering nothing is being learned.  
