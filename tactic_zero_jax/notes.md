# Notes on Minchao's implementation

## Policy networks
### Term network.
 The paper only specifies 3 networks for the policy, and the seq2seq network for the goal encoding. Context net looks to give goal values, arg net gives arguments and tac net gives tactics. Looks like term net (from train module) is only used with argument network, when the tactic is Induct\_on i.e. it is used to find the best term to use on induct\_on. The input to term\_net is given by a matrix of candidate terms, each one being comprised of "input" and "term\_tensor". Input is a combination of target\_representation and tac\_tensor, which afaik are identifiers for the current proof target theorem and the selected tactic (in this case induct\_on). Seems to make sense as the network should find the value of the term conditioned on the current target goal and tactic, as per the paper 

### Context Network
Network for learning goal values. Network itself maps from goal to value, however the implementation for training calls the network with every goal in the history (given by representations from gather\_encoded\_content), then pieces them together by the fringe with split\_by\_fringe, and then takes the product for each fringe, softmaxes and then samples. 

### Tactic Network
Maps from goal to tactic. Implemented in training with input given as described, with the goal being the first goal for the selected tactic from the context network. i.e. representations[target\_context]
Note that the context will include the assumptions. As such, the code states the size of the input will be  (1, max\_contexts, max\_assumptions+1, max\_len). 

### Argument Network 

Trivial for no argument, tactic is just assigned as itself. If induct\_on then term is sampled from term network as described above. Otherwise a list of theorems will be given as arguments, from the network with the LSTM. The initial hidden state of the LSTM will be set as the target goal (target\_representation which includes the context/assumptions). Input to the LSTM is initialised as the tactic $t$.
Candidate matrix is given by the fact\_pool of known current theorems, concatenated with the hidden state. Each theorem d in the fact\_pool is concatenated with the target representation (g), the tac\_tensor (t)  and the hidden state, and the candidate matrix is fed to a FFNN (not the LSTM). Assume this is done so the network can have the context for *each* candidate theorem, as in the equation for the LSTM policy in the paper, where each argument is conditioned on the tactic, goal, and previous arguments (which are encapsulated by the LSTM hidden state). I.e. the candidate matrix includes information for t, g, and previous arguments embedded in each theorem. 
 

Q: Not sure why the hidden state has 2 vectors in a tuple? 
A: Needed for torch syntax. (1,1,MAX\_LEN) gives us a single batch element, with 1 recurrent layer, with hidden dimension size MAX\_LEN
Q: Where does MAX\_CONTEXT come in for the shape of the inputs
Q: More precise detail on how the shapes for the network are derived. 


## Misc
- Argument for tactics, either single theorem, list of theorems, or single term for induct\_on, or no arguments. Note that a theorem is an expression and as such is encoded by the autoencoder? 
- Dimensions for candidates? Meant to be arguments, which can be e.g. HOL term

- For JAX, function will implicitly support batches as the first dimension of a tensor which is passed in. Need to be careful with this though... i.e. assert batch dims are always returned
Env class (sets\_env):

- Context in old repository is given by an assumption, goal pair. target = context["polished"] contains the goal and assumptions, with goal being a singleton and assumptions being a list

- Encoding just looks at a global dictionary of tokens, and gives them a number up to MAX\_LEN. Encoding of a string is then given by the vector of token encodings

- Doesn't seem to be done in batches in the traditional sense. Each 'batch' is an episode, where the gradient is updated from the sum of rewards for the episode. The actual inputs to the network aren't passed in as this. Some of the networks still have multiple arguments e.g. context net will be applied over a list of contexts, arg net is over candidates in some cases. But if one argument is passed in then batch norm doesn't make sense? Could use replay buffer and update in batch instead, then have separate module for the forward pass when collecting data, but would require off policy algorithm. 

## Global variables:


MAX\_LEN = 128, maximum length of an encoding in env class
MAX\_ASSUMPTIONS = 3
MAX\_CONTEXTS = 8

