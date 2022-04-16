from torch import load
from torch.distributions import Categorical
from model import *
import timeit

start = timeit.default_timer()

# policy = Policy()

# policy = torch.load("policy.ckpt")
policy = torch.load("ac_policy.ckpt")

policy.eval()

def get_states(player, ps, dealer, ace):
    ps = [ps]
    dealer = [dealer]
    ace = [int(ace)]
    s = (player + 21 * [0])[:21] + ps + dealer + ace
    return s

state = torch.tensor(get_states([4,10],14,5,False), dtype=torch.float)

def decide(state):
    t = policy(state)
    m = Categorical(t)
    a = m.sample()
    if a.item():
        return "Hit!"
    else:
        return "Stop!"

print(policy(state))

stop = timeit.default_timer()
print('Time: {}  '.format(stop - start))
