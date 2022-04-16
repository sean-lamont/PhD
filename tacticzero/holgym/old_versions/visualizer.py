from torch.utils.tensorboard import SummaryWriter
from exp_env import *
from exp_model import *
from exp_config import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

ARG_LEN = 5

GAMES = TEST_GOALS

num_episode = 4000

learning_rate = 1e-5

tac_rate = 1e-5
arg_rate = 1e-5
term_rate = 1e-5

gamma = 0.99 # 0.9

# for entropy regularization
trade_off = 1e-2

tac_net = TacPolicy(len(tactic_pool))

arg_net = ArgPolicy(MAX_LEN, MAX_LEN)

term_net = TermPolicy()

tac_net = tac_net.to(device)
arg_net = arg_net.to(device)
term_net = term_net.to(device)

optimizer_tac = torch.optim.RMSprop(list(tac_net.parameters()), lr=tac_rate)

optimizer_arg = torch.optim.RMSprop(list(arg_net.parameters()), lr=arg_rate)

optimizer_term = torch.optim.RMSprop(list(term_net.parameters()), lr=term_rate)

writer = SummaryWriter('runs/experiment_1')

c_tac = torch.randn(1, 8, 4, 128)

c_arg = torch.randn(1, 8, 8, 128)
h1 = torch.randn(1,1,128)
h2 = torch.randn(1,1,128)
h = (h1,h2)
p = torch.randn(1,128)

# writer.add_graph(tac_net, c_tac)
writer.add_graph(arg_net, (p, c_arg, h))

writer.close()
