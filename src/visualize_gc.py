import numpy as np
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

from boxenv import *
from agent import *

COND = 'GC'
STATE_DIM = 2
DIM = STATE_DIM
SEED = 123

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# initial training task list
TASKS = [(0.5, 0.8), (-0.5, 0.8)]
# TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]

policy_function = GaussianPolicyFunction(2 * STATE_DIM, 2)
policy = GaussianPolicy()

# create a stationary test task list
rng = np.random.default_rng(1)
TEST_TASKS = rng.random((10,2))
TEST_TASKS = TEST_TASKS * 2 - 1
# TEST_TASKS[:,1] = 0.8
NB_TASKS = TEST_TASKS.shape[0]

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_title("Goal-conditioned")
color = plt.cm.RdYlBu(np.linspace(0, 1, NB_TASKS))

for i, t in enumerate(TEST_TASKS):
    plt.text(*t, 'task:'+str(i), c=color[i])


box = BoxWorld()

policy_function.load_state_dict(torch.load(
    'models/{}/{}/policy_seed{}.pth'.format(COND, len(TASKS), SEED)))
for gid in range(NB_TASKS):
    s = box.reset()
    done = False
    states = []
    while not done:
        states.append(s)
        s = torch.Tensor(np.concatenate((s, TEST_TASKS[gid])))
        # get action and logprobs
        mu, sigma = policy_function(s)
        unscaled_action, logprob, entropy = policy.forward(mu, sigma)
        # scale action to environment limits
        a = box.scale_action(unscaled_action.detach().numpy())
        # step the environment
        s, _, done = box.step(a)
    state_x, state_y = zip(*states)
    ax.scatter(state_x, state_y, color=color[gid], label='TASK:'+str(gid))


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('imgs/{}/{}/{}.png'.format(COND, len(TASKS), SEED))
