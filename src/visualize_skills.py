import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

from boxenv import *
from agent import *

import torch

NB_SKILLS = 6
COND = 'OUR'
STATE_DIM = 2
SEED = 456
# TASKS = [(0.5, 0.8), (-0.5,0.8)]
TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]
# TASKS = [(0.8, 0.8), (0.8, -0.8), (-0.8, -0.8), (-0.8, 0.8)]
# TASKS = [(0.8, 0.8), (0.3, 0.8),  (-0.3, 0.8),  (-0.8, 0.8)]
# TASKS = [(0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (-0.1, 0.1)]

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

policy_function = GaussianPolicyFunction(STATE_DIM + NB_SKILLS, 2)
path = '../models/{}/{}/best_reinforce_seed{}.pth'.format(COND, len(TASKS), SEED)
print("visualizing policy from ", path)
policy_function.load_state_dict(torch.load(path)['policy_func'])
policy = GaussianPolicy(policy_function)
box = BoxWorld()
# run an episode with each value of w
fig, ax = plt.subplots(figsize=(8,8))
# color = plt.cm.tab10(np.linspace(0, 1, NB_SKILLS))
color = ['r', 'b', 'k', 'g', 'y', 'm']
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_title("Skills")
for w in range(NB_SKILLS):
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    for i in range(3):
        # reset environment
        s = box.reset()
        done = False
        states = [s]
        while not done:
            s = torch.Tensor(np.concatenate((s, w_onehot)))
            # get action and logprobs
            mu, sigma = policy_function(s.unsqueeze(0))
            unscaled_action, logprob, entropy = policy.sample(s.unsqueeze(0))
            # step the environment
            unscaled_action = unscaled_action.squeeze().detach().numpy()
            action = box.scale_action(unscaled_action)
            print("state: ", s.numpy()[:2],
                  "mean: ", mu.detach().numpy(),
                  "sigma: ", sigma.detach().numpy(),
                  "action: ", action)
            s, r, done = box.step(action)
            states.append(s)
        state_x, state_y = zip(*states)
        ax.scatter(state_x, state_y, color=color[w], label='skill:'+str(w))

for i, t in enumerate(TASKS):
    plt.text(*t, 'task:'+str(i))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('../imgs/{}/{}/skills{}.png'.format(COND, len(TASKS), SEED))
