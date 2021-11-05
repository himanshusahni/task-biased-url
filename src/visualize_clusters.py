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
TRAJ_LEN = 25
SEED = 123

TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]

policy_function = GaussianPolicyFunction(STATE_DIM + NB_SKILLS, 2)
path = 'models/{}/{}/'.format(COND, len(TASKS))
print("visualizing clusters from ", path)
policy_function.load_state_dict(torch.load(path+'policy_seed{}.pth'.format(SEED)))
policy = GaussianPolicy(policy_function)
clusters = torch.load(path+'clusters_seed{}.pth'.format(SEED))
box = BoxWorld()

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# generate trajectories using each skill
samples = []
for w in range(NB_SKILLS):
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    # reset environment
    s = box.reset()
    done = False
    states = []
    while not done:
        states.append(s)
        s = torch.Tensor(np.concatenate((s, w_onehot)))
        # get action and logprobs
        unscaled_action, logprob, entropy = policy.forward(*policy_function(s))
        # step the environment
        action = box.scale_action(unscaled_action.detach().numpy())
        s, r, done = box.step(action)
    samples.append(np.stack(states))
samples = torch.Tensor(np.stack(samples))
trajs = samples.reshape(-1, STATE_DIM) # (BATCH_SIZE*TRAJ_LEN, 2)
fig, axes = plt.subplots(NB_SKILLS, NB_SKILLS + 1, figsize=(4*NB_SKILLS, 3*NB_SKILLS+4))
for i in range(NB_SKILLS):
    traj = trajs[i*TRAJ_LEN:(i+1)*TRAJ_LEN]
    axes[i,0].scatter(traj[:,0], traj[:,1])
    axes[i,0].set_xlim([-1,1])
    axes[i,0].set_ylim([-1,1])
    # get cluster probabilities
    posteriors = clusters.log_prob(traj.unsqueeze(1).repeat([1, NB_SKILLS, 1]))
    posteriors = posteriors - logsumexp(posteriors, dim=1)
    posteriors = posteriors.exp()
    Nk = posteriors.mean(dim=0).numpy() * 100
    print(Nk)
    for j in range(1,NB_SKILLS+1):
        axes[i,j].scatter(traj[:,0],
                          traj[:,1],
                          vmin=0,
                          vmax=1,
                          c=posteriors[:, j-1],
                          cmap='Reds')

        axes[i,j].set_title('Skill: ' + str(j-1) + ' total: {:.2f}'.format(Nk[j-1]))
        axes[i,j].set_xlim([-1,1])
        axes[i,j].set_ylim([-1,1])
plt.savefig('imgs/{}/{}/clusters{}.png'.format(COND, len(TASKS), SEED))
