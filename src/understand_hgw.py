import numpy as np
import math
import matplotlib.pyplot as plt

from agent import logsumexp

import torch
from torch.distributions import Normal
import torch.nn.functional as F

def calculate_goal_w_metrics(clusters, w):
    goal_logprobs = clusters.log_prob(goals).sum(dim=-1)
    goal_w_logprobs = goal_logprobs[:, w]
    Hgw = - (goal_w_logprobs.exp() * goal_w_logprobs).sum().item()
    max_goal_w_logprobs = goal_w_logprobs.max()
    return goal_w_logprobs, Hgw, goal_w_logprobs.max(), goal_w_logprobs.sum()

NB_SKILLS = 4
SKILL_ENT = -math.log(1/NB_SKILLS)
STATE_DIM = 2
TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]
goals = torch.Tensor(np.stack(TASKS)).repeat([1, NB_SKILLS])
goals = goals.reshape(len(TASKS), NB_SKILLS, STATE_DIM)

print("CASE OF EQUALLY SPREAD OUT SKILLS")
mus = [(1, 0.6), (0.33, 0.6), (-0.33, 0.6), (-1, 0.6)]
stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
# calculate stuff
print("Leftmost skill: Goal Logprobs, Entropy, Max Goal Logprob, Sum Goal Logprob")
print(calculate_goal_w_metrics(clusters, 0))
print("Center skill: Goal Logprobs, Entropy, Max Goal Logprob, Sum Goal Logprob")
print(calculate_goal_w_metrics(clusters, 1))

# slowly move one of the skills down and see effect
# fig, ax = plt.subplots(3, 1, figsize=(8,24))
# for i in range(100):
#     gap = i/100 * 1.2
#     mus = [(1, 0.6-gap), (0.33, 0.6), (-0.33, 0.6), (-1, 0.6)]
#     stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
#     clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
#     # calculate stuff
#     _, Hgw, max_g_prob, sum_g_prob = calculate_goal_w_metrics(clusters, 0)
#     ax[0].scatter(0.6-gap, Hgw)
#     ax[1].scatter(0.6-gap, max_g_prob)
#     ax[2].scatter(0.6-gap, sum_g_prob)
# plt.show()

# slowly rotate one of the skills
fig, ax = plt.subplots(3, 2, figsize=(8,24))
xs = []
ys = []
Hgws = []
max_g_probs = []
sum_g_probs = []
Hgs = []
for xi in range(40):
    for yi in range(40):
        x = -1 + xi*2/40
        y = -1 + yi*1.6/40
        xs.append(x)
        ys.append(y)
        mus = [(x, y), (0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)]
        stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
        clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
        # calculate stuff
        _, Hgw, max_g_prob, sum_g_prob = calculate_goal_w_metrics(clusters, 0)
        Hgws.append(Hgw)
        max_g_probs.append(max_g_prob)
        sum_g_probs.append(sum_g_prob)
        goal_logprobs = clusters.log_prob(goals).sum(dim=-1)
        log_pg = logsumexp(goal_logprobs, dim=1)
        Hg =  - (log_pg.exp() * (log_pg - SKILL_ENT)).sum().item()
        Hgs.append(Hg)
# Hgws
Hgws = np.array(Hgws)
Hgws = -1*Hgws
# Hgws -= Hgws.min()
# Hgws /= Hgws.max()
ax[0,0].scatter(xs, ys, c=Hgws, cmap='jet')
ax[0,0].plot(*list(zip(*TASKS)), 'x')
ax[0,0].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
# maxs
max_g_probs = np.array(max_g_probs)
# max_g_probs -= max_g_probs.min()
# max_g_probs /= max_g_probs.max()
ax[1,0].scatter(xs, ys, c=max_g_probs, cmap='jet')
ax[1,0].plot(*list(zip(*TASKS)), 'x')
ax[1,0].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
# Hgs
Hgs = np.array(Hgs)
# Hgs -= Hgs.min()
# Hgs /= Hgs.max()
ax[2,0].scatter(xs, ys, c=Hgs , cmap='jet')
ax[2,0].plot(*list(zip(*TASKS)), 'x')
ax[2,0].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
# Hgws + max
ax[0,1].scatter(xs, ys, c=(max_g_probs+Hgws), cmap='jet')
ax[0,1].plot(*list(zip(*TASKS)), 'x')
ax[0,1].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
# Hgws + Hg
ax[1,1].scatter(xs, ys, c=(Hgs+Hgws), cmap='jet')
ax[1,1].plot(*list(zip(*TASKS)), 'x')
ax[1,1].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
# Hgws + Hg + max
ax[2,1].scatter(xs, ys, c=(Hgs+max_g_probs), cmap='jet')
ax[2,1].plot(*list(zip(*TASKS)), 'x')
ax[2,1].plot(*list(zip(*[(0.66, 0.6), (-0.33, 0.6), (-0.66, 0.6)])), 'o')
plt.show()
