import numpy as np
import math
import matplotlib.pyplot as plt

from agent import logsumexp

import torch
from torch.distributions import Normal

def calculate_goal_metrics(clusters):
    goal_logprobs = clusters.log_prob(goals).sum(dim=-1)
    log_pg = logsumexp(goal_logprobs, dim=1)
    pg = log_pg.exp().sum().item()
    Hg =  - (log_pg.exp() * (log_pg - SKILL_ENT)).sum().item()
    return log_pg, pg, Hg

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
log_pg, pg, Hg = calculate_goal_metrics(clusters)
print("log p(g) for different goals ", log_pg.t())
print("overall p(g) ", pg)
print("H(g) ", Hg)

print("CASE OF LOPSIDED SKILLS")
mus = [(-0.33, 0.6), (-0.55, 0.6), (-0.77, 0.6), (-1, 0.6)]
stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
# calculate stuff
log_pg, pg, Hg = calculate_goal_metrics(clusters)
print("log p(g) for different goals ", log_pg.t())
print("overall p(g) ", pg)
print("H(g) ", Hg)

print("CASE OF 3 SKILLS ON TOP AND ONE BOTTOM")
mus = [(-0.66, 0.6), (0, 0.6), (0.66, 0.6), (0, -0.6)]
stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
# calculate stuff
log_pg, pg, Hg = calculate_goal_metrics(clusters)
print("log p(g) for different goals ", log_pg.t())
print("overall p(g) ", pg)
print("H(g) ", Hg)

print("CASE OF ALL SKILLS ON BOTTOM EQUALLY SPREAD OUT")
mus = [(1, -0.6), (0.33, -0.6), (-0.33, -0.6), (-1, -0.6)]
stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
# calculate stuff
log_pg, pg, Hg = calculate_goal_metrics(clusters)
print("log p(g) for different goals ", log_pg.t())
print("overall p(g) ", pg)
print("H(g) ", Hg)

# moving all skills to the bottom smoothly and plotting Hg
fig, ax = plt.subplots(figsize=(8,8))
for i in range(100):
    gap = i/100 * 1.2
    mus = [(1, 0.6-gap), (0.33, 0.6-gap), (-0.33, 0.6-gap), (-1, 0.6-gap)]
    stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
    clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
    # calculate stuff
    log_pg, pg, Hg = calculate_goal_metrics(clusters)
    ax.scatter(0.6-gap, Hg)
# plt.show()

print("CASE OF TWO SKILLS ON BOTTOM AND TWO ON TOP EQUALLY SPREAD OUT")
mus = [(0.33, -0.6), (0.33, 0.6), (-0.33, -0.6), (-0.33, 0.6)]
stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
# calculate stuff
log_pg, pg, Hg = calculate_goal_metrics(clusters)
print("log p(g) for different goals ", log_pg.t())
print("overall p(g) ", pg)
print("H(g) ", Hg)

# moving ONE skill to the bottom smoothly and plotting Hg
fig, ax = plt.subplots(figsize=(8,8))
for i in range(100):
    gap = i/100 * 1.2
    mus = [(-0.66, 0.6), (0, 0.6), (0.66, 0.6), (0, 0.6-gap)]
    stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
    clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
    # calculate stuff
    log_pg, pg, Hg = calculate_goal_metrics(clusters)
    ax.scatter(0.6-gap, Hg)
# plt.show()

# moving TWO skills to the bottom smoothly and plotting Hg
fig, ax = plt.subplots(figsize=(8,8))
for i in range(100):
    gap = i/100 * 1.2
    mus = [(0, 0.6), (0.66, 0.6-gap), (-0.66, 0.6-gap), (0, -0.6)]
    stds = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
    clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
    # calculate stuff
    log_pg, pg, Hg = calculate_goal_metrics(clusters)
    ax.scatter(0.6-gap, Hg)
ax.axhline(5.1227)
# plt.show()

# gradually increasing variance and plotting Hg
fig, ax = plt.subplots(figsize=(8,8))
for i in range(100):
    gap = i/100 * 0.95
    mus = [(-0.66, 0.6), (0, 0.6), (0.66, 0.6), (0, -0.6)]
    stds = [(0.05+gap, 0.05+gap), (0.05+gap, 0.05+gap), (0.05+gap, 0.05+gap), (0.05+gap, 0.05+gap)]
    clusters = Normal(torch.Tensor(np.stack(mus)), torch.Tensor(np.stack(stds)))
    # calculate stuff
    log_pg, pg, Hg = calculate_goal_metrics(clusters)
    ax.scatter(0.05+gap, Hg, c='r')
    ax.scatter(0.05+gap, pg, c='b')
plt.show()
