import matplotlib.pyplot as plt
import pickle
import numpy as np

SEED = 123
COND = 'OUR'
TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]
WINDOW = 50

metrics = pickle.load(open('../models/{}/{}/metrics_seed{}.pkl'.format(COND, len(TASKS), SEED), 'rb'))
fig, axarr = plt.subplots(3,2, figsize=(24,16))
x = [i/25 for i in metrics['step']]

# axarr[0,0].scatter(x, metrics['value_loss'])
# axarr[0,0].set_title("Value loss")

# axarr[0,1].scatter(x, metrics['policy_loss'])
# axarr[0,1].set_title("Policy loss")

# h = np.convolve(metrics['entropy'], np.ones(WINDOW)/WINDOW, mode='valid')
# axarr[0,0].scatter(range(h.shape[0]), h)
# axarr[0,0].set_title("Policy Entropy")

axarr[0,0].scatter(x, metrics['goal_entropy'])
axarr[0,0].set_title("H(g)")

# create running average
r = np.convolve(metrics['rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
axarr[0,1].scatter(range(r.shape[0]), r)
axarr[0,1].set_title("Episodic Reward")

Hgw, w = zip(*metrics['goal_w_entropy'])
# color = plt.cm.tab10(np.linspace(0, 1, 6))
color = ['r', 'b', 'k', 'g', 'y', 'm']
axarr[1,0].scatter(x, Hgw, color=[color[i] for i in w])
axarr[1,0].set_title("H(g|w)")
# axarr[1,0].set_ylim([-40,0])

axarr[1,1].scatter(x, metrics['goal_prob'])
axarr[1,1].set_title("p(g)")

# create running average
r = np.convolve(metrics['diversity_rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
axarr[2,0].scatter(range(r.shape[0]), r)
axarr[2,0].set_title("Diversity Reward")

multiplier, w = zip(*metrics['extrinsic_rewards'])
color = ['r', 'b', 'k', 'g', 'y', 'm']
axarr[2,1].scatter(x, multiplier, color=[color[i] for i in w])
axarr[2,1].set_title("Reward Multiplier")
# v = np.convolve(metrics['variational_loss'], np.ones(WINDOW)/WINDOW, mode='valid')
# axarr[1,0].scatter(range(v.shape[0]), v)
# axarr[1,0].set_title("Variational loss")

# axarr[1,1].scatter(range(len(metrics['value'])), metrics['value'])
# axarr[1,1].set_title("Average Q Value")

# axarr[2,0].scatter(range(len(metrics['neg_logprob'])), metrics['neg_logprob'])
# axarr[2,0].set_title("neg logprob")

# axarr[2,1].scatter(range(len(metrics['alpha'])), metrics['alpha'])
# axarr[2,1].set_title("Alpha")

plt.savefig('../imgs/{}/{}/metrics_{}.png'.format(COND, len(TASKS), SEED))
