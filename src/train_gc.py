import numpy as np
from collections import deque
import random
import pickle

from boxenv import *
from agent import *

import torch.optim as optim

GAMMA = 0.99
ENTROPY_COEFF = 0.001
EPS = 1e-6
STATE_DIM = 2

TASKS = [(0.5, 0.8), (-0.5, 0.8)]
# TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]
# TASKS = [(1., 0.8), (1 - 2./7, 0.8), (1 - 4./7, 0.8), (1 - 6./7, 0.8),
#          (-1 + 6./7, 0.8), (-1 + 4./7, 0.8), (-1 + 2./7, 0.8), (-1., 0.8)]

COND = 'GC'
SEED = 123

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

box = BoxWorld()
policy_function = GaussianPolicyFunction(2 * STATE_DIM, 2)
value_function = ValueFunction(2 * STATE_DIM)
policy = GaussianPolicy()
policy_optimizer = optim.Adam(policy_function.parameters(), lr=1e-4)
value_optimizer = optim.Adam(value_function.parameters(), lr=1e-3)

def compute_rewards(s1, s2, g):
    """
    input: s1 - state before action
           s2 - state after action
    rewards based on proximity to each goal
    """
    dist1 = np.linalg.norm(s1 - g)
    dist2 = np.linalg.norm(s2 - g)
    r = dist1 - dist2
    return r

metrics = {'step': [],
           'rewards': [],
           'policy_loss': [],
           'value_loss': [],
           'entropy_loss': [],
           }

step = 0
ep = 0

# May have to eventually move to SAC if learning is not stable. But for DIYAN
# looks good so far!
for i in range(5000):
    # sample a random TASK
    gid = np.random.randint(len(TASKS))
    g = TASKS[gid]
    # reset environment
    s = box.reset()
    done = False
    rewards = []
    logprobs = []
    entropies = []
    values = []
    states = []
    while not done:
        states.append(s)
        s = torch.Tensor(np.concatenate((s, np.array(g))))
        # get value prediction
        values.append(value_function(s))
        # get action and logprobs
        unscaled_action, logprob, entropy = policy.forward(*policy_function(s))
        # step the environment
        action = box.scale_action(unscaled_action.detach().numpy())
        s, _, done = box.step(action)
        # compute extrinsic reward
        r = compute_rewards(states[-1], s, g)
        # append
        rewards.append(r)
        logprobs.append(logprob)
        entropies.append(entropy)
        step += 1
    ep += 1
    # calculate discounted return for each step at end of episode.
    rewards = np.array([rewards[i] for i in range(len(rewards))])
    rewards = np.flip(rewards)
    R = [rewards[0]]
    for i in range(1, len(rewards)):
        R.append(rewards[i] + GAMMA * R[i-1])
    R = np.flip(np.array(R))
    rewards = np.flip(rewards)
    # calculate single step value target
    values = torch.stack(values).squeeze()
    targets = rewards[:-1] + GAMMA * values[1:].detach().numpy()
    targets = np.append(targets, rewards[-1])
    # REINFORCE
    value_loss = (values - torch.Tensor(targets))**2
    value_loss = value_loss.mean()
    R = R - values.detach().numpy()
    policy_loss = - sum([R[i]*logprobs[i] for i in range(len(R))])/len(R)
    entropy_loss = - sum(entropies)/len(entropies)
    loss = value_loss + policy_loss + ENTROPY_COEFF * entropy_loss
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    value_optimizer.step()
    # metrics
    metrics['step'].append(step)
    metrics['rewards'].append(np.sum(rewards))
    metrics['policy_loss'].append(policy_loss.item())
    metrics['value_loss'].append(value_loss.item())
    metrics['entropy_loss'].append(entropy_loss.item())
    print("EPISODE ", ep, "REWARD = {:.2f}".format(np.sum(rewards)))
    if step % 5000 == 0:
        # save networks!
        torch.save(policy_function.state_dict(), 'models/{}/{}/policy_seed{}.pth'.format(COND, len(TASKS), SEED))
        torch.save(value_function.state_dict(), 'models/{}/{}/value_seed{}.pth'.format(COND, len(TASKS), SEED))
        pickle.dump(metrics, open('models/{}/{}/metrics_seed{}.pkl'.format(COND, len(TASKS), SEED), 'wb'))
