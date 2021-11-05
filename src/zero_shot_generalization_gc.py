import numpy as np
import random

from boxenv import *
from agent import *

COND = 'GC'
STATE_DIM = 2
DIM = STATE_DIM

policy_function = GaussianPolicyFunction(2 * STATE_DIM, 2)
policy = GaussianPolicy()

# initial training task list
TASKS = [(0.5, 0.8), (-0.5, 0.8)]
# TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]

# create a stationary test task list
rng = np.random.default_rng(1)
TEST_TASKS = rng.random((10,2))
TEST_TASKS = TEST_TASKS * 2 - 1
TEST_TASKS[:,1] = 0.8

print('Testing zero-shot generalization on tasks ')
print(TEST_TASKS)

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


rewards = []
for SEED in [123, 456, 789]:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    box = BoxWorld()
    NB_TASKS = TEST_TASKS.shape[0]

    policy_function.load_state_dict(torch.load(
        'models/{}/{}/policy_seed{}.pth'.format(COND, len(TASKS), SEED)))
    rewards.append([])
    for gid in range(NB_TASKS):
        rewards[-1].append([])
        for i in range(10):
            s = box.reset()
            done = False
            states = []
            rewards[-1][gid].append(0)
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
                r = compute_rewards(states[-1], s, TEST_TASKS[gid])
                rewards[-1][gid][-1] += r
rewards = np.stack(rewards)
print(rewards.mean(-1).mean(-1))
print(rewards.mean())
print(np.std(rewards.mean(-1).mean(-1)))
