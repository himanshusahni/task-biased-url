import numpy as np
from collections import deque
import random
import pickle

from boxenv import *
from agent import *

import torch.optim as optim

GAMMA = 0.99
ENTROPY_COEFF = 0.001
NB_SKILLS = 6
SKILL_ENT = -math.log(1/NB_SKILLS)
EPS = 1e-6
BATCH_SIZE = 32
ALPHA = 10
COND = 'HIER-OUR'
SEED = 123

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

box = BoxWorld()
policy_function = GaussianPolicyFunction(2 + NB_SKILLS, 2)
value_function = ValueFunction(2 + NB_SKILLS)
policy = GaussianPolicy()
policy_optimizer = optim.Adam(policy_function.parameters(), lr=1e-4)
value_optimizer = optim.Adam(value_function.parameters(), lr=1e-3)

# meta policy
meta_policy_function = DiscretePolicyFunction(2, NB_SKILLS)
meta_policy = DiscretePolicy()
meta_policy_optimizer = optim.Adam(meta_policy_function.parameters(), lr=1e-4)


d = SkillDiscriminator(2, NB_SKILLS)
variational_optimizer = optim.Adam(d.parameters(), lr=1e-4)
CE = nn.CrossEntropyLoss()

metrics = {'step': [],
           'rewards': [],
           'task_rewards': [],
           'policy_loss': [],
           'meta_policy_loss': [],
           'value_loss': [],
           'entropy_loss': [],
           'meta_entropy_loss': [],
           'final_states': [],
           'variational_loss': [],
           }

variational_buffer = deque([], maxlen=10000)

step = 0
ep = 0

# May have to eventually move to SAC if learning is not stable. But for DIYAN
# looks good so far!

for i in range(5000):
    # reset environment
    s = box.reset()
    done = False
    rewards = []
    logprobs = []
    entropies = []
    values = []
    ep_task_reward = 0
    # sample a skill according to policy
    meta_logits = meta_policy_function(torch.Tensor(s))
    w, meta_logprob, meta_entropy = meta_policy.forward(meta_logits)
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    while not done:
        s = torch.Tensor(np.concatenate((s, w_onehot)))
        # get value prediction
        values.append(value_function(s))
        # get action and logprobs
        mu, sigma = policy_function(s)
        unscaled_action, logprob, entropy = policy.forward(mu, sigma)
        # step the environment
        action = box.scale_action(unscaled_action.detach().numpy())
        s, r, done = box.step(action)
        # compute rewards using DIYAN
        logits, probs = d(torch.Tensor(s))
        diversity_reward = torch.log(probs[w] + EPS).item() + SKILL_ENT
        variational_buffer.append((s, w))
        # calculate max task reward
        task_reward = r
        # task_reward = np.mean(r)
        ep_task_reward += task_reward
        rewards.append(diversity_reward + ALPHA * task_reward)
        # rewards.append(diversity_reward)
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
    # update variational function
    batch = random.sample(variational_buffer, BATCH_SIZE)
    s, w = zip(*batch)
    s = np.stack(s)
    logits, probs = d(torch.Tensor(s))
    variational_loss = CE(logits, torch.LongTensor(w))
    variational_optimizer.zero_grad()
    variational_loss.backward()
    variational_optimizer.step()
    # reinforce meta-policy
    meta_policy_loss = - ep_task_reward * meta_logprob
    meta_loss = meta_policy_loss - ENTROPY_COEFF * meta_entropy
    meta_policy_optimizer.zero_grad()
    meta_loss.backward()
    meta_policy_optimizer.step()
    # metrics
    metrics['step'].append(step)
    metrics['rewards'].append(np.sum(rewards))
    metrics['task_rewards'].append(ep_task_reward)
    metrics['policy_loss'].append(policy_loss.item())
    metrics['meta_policy_loss'].append(meta_policy_loss.item())
    metrics['value_loss'].append(value_loss.item())
    metrics['entropy_loss'].append(entropy_loss.item())
    metrics['meta_entropy_loss'].append(-meta_entropy.item())
    metrics['final_states'].append(s)
    metrics['variational_loss'].append(variational_loss.item())
    print("EPISODE ", ep, "REWARD = {:.2f} Var loss = {:.3f}".format(np.sum(rewards), variational_loss.item()))
    if step % 10000 == 0:
        # save networks!
        torch.save(policy_function.state_dict(), 'models/{}/policy_seed{}.pth'.format(COND, SEED))
        torch.save(value_function.state_dict(), 'models/{}/value_seed{}.pth'.format(COND, SEED))
        torch.save(d.state_dict(), 'models/{}/variational_seed{}.pth'.format(COND, SEED))
        pickle.dump(metrics, open('models/{}/metrics_seed{}.pkl'.format(COND, SEED), 'wb'))
