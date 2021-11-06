import numpy as np
from collections import deque
import random
import pickle

from boxenv import *
from agent import *

import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

GAMMA = 0.99
ENTROPY_COEFF = 0.001
EPS = 1e-6
BATCH_SIZE = 128
ALPHA = 10
STATE_DIM = 2
TRAJ_LEN = 25

# TASKS = [(0.5, 0.8), (-0.5, 0.8)]
TASKS = [(1., 0.8), (0.33, 0.8), (-0.33, 0.8), (-1, 0.8)]
# TASKS = [(1., 0.8), (1 - 2./7, 0.8), (1 - 4./7, 0.8), (1 - 6./7, 0.8),
         # (-1 + 6./7, 0.8), (-1 + 4./7, 0.8), (-1 + 2./7, 0.8), (-1., 0.8)]
DIM = STATE_DIM

NB_SKILLS = 6
SKILL_ENT = -math.log(1/NB_SKILLS)

COND = 'OUR'
SEED = 123

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

box = BoxWorld()
sac = SAC(STATE_DIM + NB_SKILLS, 2)
# policy_function = GaussianPolicyFunction(STATE_DIM + NB_SKILLS, 2)
# value_function = ValueFunction(STATE_DIM + NB_SKILLS)
# policy = GaussianPolicy()
# policy_optimizer = optim.Adam(policy_function.parameters(), lr=1e-4)
# value_optimizer = optim.Adam(value_function.parameters(), lr=1e-3)

d = SkillDiscriminator(DIM, NB_SKILLS)
discriminator_optimizer = optim.Adam(d.parameters(), lr=1e-4)
CE = nn.CrossEntropyLoss()

mus = 0.1*torch.rand((NB_SKILLS, DIM))
variances = torch.ones((NB_SKILLS, DIM))
vars_matrix = torch.stack([torch.diag(v) for v in variances])
clusters = MultivariateNormal(mus, vars_matrix)
goals = torch.Tensor(np.stack(TASKS)).repeat([1, NB_SKILLS])
goals = goals.reshape(len(TASKS), NB_SKILLS, STATE_DIM)

def compute_rewards(s1, s2):
    """
    input: s1 - state before action
           s2 - state after action
    rewards based on proximity to each goal
    """
    r = []
    for g in TASKS:
        dist1 = np.linalg.norm(s1 - g)
        dist2 = np.linalg.norm(s2 - g)
        reward = dist1 - dist2
        r.append(reward)
    return r

def variational_update(batch):
    s, w = batch
    logits, _ = d(torch.Tensor(s))
    discriminator_loss = CE(logits, torch.LongTensor(w))
    discriminator_optimizer.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()
    return discriminator_loss.item()

metrics = {'step': [],
           'rewards': [],
           'diversity_rewards': [],
           # 'extrinsic_rewards': [],
           'goal_entropy': [],
           'goal_prob': [],
           'goal_w_entropy': [],
           'policy_loss': [],
           'value_loss': [],
           'alpha': [],
           'entropy': [],
           'neg_logprob': [],
           'value': [],
           'variational_loss': [],
           }

variational_buffer = deque([], maxlen=10000)
replay_buffer = deque([], maxlen=100000)

step = 0
ep = 0

# May have to eventually move to SAC if learning is not stable. But for DIYAN
# looks good so far!
for i in range(20000):
    # sample a random skill
    w = np.random.randint(0, NB_SKILLS)
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    # reset environment
    s = box.reset()
    done = False
    ep_reward = 0
    ep_diversity_reward = 0
    # ep_extrinsic_reward = 0
    goal_logprobs = clusters.log_prob(goals)
    # calculate H(g)
    log_pg = logsumexp(goal_logprobs, dim=1)
    pg = log_pg.exp().sum().item()
    Hg =  - (log_pg.exp() * log_pg).sum().item()
    # normalize probabilities for goals (sum to one over goals)
    goal_logprobs -= logsumexp(goal_logprobs)
    # calculate H(g|w)
    goal_w_logprobs = goal_logprobs[:, w]
    Hgw = - (goal_w_logprobs.exp() * goal_w_logprobs).sum().item()
    goal_probs = goal_w_logprobs.exp().numpy()
    while not done:
        prev_s = s
        s = torch.Tensor(np.concatenate((s, w_onehot)))
        # get action and logprobs
        unscaled_action, logprob, _ = sac.policy.sample(s.unsqueeze(0))
        # step the environment
        unscaled_action = unscaled_action.squeeze().detach().numpy()
        action = box.scale_action(unscaled_action)
        s, _, done = box.step(action)
        _, dr = d(torch.Tensor(s).unsqueeze(0))
        diversity_reward = dr[0,w].item() + SKILL_ENT
        # compute extrinsic reward
        r = compute_rewards(prev_s, s)
        # extrinsic_reward = np.dot(goal_probs, r)
        # extrinsic_reward = -Hgw
        # extrinsic_reward = r[0]
        # total_reward = diversity_reward + 10*extrinsic_reward
        total_reward =  diversity_reward
        if done:
            total_reward += -Hgw + Hg
        # total_reward = extrinsic_reward
        # metrics
        ep_reward += total_reward
        ep_diversity_reward += diversity_reward
        # ep_extrinsic_reward += extrinsic_reward
        replay_buffer.append((np.concatenate((prev_s, w_onehot)),
                              unscaled_action,
                              total_reward,
                              np.concatenate((s, w_onehot)),
                              done,))
        variational_buffer.append((s, w))
        step += 1
        if step > 10000:
            ################################# RL ###############################
            idxs = np.random.randint(len(replay_buffer), size=BATCH_SIZE)
            batch = [replay_buffer[idx] for idx in idxs]
            batch = zip(*batch)
            policy_loss, value_loss, alpha, entropy, neg_logprob, value = sac.update(batch)
            metrics['policy_loss'].append(policy_loss)
            metrics['value_loss'].append(value_loss)
            metrics['alpha'].append(alpha)
            metrics['entropy'].append(entropy)
            metrics['neg_logprob'].append(neg_logprob)
            metrics['value'].append(value)
        # if step > 1000:
            ########################### Discriminator ##########################
            idxs = np.random.randint(len(variational_buffer), size=BATCH_SIZE)
            batch = [variational_buffer[idx] for idx in idxs]
            batch = zip(*batch)
            discriminator_loss = variational_update(batch)
            metrics['variational_loss'].append(discriminator_loss)
    ep += 1
    # if ep > 100:
    #     ################################# CLUSTERING ###########################
    #     if ep % 10 == 0:
    #         w, s = zip(*list(variational_buffer))
    #         samples = np.stack(s)
    #         samples = torch.Tensor(samples)
    #         w = torch.Tensor(w)
    #         for sk in range(NB_SKILLS):
    #             idxs = torch.where(w == sk)
    #             trajs = samples[idxs]  # (-1, TRAJ_LEN, 2)
    #             trajs = trajs.reshape(-1, STATE_DIM)  # (-1*TRAJ_LEN, 2)
    #             mus[sk] = trajs.mean(dim=0)
    #             A = trajs - mus[sk]
    #             A = A*A
    #             A = A.sum(dim=0) / (trajs.shape[0] - 1)
    #             variances[sk] = torch.clamp(A , 1e-6)
    #         vars_matrix = torch.stack([torch.diag(v) for v in variances])
    #         clusters = MultivariateNormal(mus, vars_matrix)
    # metrics
    metrics['step'].append(step)
    metrics['rewards'].append(ep_reward)
    metrics['diversity_rewards'].append(ep_diversity_reward)
    # metrics['extrinsic_rewards'].append(ep_extrinsic_reward)
    metrics['goal_entropy'].append(Hg)
    metrics['goal_prob'].append(pg)
    metrics['goal_w_entropy'].append(Hgw)
    print("EPISODE ", ep, "REWARD = {:.2f}".format(ep_reward))
    if step % 1000 == 0:
        # save networks!
        torch.save(sac.to_save(), '../models/{}/{}/sac_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(d.state_dict(), '../models/{}/{}/variational_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(clusters, '../models/{}/{}/clusters_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        pickle.dump(metrics, open('../models/{}/{}/metrics_seed{}.pkl'.format(
            COND, len(TASKS), SEED), 'wb'))
