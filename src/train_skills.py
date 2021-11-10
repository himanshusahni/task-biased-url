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
BATCH_SIZE = 64
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
SEED = 456

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

box = BoxWorld()
reinforce = REINFORCE(STATE_DIM + NB_SKILLS, 2)

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

step = 0
ep = 0

max_running_reward = -10000
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
    pg = logsumexp(log_pg).exp().item()
    Hg =  - (log_pg.exp() * (log_pg - SKILL_ENT)).sum().item()
    # calculate H(g|w)
    goal_w_logprobs = goal_logprobs[:, w]
    Hgw = - (goal_w_logprobs.exp() * goal_w_logprobs).sum().item()
    goal_probs = goal_w_logprobs.exp().numpy()
    # rollout
    states = []
    rewards = []
    logprobs = []
    entropies = []
    while not done:
        prev_s = s
        s = torch.Tensor(np.concatenate((s, w_onehot)))
        states.append(s)
        # get action and logprobs
        unscaled_action, logprob, entropy = reinforce.policy.sample(s.unsqueeze(0))
        logprobs.append(logprob.squeeze())
        entropies.append(entropy.squeeze())
        # step the environment
        unscaled_action = unscaled_action.squeeze().detach().numpy()
        action = box.scale_action(unscaled_action)
        s, _, done = box.step(action)
        _, dr = d(torch.Tensor(s).unsqueeze(0))
        diversity_reward = dr[0,w].item() + SKILL_ENT
        # compute extrinsic reward
        # r = compute_rewards(prev_s, s)
        # extrinsic_reward = np.dot(goal_probs, r)
        # extrinsic_reward = -Hgw
        # extrinsic_reward = r[0]
        # total_reward = diversity_reward + 10*extrinsic_reward
        total_reward = -Hgw * diversity_reward
        rewards.append(total_reward)
        # metrics
        ep_reward += total_reward
        ep_diversity_reward += diversity_reward
        # ep_extrinsic_reward += extrinsic_reward
        variational_buffer.append((s, w))
        step += 1
    # print(Hg, goal_w_logprobs.sum().item())
    rewards[-1] += 10*Hg
    # metrics
    metrics['step'].append(step)
    metrics['rewards'].append(sum(rewards))
    metrics['diversity_rewards'].append(ep_diversity_reward)
    # metrics['extrinsic_rewards'].append(ep_extrinsic_reward)
    metrics['goal_entropy'].append(10*Hg)
    metrics['goal_prob'].append(pg)
    metrics['goal_w_entropy'].append((Hgw, w))
    print("EPISODE ", ep, "REWARD = {:.2f}".format(sum(rewards)))
    ################################# RL ###############################
    rollout = (states, rewards, logprobs, entropies)
    policy_loss, value_loss, value, entropy = reinforce.update(rollout)
    metrics['policy_loss'].append(policy_loss)
    metrics['value_loss'].append(value_loss)
    metrics['entropy'].append(entropy)
    metrics['value'].append(value)
    ########################### Discriminator ##########################
    if ep > 100:
        idxs = np.random.randint(len(variational_buffer), size=BATCH_SIZE)
        batch = [variational_buffer[idx] for idx in idxs]
        batch = zip(*batch)
        discriminator_loss = variational_update(batch)
        metrics['variational_loss'].append(discriminator_loss)
    ep += 1
    if ep > 100:
        ################################# CLUSTERING ###########################
        if ep % 10 == 0:
            s, w = zip(*list(variational_buffer))
            samples = np.stack(s)
            samples = torch.Tensor(samples)
            w = torch.Tensor(w)
            for sk in range(NB_SKILLS):
                idxs = torch.where(w == sk)
                trajs = samples[idxs]  # (-1, TRAJ_LEN, 2)
                trajs = trajs.reshape(-1, STATE_DIM)  # (-1*TRAJ_LEN, 2)
                mus[sk] = trajs.mean(dim=0)
                A = trajs - mus[sk]
                A = A*A
                A = A.sum(dim=0) / (trajs.shape[0] - 1)
                variances[sk] = torch.clamp(A , 1e-6)
            vars_matrix = torch.stack([torch.diag(v) for v in variances])
            clusters = MultivariateNormal(mus, vars_matrix)
    if max_running_reward < sum(metrics['rewards'][-10:])/10:
        print("NEW MAX RUNNING REWARD!: ", sum(metrics['rewards'][-10:])/10)
        max_running_reward = sum(metrics['rewards'][-10:])/10
        torch.save(reinforce.to_save(), '../models/{}/{}/best_reinforce_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(d.state_dict(), '../models/{}/{}/best_variational_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(clusters, '../models/{}/{}/best_clusters_seed{}.pth'.format(
            COND, len(TASKS), SEED))
    # save networks!
    if step % 1000 == 0:
        torch.save(reinforce.to_save(), '../models/{}/{}/reinforce_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(d.state_dict(), '../models/{}/{}/variational_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(clusters, '../models/{}/{}/clusters_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        pickle.dump(metrics, open('../models/{}/{}/metrics_seed{}.pkl'.format(
            COND, len(TASKS), SEED), 'wb'))
