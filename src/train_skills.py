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
TASKS = [(0.75, 0.8), (0.25, 0.8), (-0.25, 0.8), (-0.75, 0.8)]
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
ppo = PPO(STATE_DIM + NB_SKILLS, 2, 2000)

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

metrics = {'step': [],
           'rewards': [],
           'diversity_rewards': [],
           'extrinsic_rewards': [],
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

# buffer for each skill separately
variational_buffer = [deque([], maxlen=1500) for _ in range(NB_SKILLS)]
def sample_buffer():
    batch = []
    for i in range(BATCH_SIZE):
        # first randomly sample a skill
        w = np.random.randint(NB_SKILLS)
        # now sample a state from that buffer
        idx = np.random.randint(len(variational_buffer[w]))
        batch.append((variational_buffer[w][idx], w))
    return batch

def variational_update(batch):
    s, w = batch
    logits, _ = d(torch.Tensor(s))
    discriminator_loss = CE(logits, torch.LongTensor(w))
    discriminator_optimizer.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()
    return discriminator_loss.item()

step = 0
ep = 0

max_running_reward = -10000
# rollout
states = []
actions = []
rewards = []
logprobs = []
policy_mus = []
policy_logstds = []
for i in range(20000):
    # sample a random skill
    w = np.random.randint(0, NB_SKILLS)
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    # reset environment
    s = box.reset()
    done = False
    ep_diversity_reward = 0
    ep_states = []
    ep_rewards = []
    while not done:
        prev_s = s
        s = torch.Tensor(np.concatenate((s, w_onehot)))
        ep_states.append(s)
        # get action and logprobs
        output = ppo.policy.forward(s.unsqueeze(0))
        logprob = output['logprob']
        logprobs.append(logprob.squeeze())
        policy_mus.append(output['mu'].squeeze())
        policy_logstds.append(output['logstd'].squeeze())
        # step the environment
        unscaled_action = output['action']
        unscaled_action = unscaled_action.squeeze().detach()
        actions.append(unscaled_action)
        action = box.scale_action(unscaled_action.numpy())
        s, _, done = box.step(action)
        _, dr = d(torch.Tensor(s).unsqueeze(0))
        diversity_reward = dr[0,w].item() + SKILL_ENT
        extrinsic_reward = compute_rewards(prev_s, s)
        total_reward = diversity_reward / 10
        ep_rewards.append(total_reward)
        # metrics
        ep_diversity_reward += diversity_reward
        # ep_extrinsic_reward += extrinsic_reward
        variational_buffer[w].append(s)
        step += 1
    s = torch.Tensor(np.concatenate((s, w_onehot)))
    ep_states.append(s)
    ep += 1
    ################################# CLUSTERING ###############################
    if ep > 100:
        s = list(variational_buffer[w])
        trajs = torch.Tensor(np.stack(s))
        mus[w] = trajs.mean(dim=0)
        A = trajs - mus[w]
        A = torch.matmul(A.t(), A) / trajs.shape[0]
        vars_matrix[w] = torch.clamp(A , 1e-6)
        clusters = MultivariateNormal(mus, vars_matrix)
    # calculate goal information
    goal_logprobs = clusters.log_prob(goals)
    # calculate H(g)
    log_pgw = logsumexp(goal_logprobs, dim=1)
    pg = logsumexp(log_pgw).exp().item()
    Hg =  - (log_pgw.exp() * (log_pgw - SKILL_ENT)).sum().item()
    # calculate H(g|w)
    goal_w_logprobs = goal_logprobs[:, w]
    Hgw = - (goal_w_logprobs.exp() * goal_w_logprobs).sum().item()
    # modify rewards
    multiplier = (-Hgw + goal_w_logprobs.max().item())
    # multiplier = math.tanh(multiplier/2)
    # multiplier = 1 / (1 + math.exp(-multiplier))
    # multiplier += 1
    # modify rewards in retrospect
    ep_rewards = [r * multiplier for r in ep_rewards]
    ep_rewards[-1] += Hg + pg
    states.append(torch.stack(ep_states))
    rewards.append(torch.Tensor(ep_rewards))
    # metrics
    metrics['step'].append(step)
    metrics['rewards'].append(sum(ep_rewards))
    metrics['diversity_rewards'].append(ep_diversity_reward)
    metrics['extrinsic_rewards'].append((multiplier, w))
    metrics['goal_entropy'].append(Hg)
    metrics['goal_prob'].append(pg)
    metrics['goal_w_entropy'].append((-Hgw + goal_w_logprobs.max().item(), w))
    print("EPISODE ", ep, "REWARD = {:.2f}".format(sum(ep_rewards)))
    ################################# RL ###############################
    if ep % 10 == 0:
        rollout = (states, actions, rewards, logprobs, policy_mus, policy_logstds)
        policy_loss, value_loss, value, entropy = ppo.update(rollout)
        metrics['policy_loss'].append(policy_loss)
        metrics['value_loss'].append(value_loss)
        metrics['entropy'].append(entropy)
        metrics['value'].append(value)
        states = []
        actions = []
        rewards = []
        logprobs = []
        policy_mus = []
        policy_logstds = []
    ########################### Discriminator ##########################
    if ep > 100:
        batch = sample_buffer()
        batch = zip(*batch)
        discriminator_loss = variational_update(batch)
        metrics['variational_loss'].append(discriminator_loss)
    if max_running_reward < sum(metrics['rewards'][-10:])/10:
        print("NEW MAX RUNNING REWARD!: ", sum(metrics['rewards'][-10:])/10)
        max_running_reward = sum(metrics['rewards'][-10:])/10
        torch.save(ppo.to_save(), '../models/{}/{}/ppo/best_ppo_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(d.state_dict(), '../models/{}/{}/ppo/best_variational_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(clusters, '../models/{}/{}/ppo/best_clusters_seed{}.pth'.format(
            COND, len(TASKS), SEED))
    # save networks!
    if step % 1000 == 0:
        torch.save(ppo.to_save(), '../models/{}/{}/ppo/ppo_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(d.state_dict(), '../models/{}/{}/ppo/variational_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        torch.save(clusters, '../models/{}/{}/ppo/clusters_seed{}.pth'.format(
            COND, len(TASKS), SEED))
        pickle.dump(metrics, open('../models/{}/{}/ppo/metrics_seed{}.pkl'.format(
            COND, len(TASKS), SEED), 'wb'))
