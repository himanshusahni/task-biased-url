import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import torch.optim as optim


class GaussianPolicyFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim, action_dim):
        super(GaussianPolicyFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.mu_out = nn.Linear(200, action_dim)
        self.sigma_out = nn.Linear(200, action_dim)

    def forward(self, x):
        """return: action between [-1,1]"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        # return torch.tanh(self.mu_out(x)), F.softplus(self.sigma_out(x))
        return torch.tanh(self.mu_out(x)), torch.tanh(self.sigma_out(x))


class DiscretePolicyFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim, action_dim):
        super(DiscretePolicyFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, action_dim)

    def forward(self, x):
        """return: action between [-1,1]"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return F.softmax(self.out(x), 0)


class QValueFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim, action_dim):
        super(QValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, s, a):
        """return: scalar value"""
        x = torch.cat((s,a), dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.out(x)


class ValueFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, x):
        """return: scalar value"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.out(x)

class SkillDiscriminator(nn.Module):
    """fully connected 200x200 layers for inferring q(z|s)"""

    def __init__(self, state_dim, nb_skills):
        super(SkillDiscriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, nb_skills)

    def forward(self, x):
        """return: scalar value"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        logits = self.out(x)
        return logits, nn.LogSoftmax(dim=1)(logits)


class GaussianPolicy:
    """Gaussian policy with a learned sigma"""

    def __init__(self, policy_func):
        """
        policy_function is a neural network that outputs action mean between
        and log standard deviation
        """
        self.policy_func = policy_func

    def forward(self, s, action=None):
        """sample actions (unless actions are given),
        and calculate mu, std, logprob, and entropy"""
        mu, logstd = self.policy_func(s)
        # logstd = 0.5*(logstd + 1)  #0->1
        # logstd = 4*logstd - 3  #-3->1
        dist = Normal(mu, logstd.exp())
        if action is None:
            action = dist.rsample()
            action = action.detach()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean(dim=-1)
        return {'action': action,
                'logprob': logprob,
                'entropy': entropy,
                'mu': mu,
                'std': logstd.exp(),}


class DiscretePolicy:
    """Discrete multinomial policy"""

    def forward(self, probs):
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy


def logsumexp(x, dim=0):
    """log sum exp with stability trick"""
    max, _ = torch.max(x, dim=dim, keepdim=True)
    out = max + (x - max).exp().sum(dim=dim).log().unsqueeze(dim=dim)
    return out


class REINFORCE:

    def __init__(self, state_dim, action_dim, device):
        # policy
        self.policy_func = GaussianPolicyFunction(state_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_func.parameters(), lr=1e-4)
        self.policy = GaussianPolicy(self.policy_func)
        # value
        self.value_function = ValueFunction(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=1e-3)
        # misc
        self.gamma = 0.99
        self.entropy_coeff = 0.01

    def to_save(self):
        return {
            'policy_func': self.policy_func.state_dict(),
            'policy_optim': self.policy_optimizer.state_dict(),
            'value_func': self.value_function.state_dict(),
            'value_optim': self.value_optimizer.state_dict(),
        }

    def update(self, rollout):
        """
        rollout consists of a tuple of lists of (states(+w), rewards, logprobs, entropies)
        """
        states = rollout['states']
        actions = rollout['actions']
        rewards = rollout['rewards']
        dones = rollout['dones']
        # calculate discounted return for each step at end of episode.
        R = torch.zeros_like(rewards)
        R[:,-1] = rewards[:,-1]
        for i in reversed(range(rewards.shape[1] - 1)):
            R[:,i] = rewards[:,i] + self.gamma * R[:,i+1] * (1 - dones[:,i])
        # calculate single step value target
        values = self.value_function(states).squeeze()
        targets = rewards + self.gamma * values[:,1:].detach()
        values = values[:,:-1]
        # calculate log probabilities of actions
        output = self.policy.forward(states[:,:-1], actions)
        logprobs = output['logprob'].flatten()
        entropy = output['entropy']
        # REINFORCE
        value_loss = (values - targets)**2
        value_loss = value_loss.mean()
        R = R - values.detach()
        R = R.flatten()
        policy_loss = - (R * logprobs).mean()
        entropy_loss = - entropy.mean()
        loss = value_loss + policy_loss + self.entropy_coeff * entropy_loss
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        return policy_loss.item(), value_loss.item(), values.mean().item(), -entropy_loss.item()


class SAC:

    def __init__(self, state_dim, action_dim):
        # policy
        self.policy_func = GaussianPolicyFunction(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy_func.parameters(), lr=1e-4)
        self.policy = GaussianPolicy(self.policy_func)
        # value
        self.q1 = QValueFunction(state_dim, action_dim)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2 = QValueFunction(state_dim, action_dim)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=1e-3)
        self.target_q1 = QValueFunction(state_dim, action_dim)
        self.target_q2 = QValueFunction(state_dim, action_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        # self.V = ValueFunction(state_dim)
        # self.v_optimizer = optim.Adam(self.V.parameters(), lr=1e-3)

        init_alpha = 0.001
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -action_dim / 2

        self.gamma = 0.99
        self.polyak_constant = 0.995

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def to_save(self):
        return {
            'policy_func': self.policy_func.state_dict(),
            'policy_optim': self.policy_optimizer.state_dict(),
            'q1': self.q1.state_dict(),
            'q1_optim': self.q1_optimizer.state_dict(),
            'q2': self.q2.state_dict(),
            'q2_optim': self.q2_optimizer.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
        }

    def _polyak_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.polyak_constant * target_param.data + \
                (1 - self.polyak_constant) * param.data
            )

    def update(self, batch):
        """
        batch is given as a tuple (states, actions, rewards, next_states, dones)
        """
        s, a, r, s_, d = batch
        s = torch.Tensor(np.stack(s))
        a = torch.Tensor(np.stack(a))
        r = torch.Tensor(r)
        s_ = torch.Tensor(np.stack(s_))
        d = torch.Tensor(d)
        # sample actions from current policy
        _a_, logprob_a_, _ = self.policy.sample(s_)
        # create target
        q_s_ = (self.target_q1(s_, _a_), self.target_q2(s_, _a_))
        q_s_ = torch.cat(q_s_, dim=1)
        q_s_ = q_s_.min(dim=1)[0]
        # V_s_ = self.V(s_)
        target = r + self.gamma * (q_s_ - self.alpha * logprob_a_)
        # target = r + self.gamma * V_s_
        target = target.detach()
        # create qvalue loss
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        qvalue_loss = (q1 - target)**2 + (q2 - target)**2
        qvalue_loss = qvalue_loss.mean()
        # update value function
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        qvalue_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        # update target networks with polyak averaging
        self._polyak_update(self.q1, self.target_q1)
        self._polyak_update(self.q2, self.target_q2)
        # sample actions from current policy
        _a, logprob_a, entropy = self.policy.sample(s)
        # construct policy loss
        q_s = (self.q1(s, _a), self.q2(s, _a))
        q_s = torch.cat(q_s, dim=1)
        q_s = q_s.min(dim=1)[0]
        policy_loss = - (q_s - self.alpha * logprob_a)
        policy_loss = policy_loss.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        # # construct value loss
        # V_s = self.V(s)
        # target = q_s - self.alpha * logprob_a
        # target = target.detach()
        # value_loss = (V_s - target)**2
        # value_loss = value_loss.mean()
        # self.v_optimizer.zero_grad()
        # value_loss.backward()
        # self.v_optimizer.step()
        # update alpha
        alpha_loss = (self.alpha *
                      (-logprob_a / 2 - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return policy_loss.item(), qvalue_loss.item(), self.alpha.item(), entropy.mean().item(), (-logprob_a / 2).mean().item(), q_s.mean().item()


class PPO:

    def __init__(self, state_dim, action_dim, max_train_steps, device):
        # policy
        self.policy_func = GaussianPolicyFunction(
            state_dim, action_dim).to(device)
        self.policy = GaussianPolicy(self.policy_func)
        self.policy_func_new = GaussianPolicyFunction(
            state_dim, action_dim).to(device)
        self.policy_new = GaussianPolicy(self.policy_func_new)
        self.policy_optimizer = optim.Adam(
            self.policy_func_new.parameters(), lr=1e-4)
        self.policy_func.load_state_dict(self.policy_func_new.state_dict())
        # value
        self.value_function = ValueFunction(state_dim).to(device)
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(), lr=1e-3)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_train_steps = max_train_steps
        self.train_steps = 0

        self.gamma = 0.99
        self.entropy_coeff = 0.001
        self.ppo_epochs = 80
        self.l = 0.95
        self.start_clip_param = 0.2
        self.end_clip_param = 0.05
        self.target_kl = 0.05
        self.batch_size = 256

    @property
    def clip_param(self):
        # return self.start_clip_param +\
        #     (self.end_clip_param - self.start_clip_param) *\
        #     self.train_steps / self.max_train_steps
        return self.start_clip_param

    def to_save(self):
        return {
            'policy_func': self.policy_func.state_dict(),
            'policy_optim': self.policy_optimizer.state_dict(),
            'value_func': self.value_function.state_dict(),
            'value_optim': self.value_optimizer.state_dict(),
        }

    def update(self, rollout):
        """
        states include initial and final state so one longer than the rest
        assumes that the final state of a rollout is done
        states: nb_rollout,  rollout_steps + 1, state_dim
        actions: nb_rollout, rollout_steps, action_dim
        rewards: nb_rollout, rollout_steps
        dones: nb_rollout, ep_length,
        """
        s = rollout['states']
        a = rollout['actions']
        r = rollout['rewards']
        d = rollout['dones']
        # get value estimate
        values = self.value_function(s).squeeze().detach()
        # as an approximation, replace values of after terminal states with
        # the last states's value (instead of 0)
        values[:,1:] = values[:,1:] * (1 - d) + values[:,:-1] * d
        adv = torch.zeros_like(values)
        for i in reversed(range(r.shape[1])):
            delta = r[:,i] + self.gamma * values[:,i+1] - values[:,i]
            adv[:,i] = delta + self.gamma * self.l * adv[:,i+1] * (1 - d[:,i])
        adv = adv[:,:-1].flatten()
        values = values[:,:-1].flatten()
        returns = values + adv
        s = s[:,:-1].reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        output = self.policy.forward(s,a)
        logprobs = output['logprob'].detach()
        mus = output['mu'].squeeze().detach()
        logstds = output['std'].squeeze().detach().log()
        # # standardize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        epoch = 0
        kldiv = 0
        avg_policy_loss = 0
        while epoch < self.ppo_epochs and kldiv < self.target_kl:
            # sample a minibatch to learn from
            idxs = np.random.randint(s.shape[0], size=(self.batch_size,))
            s_b = s[idxs]
            a_b = a[idxs]
            logprobs_b = logprobs[idxs]
            adv_b = adv[idxs]
            mus_b = mus[idxs]
            logstds_b = logstds[idxs]
            returns_b = returns[idxs]
            # calculate policy loss
            output = self.policy_new.forward(s_b, a_b)
            new_logprobs = output['logprob']
            new_entropy = output['entropy']
            new_mus = output['mu'].squeeze().detach()
            new_logstds = output['std'].squeeze().detach().log()
            ratio = (new_logprobs - logprobs_b).exp()
            loss1 = ratio * adv_b
            loss2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv_b
            policy_loss = -torch.min(loss1, loss2).mean()
            entropy_loss = -new_entropy.mean()
            self.policy_optimizer.zero_grad()
            (policy_loss + self.entropy_coeff * entropy_loss).backward()
            nn.utils.clip_grad_norm_(self.policy_func_new.parameters(), 0.5)
            self.policy_optimizer.step()
            # value loss TODO: could clip value loss
            values_b = self.value_function(s_b).squeeze()
            value_loss = (values_b - returns_b)**2
            value_loss = value_loss.mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_function.parameters(), 0.5)
            self.value_optimizer.step()
            # calculate kl divergence of new policy vs. old
            # kldiv = logstds_b - new_logstds +\
            #         0.5 * (new_logstds.exp()/logstds_b.exp())**2 +\
            #         0.5 * ((mus_b - new_mus) / logstds_b.exp())**2 - 0.5
            # kldiv = kldiv.sum(dim=-1).mean().item()
            kldiv = (logprobs_b - new_logprobs).mean().item()
            epoch += 1
            avg_policy_loss += policy_loss.mean().item()
        print(epoch)
        self.policy_func.load_state_dict(self.policy_func_new.state_dict())
        avg_policy_loss /= epoch
        self.train_steps += 1
        return avg_policy_loss, value_loss.item(), values.mean().item(), new_entropy.mean().item()
