from dm_control import suite

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import time
from collections import deque

import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn

from agent import PPO, GaussianPolicy, SkillDiscriminator


SEED = 123
NB_THREADS = 4
NB_ROLLOUT_STEPS = 256
MAX_TRAIN_STEP = 10000
NB_VARIATIONAL_UPDATES = 5
VARIATIONAL_BATCH_SIZE = 128
DEVICE = 'cuda:0'
NB_SKILLS = 4

np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env():
    random_state = np.random.RandomState(np.random.randint(1000))
    return suite.load('quadruped', 'walk',
                      task_kwargs={'random':random_state})

env = make_env()
action_spec = env.action_spec()
obs_spec = env.observation_spec()
STATE_DIM = obs_spec['egocentric_state'].shape[0]# + obs_spec['velocity'].shape[0]
DIM = STATE_DIM
STATE_DIM += NB_SKILLS
ACTION_DIM = action_spec.shape[0]


class Clone:
    def __init__(self, t, env, policy, nb_steps, rollout):
        self.t = t
        self.env = env
        self.action_spec = env.action_spec()
        self.policy = policy
        self.nb_rollout_steps = nb_steps
        self.rollout = rollout
        self.random_state = np.random.RandomState(np.random.randint(1000))

    def scale_action(self, unscaled_action):
        action = (unscaled_action + 1) / 2
        action *= (self.action_spec.maximum - self.action_spec.minimum)
        action += self.action_spec.minimum
        return action

    def run(self, startq, stopq):
        done = True
        while True:
            startq.get()
            for step in range(self.nb_rollout_steps):
                if done:
                    # sample a random skill
                    w = self.random_state.randint(0, NB_SKILLS)
                    w_onehot = np.zeros(NB_SKILLS)
                    w_onehot[w] = 1
                    time_step = self.env.reset()
                    obs = np.concatenate((time_step.observation['egocentric_state'],
                                          # time_step.observation['velocity'],
                                          w_onehot))
                    obs = torch.Tensor(obs).to(DEVICE)
                    ep_steps = 0
                output = self.policy.forward(obs.unsqueeze(0))
                unscaled_action = output['action'].squeeze(-1).detach()
                action = self.scale_action(unscaled_action.cpu().numpy())
                time_step = self.env.step(action)
                ep_steps += 1
                if ep_steps > 255:
                    done = True
                else:
                    done = time_step.last()
                self.rollout['states'][self.t, step] = obs
                self.rollout['actions'][self.t, step] = unscaled_action
                self.rollout['rewards'][self.t, step] = time_step.reward
                self.rollout['dones'][self.t, step] = float(done)
                obs = np.concatenate((time_step.observation['egocentric_state'],
                                      # time_step.observation['velocity'],
                                      w_onehot))
                obs = torch.Tensor(obs).to(DEVICE)
            self.rollout['states'][self.t, step+1] = obs
            stopq.put(self.t)

if __name__ == '__main__':
    mp.set_start_method("spawn")

    d = SkillDiscriminator(DIM, NB_SKILLS)
    discriminator_optimizer = optim.Adam(d.parameters(), lr=1e-4)
    CE = nn.CrossEntropyLoss()

    class DiscriminatorBuffer:
        def __init__(self, maxlen):
            self.buffer = deque([], maxlen=maxlen)

        def append(self, states):
            # strip the skills from the states
            states, w_onehots = states.split([DIM, NB_SKILLS], dim=-1)
            # get the skills from the onehots
            ws = w_onehots.argmax(-1)
            # flatten each
            states = states.view(-1, DIM)
            ws = ws.flatten()
            # insert into the variational buffer
            for i in range(states.shape[0]):
                self.buffer.append((states[i], ws[i]))

        def sample(self):
            idxs = np.random.randint(
                len(self.buffer), size=(VARIATIONAL_BATCH_SIZE))
            batch = [self.buffer[i] for i in idxs]
            batch = zip(*batch)
            return batch

    discriminator_buffer = DiscriminatorBuffer(10000)

    def discriminator_update(batch):
        s, w = batch
        s = torch.stack(s)
        w = torch.stack(w)
        logits, _ = d(torch.Tensor(s))
        discriminator_loss = CE(logits, torch.LongTensor(w))
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        return discriminator_loss.item()


    ppo = PPO(STATE_DIM, ACTION_DIM, MAX_TRAIN_STEP, DEVICE)
    ppo.policy_func.share_memory()
    rollout = {
        'states': torch.empty(
            NB_THREADS,
            NB_ROLLOUT_STEPS + 1,
            STATE_DIM).to(DEVICE).share_memory_(),
        'actions': torch.empty(
            NB_THREADS,
            NB_ROLLOUT_STEPS,
            ACTION_DIM).to(DEVICE).share_memory_(),
        'rewards': torch.empty(
            NB_THREADS,
            NB_ROLLOUT_STEPS).to(DEVICE).share_memory_(),
        'dones': torch.empty(
            NB_THREADS,
            NB_ROLLOUT_STEPS).to(DEVICE).share_memory_(),
    }
    metrics = {'rewards': [],
               'policy_loss': [],
               'value_loss': [],
               'entropy': [],
               'value': [],
               'discriminator_loss': []}
    max_running_reward = -10000
    startqs = []
    stopqs = []
    procs = []
    for t in range(NB_THREADS):
        startq = mp.Queue(1)
        startqs.append(startq)
        stopq = mp.Queue(1)
        stopqs.append(stopq)
        policy = GaussianPolicy(ppo.policy_func)
        c = Clone(t, make_env(), policy, NB_ROLLOUT_STEPS, rollout)
        proc = mp.Process(target=c.run, args=(startq, stopq))
        procs.append(proc)
        proc.start()

    for step in range(MAX_TRAIN_STEP):
        start_time = time.time()
        # start collecting data
        for start in startqs:
            start.put(1)
        for stop in stopqs:
            stop.get()
        # # recompute rewards using diversity
        # states, w_onehots = rollout['states'].split([DIM, NB_SKILLS], dim=-1)
        # _, dr = d(torch.Tensor(states))
        # diversity_reward = (dr * w_onehots).sum(-1).detach() / 10
        # diversity_reward = diversity_reward[:, 1:]
        # rollout['rewards'] = diversity_reward
        # do RL
        policy_loss, value_loss, value, entropy = ppo.update(rollout)
        metrics['policy_loss'].append(policy_loss)
        metrics['value_loss'].append(value_loss)
        metrics['entropy'].append(entropy)
        metrics['value'].append(value)
        metrics['rewards'].append(rollout['rewards'].mean().item())
        print("STEP ", step, "TIME = {:.2f}, STEP REWARD = {:.2f}".format(
            time.time()-start_time, metrics['rewards'][-1]))
        # ############################# Discriminator ############################
        # discriminator_buffer.append(rollout['states'][:, :-1].clone())
        # if step > 5:
        #     for _ in range(NB_VARIATIONAL_UPDATES):
        #         batch = discriminator_buffer.sample()
        #         discriminator_loss = discriminator_update(batch)
        #         metrics['discriminator_loss'].append(discriminator_loss)
        ################################  SAVING ###############################
        path = '../models/mujoco/ant/DIAYN/'
        if max_running_reward < sum(metrics['rewards'][-10:])/10:
            print("NEW MAX RUNNING REWARD!: ", sum(metrics['rewards'][-10:])/10)
            max_running_reward = sum(metrics['rewards'][-10:])/10
            torch.save(ppo.to_save(), path+'best_ppo_seed{}.pth'.format(SEED))
            torch.save(
                d.state_dict(), path+'best_variational_seed{}.pth'.format(SEED))
        if step % 10 == 0:
            torch.save(ppo.to_save(), path+'ppo_seed{}.pth'.format(SEED))
            torch.save(
                d.state_dict(), path+'best_variational_seed{}.pth'.format(SEED))
            pickle.dump(
                metrics, open(path+'metrics_seed{}.pkl'.format(SEED), 'wb'))
# terminate all processes
    for p in procs:
        p.terminate()
