from dm_control import suite
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import time

import torch
import torch.multiprocessing as mp

from agent import PPO, GaussianPolicy


SEED = 123
NB_THREADS = 4
NB_ROLLOUT_STEPS = 1000
MAX_TRAIN_STEP = 10000
DEVICE = 'cpu'

np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env():
    random_state = np.random.RandomState(np.random.randint(1000))
    return suite.load('cartpole', 'swingup',
                      task_kwargs={'random':random_state})

env = make_env()
action_spec = env.action_spec()
obs_spec = env.observation_spec()
STATE_DIM = obs_spec['position'].shape[0] + obs_spec['velocity'].shape[0]
ACTION_DIM = action_spec.shape[0]


class Clone:
    def __init__(self, t, env, policy, nb_steps, rollout):
        self.t = t
        self.env = env
        self.action_spec = env.action_spec()
        self.policy = policy
        self.nb_rollout_steps = nb_steps
        self.rollout = rollout

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
                    time_step = self.env.reset()
                    obs = np.concatenate((time_step.observation['position'],
                                          time_step.observation['velocity']))
                    obs = torch.Tensor(obs).to(DEVICE)
                output = self.policy.forward(obs.unsqueeze(0))
                unscaled_action = output['action'].squeeze(-1).detach()
                action = self.scale_action(unscaled_action.cpu().numpy())
                time_step = self.env.step(action)
                done = time_step.last()
                self.rollout['states'][self.t, step] = obs
                self.rollout['actions'][self.t, step] = unscaled_action
                self.rollout['rewards'][self.t, step] = time_step.reward
                self.rollout['dones'][self.t, step] = float(done)
                obs = np.concatenate((time_step.observation['position'],
                                      time_step.observation['velocity']))
                obs = torch.Tensor(obs).to(DEVICE)
            self.rollout['states'][self.t, step+1] = obs
            stopq.put(self.t)

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

if __name__ == '__main__':
    mp.set_start_method("spawn")
    ppo = PPO(STATE_DIM, ACTION_DIM, MAX_TRAIN_STEP, DEVICE)
    ppo.policy_func.share_memory()

    metrics = {'rewards': [],
               'policy_loss': [],
               'value_loss': [],
               'entropy': [],
               'value': [],}
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
        # update!
        policy_loss, value_loss, value, entropy = ppo.update(rollout)
        metrics['policy_loss'].append(policy_loss)
        metrics['value_loss'].append(value_loss)
        metrics['entropy'].append(entropy)
        metrics['value'].append(value)
        metrics['rewards'].append(rollout['rewards'].mean().item())
        print("STEP ", step, "TIME = {:.2f}, STEP REWARD = {:.2f}".format(
            time.time()-start_time, metrics['rewards'][-1]))
        ################################  SAVING ###################################
        path = '../models/mujoco/pendulum/ppo/'
        if max_running_reward < sum(metrics['rewards'][-10:])/10:
            print("NEW MAX RUNNING REWARD!: ", sum(metrics['rewards'][-10:])/10)
            max_running_reward = sum(metrics['rewards'][-10:])/10
            torch.save(ppo.to_save(), path+'best_ppo_seed{}.pth'.format(SEED))
        if step % 10 == 0:
            torch.save(ppo.to_save(), path+'ppo_seed{}.pth'.format(SEED))
            pickle.dump(metrics, open(path+'metrics_seed{}.pkl'.format(SEED), 'wb'))
# terminate all processes
    for p in procs:
        p.terminate()
