import torch

from dm_control import suite
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from agent import PPO

SEED = 123
NB_SKILLS = 4

random_state = np.random.RandomState(SEED)
torch.manual_seed(SEED)

env = suite.load('quadruped', 'walk', task_kwargs={'random':random_state})
action_spec = env.action_spec()
obs_spec = env.observation_spec()
STATE_DIM = obs_spec['egocentric_state'].shape[0]# + obs_spec['velocity'].shape[0]
STATE_DIM += NB_SKILLS
ACTION_DIM = action_spec.shape[0]

def scale_action(unscaled_action, action_spec):
    action = (unscaled_action + 1) / 2
    action *= (action_spec.maximum - action_spec.minimum)
    action += action_spec.minimum
    return action

ppo = PPO(STATE_DIM, ACTION_DIM, 0, 'cpu')
best = 1
if best:
    path = '../models/mujoco/ant/DIAYN/best_ppo_seed{}.pth'.format(SEED)
else:
    path = '../models/mujoco/ant/DIAYN/ppo_seed{}.pth'.format(SEED)
print("visualizing policy from ", path)
ppo.policy_func.load_state_dict(torch.load(path)['policy_func'])

# visualize the trained policy
def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
				   interval=interval, blit=True, repeat=False)
    return anim

for w in range(NB_SKILLS):
    w_onehot = np.zeros(NB_SKILLS)
    w_onehot[w] = 1
    frames = []
    time_step = env.reset()
    ep_step = 0
    while ep_step < 300:
        # build observation
        obs = np.concatenate((time_step.observation['egocentric_state'],
                              # time_step.observation['velocity'],
                              w_onehot))
        obs = torch.Tensor(obs)
        output = ppo.policy.forward(obs.unsqueeze(0))
        unscaled_action = output['action'].squeeze(-1).detach()
        action = scale_action(unscaled_action.numpy(), action_spec)
        time_step = env.step(action)
        camera0 = env.physics.render(camera_id=0, height=200, width=200)
        camera1 = env.physics.render(camera_id=1, height=200, width=200)
        frames.append(np.hstack((camera0, camera1)))
        ep_step += 1
    anim = display_video(frames)
    anim.save('skill{}.mp4'.format(w), fps=30)
