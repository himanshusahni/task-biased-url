import numpy as np
import math

class BoxWorld:

    def __init__(self):
        """
        for now 2x2 2D world with 4 goals evenly spread out.
        """
        # self.tasks = [(0.8, 0.8), (0.8, -0.8), (-0.8, -0.8), (-0.8, 0.8)]
        # self.tasks = [(0.8, 0.8), (0.3, 0.8), (-0.3, 0.8), (-0.8, 0.8)]
        # self.tasks = [(0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (-0.1, 0.1)]
        # self.tasks = 0.8
        self.max_steps = 25
        self.reset()

    def reset(self):
        """agent always starts in the middle"""
        self.agent = np.array((0.,0.))
        self.epstep = 0
        # self.sample_goal()
        return self.agent.copy()


    def step(self, a):
        """
        action is a 2D continuous value in [-0.1,0.1] that directly changes
        the agent's position
        return: agent's state, list of rewards according to all tasks, done
        """
        self.epstep += 1
        s1 = self.agent.copy()
        self.agent += a
        self.agent = np.clip(self.agent, -1, 1)
        done = self._compute_done()
        r = self._compute_rewards(s1, self.agent)
        return self.agent.copy(), r, done

    def scale_action(self, a):
        """
        input: action in [-1, 1]
        output: action in appropriate scale for environment
        """
        a = a.clip(-1, 1)
        a *= 0.1
        return a

    def sample_goal(self):
        """
        new goal from distribution of tasks
        """
        self.goal = (2 * np.random.random() - 1, self.tasks)

    def _compute_rewards(self, s1, s2):
        """
        input: s1 - state before action
               s2 - state after action
        rewards based on proximity to each goal
        """
        # r = []
        # for g in self.tasks:
        #     dist1 = np.linalg.norm(s1 - g)
        #     dist2 = np.linalg.norm(s2 - g)
        #     reward = dist1 - dist2
        #     # if dist2 < 0.05:
        #     #     reward += 1
        #     r.append(reward)
        #     # r.append(math.exp(-dist2))
        # dist1 = np.linalg.norm(s1 - self.goal)
        # dist2 = np.linalg.norm(s2 - self.goal)
        # return dist1 - dist2
        return 0
        # return r

    def _compute_done(self):
        """
        done is true if max steps have reached
        """
        done = (self.epstep >= self.max_steps)
        return done
