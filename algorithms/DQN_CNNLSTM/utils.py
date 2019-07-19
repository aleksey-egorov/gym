import os
import torch
import gym
import numpy as np
from collections import deque

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path



class FrameStack(gym.Wrapper):
    def __init__(self, env, stack_frames):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque([], maxlen=self.stack_frames)
        self.obs_norm = MaxMinFilter() #NormalizedEnv() alternative or can just not normalize observations as environment is already kinda normalized


    def reset(self):
        ob = self.env.reset()
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        self.frames.append(ob)
        return self.observation(), rew, done, info

    def observation(self):
        assert len(self.frames) == self.stack_frames
        st = np.stack(self.frames, axis=0)
        #st = np.swapaxes(st, 0, 1)
        st = np.expand_dims(st, axis=0)
        return st



class MaxMinFilter():
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

