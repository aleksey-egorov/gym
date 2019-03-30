import time
import torch
import torch.multiprocessing as mp
import numpy as np
import os

from A3C_Cnt.model import A3C_MLP, A3C_CONV
from A3C_Cnt.env import create_env
from A3C_Cnt.test import test
from A3C_Cnt.train import train
from A3C_Cnt.shared_optim import SharedRMSprop, SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"


class A3C_Cnt_Trainer():

    def __init__(self, env_name, random_seed=1, lr=0.0001,
                   gamma=0.99, tau=0.99, workers=32, num_steps=10,
                   max_episode_length=100000, shared_optimizer=True, save_max=True,
                   optimizer=False, model='MLP', stack_frames=1, gpu_ids=-1, amsgrad=True, threshold=None):

        self.algorithm_name = 'a3c_cnt'
        self.env_name = env_name
        self.stack_frames = stack_frames
        self.env = env = create_env(self.env_name, self.stack_frames)

        self.threshold = threshold
        self.random_seed = random_seed
        self.shared_optimizer = shared_optimizer
        self.optimizer = optimizer
        self.lr = lr
        self.amsgrad = amsgrad
        self.workers = workers
        self.gamma = gamma
        self.tau = tau
        self.num_steps = num_steps
        self.model = model
        self.save_max = save_max
        self.max_episode_length = max_episode_length
        self.log_dir = './log/'
        self.save_dir = './saved_models/'

        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        if gpu_ids == -1:
            self.gpu_ids = [-1]
        else:
            self.gpu_ids = gpu_ids
            torch.cuda.manual_seed(self.random_seed)

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        if self.model == 'MLP':
            self.shared_model = A3C_MLP(env.observation_space.shape[0], env.action_space, self.stack_frames)
        if self.model == 'CONV':
            self.shared_model = A3C_CONV(self.stack_frames, env.action_space)

        self.shared_model.share_memory()

        if self.shared_optimizer:
            if self.optimizer == 'RMSprop':
                self.optimizer = SharedRMSprop(self.shared_model.parameters(), lr=self.lr)
            if self.optimizer == 'Adam':
                self.optimizer = SharedAdam(self.shared_model.parameters(), lr=self.lr, amsgrad=self.amsgrad)
            self.optimizer.share_memory()
        else:
            self.optimizer = None


    def train(self):

        print ("Training started ... ")
        args = {
            'env': self.env_name,
            'gpu_ids': self.gpu_ids,
            'log_dir': self.log_dir,
            'seed': self.random_seed,
            'stack_frames': self.stack_frames,
            'model': self.model,
            'save_max': self.save_max,
            'save_model_dir': self.save_dir,
            'lr': self.lr,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'tau': self.tau,
            'max_episode_length': self.max_episode_length,
            'render': False,
            'save_gif': False,
            'threshold': self.threshold
        }
        self.processes = []

        p = mp.Process(target=test, args=(args, self.shared_model))
        p.start()
        self.processes.append(p)
        time.sleep(0.1)
        for rank in range(0, self.workers):
            p = mp.Process(target=train, args=(
                rank, args, self.shared_model, self.optimizer))
            p.start()
            self.processes.append(p)
            time.sleep(0.1)
        for p in self.processes:
            time.sleep(0.1)
            p.join()

    def test(self):

        saved_state = torch.load('{0}{1}.dat'.format(self.save_dir, self.env_name), map_location=lambda storage, loc: storage)
        self.shared_model.load_state_dict(saved_state)

        args = {
            'env': self.env_name,
            'gpu_ids': self.gpu_ids,
            'log_dir': self.log_dir,
            'seed': self.random_seed,
            'stack_frames': self.stack_frames,
            'model': self.model,
            'save_max': self.save_max,
            'save_model_dir': self.save_dir,
            'lr': self.lr,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'tau': self.tau,
            'max_episode_length': self.max_episode_length,
            'render': True,
            'save_gif': True,
            'threshold': self.threshold
        }
        test(args, self.shared_model)