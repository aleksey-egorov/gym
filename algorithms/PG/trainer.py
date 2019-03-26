import gym
import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from gym import wrappers
from PIL import Image
from itertools import count
from collections import namedtuple
import ptan

from PG.pg import PG
from PG.utils import mkdir
from PG.buffer import MeanBuffer



class PG_Trainer():

    def __init__(self, env_name, config, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 gamma=0.99, batch_size=32,
                 max_episodes=100000, max_timesteps=3000,
                 log_interval=5, threshold=None, lr_minimum=1e-10,
                 entropy_beta=0.01, bellman_steps=10, baseline_steps=50000, log_dir='./log/'):

        self.algorithm_name = 'pg'
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.log_dir = os.path.join(log_dir, self.algorithm_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.algorithm_name + "_" + self.env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.should_record = False
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.config = config
        self.config[0]['dim'][0] = self.state_dim
        self.config[-1]['dim'][1] = self.action_dim


        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.gamma = gamma
        self.batch_size = batch_size

        self.entropy_beta = entropy_beta
        self.bellman_steps = bellman_steps
        self.baseline_steps = baseline_steps

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.log_interval = log_interval


        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = PG(self.env, self.config, self.gamma, self.bellman_steps)

        # The experience source interacts with the environment and returns (s,a,r,s') transitions
        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.policy.ptan_agent,
                                                                    gamma=self.gamma,
                                                                    steps_count=self.bellman_steps)

        self.baseline_buffer = MeanBuffer(self.baseline_steps)


        self.reward_history = []
        self.make_plots = False

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def train(self):

        start_time = time.time()
        print("action_space={}".format(self.env.action_space))
        print("obs_space={}".format(self.env.observation_space))
        print("threshold={} \n".format(self.threshold))

        # loading models
        self.policy.load(self.directory, self.filename)


        print("\nTraining started ... ")
        avg_loss = 0
        total_rewards = []
        step_rewards = []
        step_idx = 0
        episode = 0

        batch_states, batch_actions, batch_scales = [], [], []
        learning_rate = self.lr_base
        self.policy.set_optimizers(lr=learning_rate)

        # each iteration runs one action in the environment and returns a (s,a,r,s') transition
        for step_idx, exp in enumerate(self.exp_source):
            self.baseline_buffer.add(exp.reward)
            baseline = self.baseline_buffer.mean()
            self.writer.add_scalar("baseline", baseline, step_idx)

            batch_states.append(exp.state)
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline)

            # handle when an episode is completed
            episode_rewards = self.exp_source.pop_total_rewards()
            if episode_rewards:
                episode += 1
                reward = episode_rewards[0]
                total_rewards.append(reward)
                avg_reward = float(np.mean(total_rewards[-100:]))

                if len(self.policy.loss_list) > 0:
                    avg_loss = np.mean(self.policy.loss_list[-100:])

                    # Print avg reward every log interval:
                if episode % self.log_interval == 0:
                    self.policy.save(self.directory, self.filename)
                    print("Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Loss: {:8.6f}".format(
                        episode, reward, avg_reward, learning_rate, avg_loss))


                learning_rate = max(self.lr_base / (1.0 + episode * self.lr_decay), self.lr_minimum)


                self.writer.add_scalar("reward", reward, step_idx)
                self.writer.add_scalar("reward_100", avg_reward, step_idx)
                self.writer.add_scalar("episodes", episode, step_idx)

                # if avg reward > threshold then save and stop traning:
                if avg_reward >= self.threshold and episode > 100:
                    print("Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Loss: {:8.6f}".format(
                        episode, reward, avg_reward, learning_rate, avg_loss))
                    print("########## Solved! ###########")
                    name = self.filename + '_solved'
                    self.policy.save(self.directory, name)
                    self.env.close()
                    training_time = time.time() - start_time
                    print("Training time: {:6.2f} sec".format(training_time))
                    break

            if len(batch_states) < self.batch_size:
                continue

            scalars = self.policy.update(batch_states, batch_actions, batch_scales, self.batch_size, self.entropy_beta)
            entropy_v, entropy_loss_v, loss_policy_v, loss_v, grad_means, grad_count, grad_max = scalars

            self.writer.add_scalar("baseline", baseline, step_idx)
            self.writer.add_scalar("entropy", entropy_v.item(), step_idx)
            self.writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            self.writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            self.writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
            self.writer.add_scalar("loss_total", loss_v.item(), step_idx)
            self.writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            self.writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()



    def test(self, episodes=3, render=True, save_gif=True):

        gifdir = mkdir('.' ,'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        t = 0
        for episode in range(1, episodes+1):
            ep_reward = 0.0
            epdir = mkdir(algdir, str(episode))

            for step_idx, exp in enumerate(self.exp_source):
                self.baseline_buffer.add(exp.reward)
                baseline = self.baseline_buffer.mean()

                if save_gif:
                    img = self.env.render(mode = 'rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))
                    t+= 1

                # handle when an episode is completed
                episode_rewards = self.exp_source.pop_total_rewards()
                if episode_rewards:
                    ep_reward = episode_rewards[0]
                    t = 0
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            self.env.close()

