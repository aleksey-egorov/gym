import gym
import os
import time
import torch
import numpy as np
from gym import wrappers
from PIL import Image
from itertools import count
from collections import namedtuple

from DuelingDDQN.ddqn import DuelingDDQN
from DuelingDDQN.utils import mkdir, ReplayBuffer


class DuelingDDQN_Trainer():

    def __init__(self, env_name, config, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 epsilon_base=0.3, epsilon_decay=0.0001, gamma=0.99, batch_size=1024,
                 max_episodes=100000, max_timesteps=3000, max_buffer_length=5000000,
                 log_interval=5, threshold=None, lr_minimum=1e-10, epsilon_minimum=1e-10,
                 record_videos=True, record_interval=100):

        self.algorithm_name = 'duel_ddqn'
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.record_videos = record_videos
        self.record_interval = record_interval
        if self.record_videos == True:
            videos_dir = mkdir('.', 'videos')
            monitor_dir = mkdir(videos_dir, self.algorithm_name)
            should_record = lambda i: self.should_record
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.should_record = False
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.config = config
        self.config[0][0]['dim'][0] = self.state_dim
        self.config[1][-1]['dim'][1] = self.action_dim


        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.epsilon_base = epsilon_base
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minimum
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.max_buffer_length = max_buffer_length
        self.log_interval = log_interval

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = DuelingDDQN(self.env, config)
        self.replay_buffer = ReplayBuffer(size=self.max_buffer_length)

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

        # logging variables:
        log_f = open("train_{}.txt".format(self.algorithm_name), "w+")

        print("\nTraining started ... ")

        # training procedure:
        for episode in range(self.max_episodes):

            # Only record video during evaluation, every n steps
            if episode % self.record_interval == 0:
                self.should_record = True

            ep_reward = 0.0
            state = self.env.reset()

            # calculate params
            epsilon = max(self.epsilon_base / (1.0 + episode * self.epsilon_decay), self.epsilon_minimum)
            learning_rate = max(self.lr_base / (1.0 + episode * self.lr_decay), self.lr_minimum)
            self.policy.set_optimizers(lr=learning_rate)

            for t in range(self.max_timesteps):

                action = self.policy.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, float(done))

                # Updating policy
                self.policy.update(self.replay_buffer, t, self.batch_size, self.gamma)

                state = next_state
                ep_reward += reward

                if done:
                    break

            self.reward_history.append(ep_reward)
            avg_reward = np.mean(self.reward_history[-100:])

            # logging updates:
            log_f.write('{},{}\n'.format(episode, ep_reward))
            log_f.flush()

            if len(self.policy.Q_loss_list) > 0:
                avg_Q_loss = np.mean(self.policy.Q_loss_list[-100:])

                # Truncate training history if we don't plan to plot it later
            if not self.make_plots:
                self.policy.truncate_loss_lists()
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                    # Print avg reward every log interval:
            if episode % self.log_interval == 0:
                self.policy.save(self.directory, self.filename)
                print \
                    ("Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EPS:{:0.4f}  Loss: {:8.6f}".format(
                    episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                    epsilon, avg_Q_loss))

            self.should_record = False

            # if avg reward > threshold then save and stop traning:
            if avg_reward >= self.threshold and episode > 100:
                print \
                    ("Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EPS:{:0.4f}  Loss: {:8.6f}".format(
                    episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                    epsilon, avg_Q_loss))
                print("########## Solved! ###########")
                name = self.filename + '_solved'
                self.policy.save(self.directory, name)
                log_f.close()
                self.env.close()
                training_time = time.time() - start_time
                print("Training time: {:6.2f} sec".format(training_time))
                break

    def test(self, episodes=3, render=True, save_gif=True):

        gifdir = mkdir('.' ,'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        for episode in range(1, episodes+1):
            ep_reward = 0.0
            state = self.env.reset()
            epdir = mkdir(algdir, str(episode))

            for t in range(self.max_timesteps):
                action = self.policy.select_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                ep_reward += reward

                if save_gif:
                    img = self.env.render(mode = 'rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))
                if done:
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            self.env.close()

