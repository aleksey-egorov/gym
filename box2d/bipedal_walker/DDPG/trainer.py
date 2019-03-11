import gym
import os
import time
import torch
import numpy as np
from gym import wrappers
from PIL import Image
from tensorboardX import SummaryWriter

from DDPG.ddpg import DDPG
from DDPG.utils import ReplayBuffer, mkdir


class DDPG_Trainer():

    def __init__(self, env_name, actor_config, critic_config, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 exp_noise_base=0.3, exp_noise_decay=0.0001, exploration_mu=0, exploration_theta=0.15,
                 exploration_sigma=0.2, gamma=0.99, batch_size=1024, polyak=0.9999,
                 max_episodes=100000, max_timesteps=3000, max_buffer_length=5000000,
                 log_interval=5, threshold=None, lr_minimum=1e-10, exp_noise_minimum=1e-10,
                 record_videos=True, record_interval=100, log_dir='./log/'):

        self.algorithm_name = 'ddpg'
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.log_dir = os.path.join(log_dir, self.algorithm_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.algorithm_name + "_" + self.env_name)

        self.record_videos = record_videos
        self.record_interval = record_interval
        if self.record_videos == True:
            videos_dir = mkdir('.', 'videos')
            monitor_dir = mkdir(videos_dir, self.algorithm_name)
            should_record = lambda i: self.should_record
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.should_record = False
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.actor_config = actor_config
        self.critic_config = critic_config
        self.actor_config[0]['dim'][0] = self.state_dim
        self.actor_config[-1]['dim'][1] = self.action_dim
        self.critic_config[0]['dim'][0] = self.state_dim + self.action_dim

        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.exp_noise_base = exp_noise_base
        self.exp_noise_decay = exp_noise_decay
        self.exp_noise_minimum = exp_noise_minimum
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.gamma = gamma
        self.batch_size = batch_size
        self.polyak = polyak
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.max_buffer_length = max_buffer_length
        self.log_interval = log_interval

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = DDPG(self.actor_config, self.critic_config, self.action_dim, self.action_low, self.action_high,
                           self.exploration_mu, exploration_theta, exploration_sigma)
        self.replay_buffer = ReplayBuffer(max_length=self.max_buffer_length)

        self.reward_history = []
        self.make_plots = False

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def train(self):

        start_time = time.time()
        print("Training started ... \n")
        print("action_space={}".format(self.env.action_space))
        print("obs_space={}".format(self.env.observation_space))
        print("threshold={}".format(self.threshold))
        print("action_low={} action_high={} \n".format(self.action_low, self.action_high))

        # loading models
        self.policy.load(self.directory, self.filename)

        avg_actor_loss = 0.0
        avg_critic_loss = 0.0

        # training procedure:
        for episode in range(self.max_episodes):

            # Only record video during evaluation, every n steps
            if episode % self.record_interval == 0:
                self.should_record = True

            ep_reward = 0.0
            state = self.env.reset()

            # calculate params
            noise_coeff = max(self.exp_noise_base / (1.0 + episode * self.exp_noise_decay), self.exp_noise_minimum)
            learning_rate = max(self.lr_base / (1.0 + episode * self.lr_decay), self.lr_minimum)
            self.policy.set_optimizers(lr=learning_rate)

            for t in range(self.max_timesteps):

                action = self.policy.select_action(state, noise_coeff)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, float(done)))

                # Updating policy
                self.policy.update(self.replay_buffer, self.batch_size, self.gamma, self.polyak)

                state = next_state
                ep_reward += reward

                if done:
                    break

            self.reward_history.append(ep_reward)
            avg_reward = np.mean(self.reward_history[-100:])

            if len(self.policy.actor_loss_list) > 0:
                avg_actor_loss = np.mean(self.policy.actor_loss_list[-100:])
                avg_critic_loss = np.mean(self.policy.critic_loss_list[-100:])

            if not self.make_plots and len(self.policy.actor_loss_list) > 200:
                self.policy.actor_loss_list.pop(0)
                self.policy.critic_loss_list.pop(0)
                self.reward_history.pop(0)

                # Print avg reward every log interval:
            if episode % self.log_interval == 0:
                self.policy.save(self.directory, self.filename)
                print(
                    "Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EN:{:0.4f}  Loss: {:5.3f} {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                        noise_coeff, avg_actor_loss, avg_critic_loss))

            self.should_record = False

            # if avg reward > threshold then save and stop traning:
            if avg_reward >= self.threshold and episode > 100:
                print(
                    "Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EN:{:0.4f}  Loss: {:5.3f} {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                        noise_coeff, avg_actor_loss, avg_critic_loss))
                print("########## Solved! ###########")
                name = self.filename + '_solved'
                self.policy.save(self.directory, name)
                self.env.close()
                training_time = time.time() - start_time
                print("Training time: {:6.2f} sec".format(training_time))
                break

    def test(self, episodes=3, render=True, save_gif=True):

        gifdir = mkdir('.', 'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        for episode in range(1, episodes + 1):
            ep_reward = 0.0
            state = self.env.reset()
            epdir = mkdir(algdir, str(episode))

            for t in range(self.max_timesteps):
                action = self.policy.select_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                ep_reward += reward

                if save_gif:
                    img = self.env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))
                if done:
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            self.env.close()
