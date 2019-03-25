import os
import time
import torch
import gym
import numpy as np
from gym import wrappers
from PIL import Image
from tensorboardX import SummaryWriter

from TD3_PER.td3 import TD3_PER
from TD3_PER.utils import mkdir
from TD3_PER.buffer import PrioritizedReplayBuffer


class TD3_PER_Trainer():

    def __init__(self, env_name, actor_config, critic_config, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 exp_noise_base=0.3, exp_noise_decay=0.0001, gamma=0.99, batch_size=1024,
                 polyak=0.9999, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 max_episodes=100000, max_timesteps=3000, max_buffer_length=5000000,
                 log_interval=5, threshold=None, lr_minimum=1e-10, exp_noise_minimum=1e-10,
                 record_videos=True, record_interval=100, alpha=0.9, beta_base=0.3, beta_multiplier=0.0001, log_dir='./log/'):

        self.algorithm_name = 'td3'
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

        self.actor_config = actor_config
        self.critic_config = critic_config
        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.exp_noise_base = exp_noise_base
        self.exp_noise_decay = exp_noise_decay
        self.exp_noise_minimum = exp_noise_minimum
        self.gamma = gamma
        self.batch_size = batch_size
        self.polyak = polyak
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.alpha = alpha
        self.beta_base = beta_base
        self.beta_multiplier = beta_multiplier

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.max_buffer_length = max_buffer_length
        self.log_interval = log_interval

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = TD3_PER(self.actor_config, self.critic_config, self.action_low, self.action_high)
        self.replay_buffer = PrioritizedReplayBuffer(size=self.max_buffer_length, alpha=self.alpha)

        self.reward_history = []
        self.make_plots = False

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def train(self):

        start_time = time.time()
        print("Action_space: {}".format(self.env.action_space))
        print("Obs_space: {}".format(self.env.observation_space))
        print("Threshold: {}".format(self.threshold))
        print("action_low: {} action_high: {} \n".format(self.action_low, self.action_high))

        # loading models
        self.policy.load(self.directory, self.filename)

        print("Training started ... \n")

        # training procedure:
        for episode in range(1, self.max_episodes + 1):

            # Only record video during evaluation, every n steps
            if episode % self.record_interval == 0:
                self.should_record = True

            ep_reward = 0.0
            state = self.env.reset()

            # calculate params
            exploration_noise = max(self.exp_noise_base / (1.0 + episode * self.exp_noise_decay),
                                    self.exp_noise_minimum)
            learning_rate = max(self.lr_base / (1.0 + episode * self.lr_decay), self.lr_minimum)
            beta = min(self.beta_base + episode * self.beta_multiplier, 1)
            self.policy.set_optimizers(lr=learning_rate)

            for t in range(self.max_timesteps):

                # select action and add exploration noise:
                action = self.policy.select_action(state)
                action = action + np.random.normal(0, exploration_noise, size=self.action_dim)
                action = action.clip(self.action_low, self.action_high)

                # take action in env:
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, float(done))
                state = next_state

                ep_reward += reward

                # if episode is done then update policy:
                if done or t == (self.max_timesteps - 1):
                    self.policy.update(self.replay_buffer, t, self.batch_size, self.gamma, self.polyak,
                                       self.policy_noise, self.noise_clip, self.policy_delay, beta)
                    break

            self.reward_history.append(ep_reward)
            avg_reward = np.mean(self.reward_history[-100:])


            # Calculate polyak
            # part = (env.spec.reward_threshold - avg_reward) / (env.spec.reward_threshold + 150)
            # if part > 1:
            #    part = 1
            # polyak = polyak_int[0] + (1 - part) * (polyak_int[1] - polyak_int[0])

            # Calculate LR
            # part = min((env.spec.reward_threshold - avg_reward) / (env.spec.reward_threshold + 150), 1)

            avg_actor_loss = np.mean(self.policy.actor_loss_list[-100:])
            avg_Q1_loss = np.mean(self.policy.Q1_loss_list[-100:])
            avg_Q2_loss = np.mean(self.policy.Q2_loss_list[-100:])

            # Truncate training history if we don't plan to plot it later
            if not self.make_plots:
                self.policy.truncate_loss_lists()
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)

                    # Print avg reward every log interval:
            if episode % self.log_interval == 0:
                self.policy.save(self.directory, self.filename)
                print(
                    "Ep:{:4d}   Rew:{:5.2f}  Avg Rew:{:5.2f}  LR:{:8.8f}  Bf:{:2.0f} Beta:{:0.4f}  EN:{:0.4f}  Loss: {:5.3f} {:5.3f} {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(), beta,
                        exploration_noise, avg_actor_loss, avg_Q1_loss, avg_Q2_loss))

            self.should_record = False

            # if avg reward > threshold then save and stop traning:
            if avg_reward >= self.threshold and episode > 100:
                print(
                    "Ep:{:4d}   Rew:{:5.2f}  Avg Rew:{:5.2f}  LR:{:8.8f}  Bf:{:2.0f} Beta:{:0.4f}  EN:{:0.4f}  Loss: {:5.3f} {:5.3f} {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(), beta,
                        exploration_noise, avg_actor_loss, avg_Q1_loss, avg_Q2_loss))
                print("########## Solved! ###########")
                name = self.filename + '_solved'
                self.policy.save(self.directory, name)
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
                action = self.policy.select_action(state)
                state, reward, done, _ = self.env.step(action)
                ep_reward += reward

                if save_gif:
                    img = self.env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))
                if done:
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            self.env.close()
