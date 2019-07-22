import os
import time
import torch
import numpy as np
from gym import wrappers
from PIL import Image
from tensorboardX import SummaryWriter

from Dueling_DDQN_PER_CNNLSTM import wrappers
from Dueling_DDQN_PER_CNNLSTM.ddqn import Dueling_DDQN_PER_CNNLSTM
from Dueling_DDQN_PER_CNNLSTM.utils import mkdir
from Dueling_DDQN_PER_CNNLSTM.buffer import PrioritizedReplayBuffer


class Dueling_DDQN_PER_CNNLSTM_Trainer():

    def __init__(self, env_name, config, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 epsilon_base=0.3, epsilon_decay=0.0001, gamma=0.99, batch_size=1024,
                 max_episodes=100000, max_timesteps=3000, max_buffer_length=5000000,
                 log_interval=5, threshold=None, lr_minimum=1e-10, epsilon_minimum=1e-10,
                 alpha=0.9, beta_base=0.3, beta_multiplier=0.0001, log_dir='./log/'):

        self.stack_frames = 4
        self.algorithm_name = 'duel_ddqn_cnnlstm'
        self.env_name = env_name
        self.env = wrappers.make_env(env_name) #gym.make(env_name)
        self.log_dir = os.path.join(log_dir, self.algorithm_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.algorithm_name + "_" + self.env_name)

        #self.state_dim = self.env.observation_space.shape
        self.state_dim = self.stack_frames
        self.action_dim = self.env.action_space.n
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.config = config

        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.epsilon_base = epsilon_base
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minimum
        self.gamma = gamma
        self.batch_size = batch_size

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

        self.policy = Dueling_DDQN_PER_CNNLSTM(self.env, self.state_dim, self.action_dim, self.batch_size)
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
        print("Threshold: {} \n".format(self.threshold))

        # loading models
        self.policy.load(self.directory, self.filename)

        print("Training started ... ")
        avg_Q_loss = 0.0

        # training procedure:
        for episode in range(self.max_episodes):

            ep_reward = 0.0
            state = self.env.reset()

            # calculate params
            epsilon = max(self.epsilon_base / (1.0 + episode * self.epsilon_decay), self.epsilon_minimum)
            learning_rate = max(self.lr_base / (1.0 + episode * self.lr_decay), self.lr_minimum)
            beta = min(self.beta_base + episode * self.beta_multiplier, 1)
            self.policy.set_optimizers(lr=learning_rate)

            for t in range(self.max_timesteps):

                action = self.policy.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, float(done))

                # Updating policy
                self.policy.update(self.replay_buffer, t, self.batch_size, self.gamma, beta)

                state = next_state
                ep_reward += reward

                if done:
                    break

            self.reward_history.append(ep_reward)
            avg_reward = np.mean(self.reward_history[-100:])

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
                print(
                    "Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EPS:{:0.4f}  Loss: {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                        epsilon, avg_Q_loss))

            self.writer.add_scalar("reward", ep_reward, episode)
            self.writer.add_scalar("avg_reward", avg_reward, episode)
            self.writer.add_scalar("avg_loss", avg_Q_loss, episode)


            # if avg reward > threshold then save and stop traning:
            if avg_reward >= self.threshold:
                print(
                    "Ep:{:5d}  Rew:{:8.2f}  Avg Rew:{:8.2f}  LR:{:8.8f}  Bf:{:2.0f}  EPS:{:0.4f}  Loss: {:5.3f}".format(
                        episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(),
                        epsilon, avg_Q_loss))
                print("########## Solved! ###########")
                name = self.filename + '_solved'
                self.policy.save(self.directory, name)
                training_time = time.time() - start_time
                print("Training time: {:6.2f} sec".format(training_time))
                self.env.close()
                break

    def test(self, episodes=3, save_gif=True):

        gifdir = mkdir('.', 'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        # loading models
        self.policy.load(self.directory, self.filename)

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

