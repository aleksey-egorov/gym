import os
import gym
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image
from gym import wrappers

from PPO.ppo import PPO
from PPO.utils import mkdir
from PPO.multiprocessing_env import SubprocVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO_Trainer():

    def __init__(self, env_name, hidden_size=256, num_envs=8, random_seed=42, lr_base=0.0001, lr_decay=0.0001,
                 lr_minimum=1e-7, gamma=0.99, gae_lambda=0.95, ppo_epsilon=0.2, critic_discount=0.5,
                 batch_size=32, entropy_beta=0.001, ppo_steps=256, ppo_epochs=10,
                 test_epochs=10, num_tests=10, log_interval=10, threshold=None,
                 log_dir='./log/'):

        self.algorithm_name = 'ppo'
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs_unwrapped = [self.make_env() for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs_unwrapped)
        self.env = gym.make(self.env_name)

        self.test_env = gym.make(env_name)
        self.log_dir = os.path.join(log_dir, self.algorithm_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.algorithm_name + "_" + self.env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.hidden_size = hidden_size

        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum

        self.gae_lambda = gae_lambda
        self.ppo_epsilon = ppo_epsilon
        self.entropy_beta = entropy_beta
        self.critic_discount = critic_discount
        self.gamma = gamma
        self.ppo_steps = ppo_steps
        self.ppo_epochs = ppo_epochs
        self.test_epochs = test_epochs
        self.num_tests = num_tests
        self.max_timesteps = 1000000

        self.batch_size = batch_size
        self.log_interval = log_interval
        self.videos_dir = mkdir('.', 'videos')

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = PPO(self.state_dim, self.action_dim, self.hidden_size, self.entropy_beta, self.gamma,
                          self.gae_lambda, self.batch_size, self.ppo_epsilon, self.ppo_epochs, self.critic_discount)

        self.reward_history = []
        self.make_plots = False

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def make_env(self):
        # returns a function which creates a single environment
        def _thunk():
            env = gym.make(self.env_name)
            return env

        return _thunk

    def train(self):

        start_time = time.time()
        print("Envs number: {}".format(self.num_envs))
        print("Action_space: {}".format(self.env.action_space))
        print("Obs_space: {}".format(self.env.observation_space))
        print("Threshold: {}".format(self.threshold))
        print("action_low: {} action_high: {} \n".format(self.action_low, self.action_high))

        # loading models
        self.policy.load(self.directory, self.filename)
        self.policy.set_optimizers(lr=self.lr_base)

        print("Training started ... \n")

        frame_idx = 0
        train_epoch = 0
        best_reward = None

        state = self.envs.reset()
        early_stop = False

        while not early_stop:

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []

            for _ in range(self.ppo_steps):
                state = torch.FloatTensor(state).to(device)
                dist, value = self.policy.model(state)
                action = dist.sample()

                # each state, reward, done is a list of results from each parallel environment
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
                log_prob = dist.log_prob(action)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

                states.append(state)
                actions.append(action)

                state = next_state
                frame_idx += 1


            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.policy.model(next_state)
            returns = self.policy.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values
            advantage = self.policy.normalize(advantage)

            res = self.policy.ppo_update(states, actions, log_probs, returns, advantage)
            sum_returns_m, sum_advantage_m, sum_loss_actor_m, sum_loss_critic_m, sum_entropy_m, sum_loss_total_m = res

            self.writer.add_scalar("returns", sum_returns_m, frame_idx)
            self.writer.add_scalar("advantage", sum_advantage_m, frame_idx)
            self.writer.add_scalar("loss_actor", sum_loss_actor_m, frame_idx)
            self.writer.add_scalar("loss_critic", sum_loss_critic_m, frame_idx)
            self.writer.add_scalar("entropy", sum_entropy_m, frame_idx)
            self.writer.add_scalar("loss_total", sum_loss_total_m, frame_idx)

            train_epoch += 1
            self.policy.save(self.directory, self.filename)



            if train_epoch % self.test_epochs == 0:
                test_reward = np.mean([self.policy.test_env(self.env, device) for _ in range(self.num_tests)])
                self.writer.add_scalar("test_rewards", test_reward, frame_idx)
                print('Frame %s. reward: %s' % (frame_idx, test_reward))
                # Save a checkpoint every time we achieve a best reward
                if best_reward is None or best_reward < test_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                        fname = "%s_best_%+.3f_%d.dat" % (self.env_name, test_reward, frame_idx)
                        self.policy.save(self.directory, fname)
                    best_reward = test_reward
                if test_reward > self.threshold:
                    print("########## Solved! ###########")
                    name = self.filename + '_solved'
                    self.policy.save(self.directory, name)
                    training_time = time.time() - start_time
                    print("Training time: {:6.2f} sec".format(training_time))
                    early_stop = True


    def test(self, episodes=3, save_gif=True):

        gifdir = mkdir('.', 'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        #videos_dir = mkdir('.', 'videos')
        #monitor_dir = mkdir(videos_dir, self.algorithm_name)
        #should_record = lambda i: True
        #env = wrappers.Monitor(env, monitor_dir, video_callable=should_record, force=True)

        # loading models
        self.policy.load(self.directory, self.filename)
        states_pr = torch.zeros((self.num_envs, self.state_dim))


        for episode in range(1, episodes + 1):

            state = self.env.reset()
            ep_reward = 0.0
            total_steps = 0
            epdir = mkdir(algdir, str(episode))

            for t in range(self.max_timesteps):
                state = torch.FloatTensor(state).reshape(1, -1).expand_as(states_pr).to(device)
                dist, value = self.policy.model(state)
                action = dist.sample()

                # each state, reward, done is a list of results from each parallel environment
                next_state, reward, done, _ = self.env.step(action[0].cpu().numpy())
                state = next_state
                ep_reward += reward
                total_steps += 1

                if save_gif:
                    img = self.env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))

                if done:
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            self.env.close()
