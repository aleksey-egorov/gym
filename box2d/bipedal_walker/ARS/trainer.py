import os
import time
import numpy as np
import gym
from gym import wrappers
from PIL import Image
from tensorboardX import SummaryWriter

from ARS.utils import mkdir


class ARSTrainer():
    def __init__(self,
                 env_name='BipedalWalker-v2',
                 max_episodes=30000,
                 max_timesteps=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 random_seed=1,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 record_videos=True, record_interval=100, log_interval=5, threshold=None, log_dir='./log/'):

        self.algorithm_name = 'ars'
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

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.random_seed = random_seed

        self.input_size = input_size or self.state_dim
        self.output_size = output_size or self.action_dim
        self.log_interval = log_interval

        self.rewards_list = []
        self.optimal_policy = None

        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = Policy(self.input_size, self.output_size, self.noise, self.learning_rate,
                             self.num_deltas, self.num_best_deltas)

        if not random_seed == None:
            print("Random Seed: {} \n".format(self.random_seed))
            self.env.seed(self.random_seed)
            np.random.seed(self.random_seed)

            # Explore the policy on one specific direction and over one episode

    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.max_timesteps:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self):
        start_time = time.time()
        print("Training started ... \n")
        print("action_space={}".format(self.env.action_space))
        print("obs_space={}".format(self.env.observation_space))
        print("threshold={}".format(self.threshold))
        print("action_low={} action_high={} \n".format(self.action_low, self.action_high))



        for episode in range(self.max_episodes):
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.num_deltas
            negative_rewards = [0] * self.num_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(self.num_deltas):
                positive_rewards[k] = self.explore(direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(direction="-", delta=deltas[k])

            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, sigma_rewards)

            # Only record video during evaluation, every n steps
            if episode % self.record_interval == 0:
                self.should_record = True

            # Play an episode with the new weights and print the score
            reward_evaluation = self.explore()

            self.rewards_list.append(reward_evaluation)
            avg_reward = np.mean(self.rewards_list[-100:])


            if episode % self.log_interval == 0:
                print('Step: {}  reward: {:4.2f} avg_reward: {:4.2f}  sigma: {:4.2f}'.format(
                    episode, reward_evaluation, avg_reward, sigma_rewards))

            self.should_record = False

            # if avg reward > threshold then save and stop traning:
            if avg_reward >= self.threshold:
                print("########## Solved! ###########")
                self.optimal_policy = self.policy
                training_time = time.time() - start_time
                print("Training time: {:6.2f} sec".format(training_time))
                break

    def test(self, direction=None, delta=None, episodes=3, render=True, save_gif=True):

        gifdir = mkdir('.', 'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            epdir = mkdir(algdir, str(episode))
            ep_reward = 0.0

            for t in range(self.max_timesteps):
                self.normalizer.observe(state)
                state = self.normalizer.normalize(state)
                action = self.optimal_policy.evaluate(state, delta, direction)
                state, reward, done, _ = self.env.step(action)
                reward = max(min(reward, 1), -1)
                ep_reward += reward

                if save_gif:
                    img = self.env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('{}/{}.jpg'.format(epdir, t))
                if done:
                    break

            print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))
            ep_reward = 0
            self.env.close()

class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Policy():
    def __init__(self, input_size, output_size, noise, learning_rate, num_deltas, num_best_deltas):
        self.theta = np.zeros((output_size, input_size))
        self.noise = noise
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.learning_rate / (self.num_best_deltas * sigma_rewards) * step