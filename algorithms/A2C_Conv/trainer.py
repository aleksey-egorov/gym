import os
import gym
import time
import ptan
import torch
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image

from A2C_Conv.a2c import A2C_Conv
from A2C_Conv.utils import mkdir, RewardTracker



class A2C_Conv_Trainer():

    def __init__(self, env_name, num_envs=10, hidden_size=256, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 batch_size=32, max_episodes=10000, max_timesteps=10000,
                 entropy_beta=1e-4, gamma=0.99, bellman_steps=4, clip_grad=0.1,
                 log_interval=5, threshold=None, test_iters=10000, lr_minimum=1e-10,
                 log_dir='./log/'):

        self.algorithm_name = 'a2c_conv'
        self.env_name = env_name
        self.num_envs = num_envs
        self.make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make(self.env_name))
        self.envs = [self.make_env() for _ in range(self.num_envs)]
        self.env = self.envs[0]

        self.log_dir = os.path.join(log_dir, self.algorithm_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.algorithm_name + "_" + self.env_name)

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        if not threshold == None:
            self.threshold = threshold
        else:
            self.threshold = self.env.spec.reward_threshold

        self.hidden_size = hidden_size

        self.random_seed = random_seed
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps

        self.entropy_beta = entropy_beta
        self.test_iters = test_iters
        self.gamma = gamma
        self.bellman_steps = bellman_steps
        self.clip_grad = clip_grad

        self.batch_size = batch_size
        self.log_interval = log_interval
        self.videos_dir = mkdir('.', 'videos')

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = A2C_Conv(self.state_dim, self.action_dim, self.entropy_beta, self.gamma, self.bellman_steps, self.batch_size, self.clip_grad)
        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.envs, self.policy.agent, gamma=self.gamma, steps_count=self.bellman_steps)

        self.reward_history = []
        self.make_plots = False

        if self.random_seed:
            print("Random Seed: {}".format(self.random_seed))
            self.env.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

    def train(self):

        start_time = time.time()
        print("Envs number: {}".format(self.num_envs))
        print("Action_space: {}".format(self.env.action_space))
        print("Obs_space: {}".format(self.env.observation_space))
        print("Threshold: {} \n".format(self.threshold))

        # loading models
        self.policy.load(self.directory, self.filename)
        self.policy.set_optimizers(lr=self.lr_base)

        print("Training started ... \n")

        batch = []
        avg_reward = 0.0
        episode = 0
        with RewardTracker(self.writer, stop_reward=self.threshold) as tracker:
            with ptan.common.utils.TBMeanTracker(self.writer, batch_size=100) as tb_tracker:

                for step_idx, exp in enumerate(self.exp_source):

                    batch.append(exp)

                    # handle new rewards
                    new_rewards = self.exp_source.pop_total_rewards()
                    if new_rewards:
                        finished, save_checkpoint = tracker.reward(new_rewards[0], step_idx)
                        if finished:
                            self.reward_history.append(new_rewards[0])
                            avg_reward = np.mean(self.reward_history[-100:])
                            episode += 1
                            break

                    if len(batch) < self.batch_size:
                        continue

                    res = self.policy.update(batch)
                    adv_v, value_v, q_vals_v, entropy_loss_v, loss_policy_v, loss_value_v, loss_v, grads = res

                    tb_tracker.track("advantage", adv_v.cpu(), step_idx)
                    tb_tracker.track("values", value_v.cpu(), step_idx)
                    tb_tracker.track("batch_rewards", q_vals_v.cpu(), step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v.cpu(), step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v.cpu(), step_idx)
                    tb_tracker.track("loss_value", loss_value_v.cpu(), step_idx)
                    tb_tracker.track("loss_total", loss_v.cpu(), step_idx)
                    tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                    tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                    tb_tracker.track("grad_var", np.var(grads), step_idx)


                    if len(self.reward_history) > 100:
                        self.reward_history.pop(0)

                    # if avg reward > threshold then save and stop traning:
                    if avg_reward >= self.threshold:
                        print("########## Solved! ###########")
                        name = self.filename + '_solved'
                        self.policy.save(self.directory, name)
                        training_time = time.time() - start_time
                        print("Training time: {:6.2f} sec".format(training_time))
                        break


    def test(self, episodes=3, save_gif=True):

        gifdir = mkdir('.', 'gif')
        algdir = mkdir(gifdir, self.algorithm_name)

        #self.env = gym.wrappers.Monitor(self.env, self.videos_dir)

        # loading models
        self.policy.load(self.directory, self.filename)
        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.policy.agent, gamma=self.gamma,
                                                                    steps_count=self.bellman_steps)

        self.env.reset()
        episode = 0
        batch = []
        t = 0
        finished = False
        epdir = mkdir(algdir, str(episode))

        with RewardTracker(self.writer, stop_reward=self.threshold) as tracker:
            with ptan.common.utils.TBMeanTracker(self.writer, batch_size=10) as tb_tracker:

                for step_idx, exp in enumerate(self.exp_source):

                    batch.append(exp)

                    # handle new rewards
                    new_rewards = self.exp_source.pop_total_rewards()
                    if new_rewards:
                        finished, save_checkpoint = tracker.reward(new_rewards[0], step_idx)
                        if finished:
                            episode += 1
                            break

                    if save_gif:
                        img = self.env.render(mode='rgb_array')
                        img = Image.fromarray(img)
                        img.save('{}/{}.jpg'.format(epdir, t))

                    if finished:
                        ep_reward = new_rewards[0]
                        episode += 1
                        epdir = mkdir(algdir, str(episode))
                        t = 0
                        print('Test episode: {}\tReward: {:4.2f}'.format(episode, ep_reward))

                    t += 1
                    if episode == episodes:
                        break

        self.env.close()
