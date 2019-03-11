import os
import gym
import time
import ptan
import torch
import numpy as np
from tensorboardX import SummaryWriter

from A2C_Cnt.a2c_cnt import A2C_Cnt
from A2C_Cnt.utils import mkdir



class A2C_Cnt_Trainer():

    def __init__(self, env_name, hidden_size=256, random_seed=42, lr_base=0.001, lr_decay=0.00005,
                 batch_size=32, max_episodes=10000, max_timesteps=10000, entropy_beta=1e-4, gamma=0.99, reward_steps=2,
                 log_interval=5, threshold=None, test_iters=10000, lr_minimum=1e-10,
                 log_dir='./log/'):

        self.algorithm_name = 'a2c_cnt'
        self.env_name = env_name
        self.env = gym.make(env_name)
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

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps

        self.entropy_beta = entropy_beta
        self.test_iters = test_iters
        self.gamma = gamma
        self.reward_steps = reward_steps

        self.batch_size = batch_size
        self.log_interval = log_interval

        prdir = mkdir('.', 'preTrained')
        self.directory = mkdir(prdir, self.algorithm_name)
        self.filename = "{}_{}_{}".format(self.algorithm_name, self.env_name, self.random_seed)

        self.policy = A2C_Cnt(self.state_dim, self.action_dim, self.hidden_size, self.entropy_beta, self.gamma, self.reward_steps)
        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.policy.agent, self.gamma, steps_count=self.reward_steps)

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
        self.policy.set_optimizers(lr=self.lr_base)

        print("Training started ... \n")

        batch = []
        best_reward = None
        with ptan.common.utils.RewardTracker(self.writer) as tracker:
            with ptan.common.utils.TBMeanTracker(self.writer, batch_size=10) as tb_tracker:

                for step_idx, exp in enumerate(self.exp_source):
                    rewards_steps = self.exp_source.pop_rewards_steps()
                    if rewards_steps:
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track("episode_steps", steps[0], step_idx)
                        tracker.reward(rewards[0], step_idx)

                    if step_idx % self.test_iters == 0:
                        ts = time.time()
                        rewards, steps = self.policy.test_net(self.test_env)
                        print("Test done is %.2f sec, reward %.3f, steps %d" % (
                            time.time() - ts, rewards, steps))
                        self.writer.add_scalar("test_reward", rewards, step_idx)
                        self.writer.add_scalar("test_steps", steps, step_idx)

                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                                self.policy.save(self.directory, name)
                            best_reward = rewards

                    batch.append(exp)
                    if len(batch) < self.batch_size:
                        continue

                    res = self.policy.update(batch)
                    adv_v, value_v, vals_ref_v, entropy_loss_v, loss_policy_v, loss_value_v, loss_v = res

                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)


                    '''
                    # Print avg reward every log interval:
                    if episode % self.log_interval == 0:
                        self.policy.save(self.directory, self.filename)
                        print(
                            "Ep:{:4d}   Rew:{:5.2f}  Avg Rew:{:5.2f}  LR:{:8.8f}  Bf:{:2.0f} Beta:{:0.4f}  EN:{:0.4f}  Loss: {:5.3f} {:5.3f} {:5.3f}".format(
                                episode, ep_reward, avg_reward, learning_rate, self.replay_buffer.get_fill(), beta,
                                exploration_noise, avg_actor_loss, avg_Q1_loss, avg_Q2_loss))

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
                        break'''

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
