import torch
import ptan
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from PPO.models import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO():

    def __init__(self, config, action_dim,  entropy_beta, gamma, gae_lambda, batch_size, ppo_epsilon,
                 ppo_epochs, critic_discount):
        super(PPO, self).__init__()

        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.critic_discount = critic_discount
        self.model = ActorCritic(config, action_dim, device)

    def set_optimizers(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def test_env(self, env, device, deterministic=True):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = self.model(state)
            action = dist.mean.detach().cpu().numpy()[0] if deterministic \
                else dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        return total_reward

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def compute_gae(self, next_value, rewards, masks, values):
        lam = self.gae_lambda
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * lam * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // self.batch_size):
            rand_ids = np.random.randint(0, batch_size, self.batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                           rand_ids, :]

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        clip_param = self.ppo_epsilon
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(self.ppo_epochs):

            # grabs random mini-batches several times until we have covered all data
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns,
                                                                             advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # track statistics
                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps += 1

        sum_returns_m  = sum_returns / count_steps
        sum_advantage_m = sum_advantage / count_steps
        sum_loss_actor_m = sum_loss_actor / count_steps
        sum_loss_critic_m = sum_loss_critic / count_steps
        sum_entropy_m = sum_entropy / count_steps
        sum_loss_total_m = sum_loss_total / count_steps

        return sum_returns_m, sum_advantage_m, sum_loss_actor_m, sum_loss_critic_m, sum_entropy_m, sum_loss_total_m


    def save(self, directory, name):
        torch.save(self.model.state_dict(), '%s/%s_actor.pth' % (directory, name))

    def load(self, directory, name):
        print ("DIR={} NAME={}".format(directory, name))
        try:
            self.model.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name)))
            print("Models loaded")
        except:
            print("No models to load")

    def truncate_loss_lists(self):
        if len(self.loss_list) > self.max_loss_list:
            self.loss_list.pop(0)




        
        
      
        