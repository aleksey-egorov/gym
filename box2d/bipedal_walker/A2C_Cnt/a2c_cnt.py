import torch
import ptan
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from A2C_Cnt.models import A2C_Model
from A2C_Cnt.utils import unpack_batch_continuous

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2C_Agent(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class A2C_Cnt():

    def __init__(self, state_dim, action_dim, entropy_beta, gamma, reward_steps):
        super(A2C_Cnt, self).__init__()

        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.reward_steps = reward_steps
        self.model = A2C_Model(state_dim, action_dim).to(device)
        self.agent = A2C_Agent(self.model, device)


    def set_optimizers(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def test_net(self, env, count=10, device=device):
        net = self.model
        rewards = 0.0
        steps = 0
        for _ in range(count):
            obs = env.reset()
            while True:
                obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
                mu_v = net(obs_v)[0]
                action = mu_v.squeeze(dim=0).data.cpu().numpy()
                action = np.clip(action, -1, 1)
                obs, reward, done, _ = env.step(action)
                rewards += reward
                steps += 1
                if done:
                    break
        return rewards / count, steps / count

    def calc_logprob(self, mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2

    def update(self, batch):

        states_v, actions_v, vals_ref_v = \
            unpack_batch_continuous(batch, self.model, last_val_gamma=self.gamma ** self.reward_steps, device=device)
        batch.clear()

        self.optimizer.zero_grad()
        mu_v, var_v, value_v = self.model(states_v)

        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * self.calc_logprob(mu_v, var_v, actions_v)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = self.entropy_beta * (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()

        loss_v = loss_policy_v + entropy_loss_v + loss_value_v
        loss_v.backward()
        self.optimizer.step()

        return adv_v, value_v, vals_ref_v, entropy_loss_v, loss_policy_v, loss_value_v, loss_v

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




        
        
      
        