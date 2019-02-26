import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from DQN.models import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, env, fc_config):
        self.env = env
        self.Q_loss_list = []

        self.net = Network(fc_config)
        self.net_target = Network(fc_config)

        self.target_update_interval = 1000
        self.max_loss_list = 100

    def set_optimizers(self, lr):
        self.lr = lr
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).float()
            q_vals_v = self.net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action

    def update(self, buffer, t, batch_size, gamma):

        if t % self.target_update_interval == 0:
            self.net_target.load_state_dict(self.net.state_dict())

        self.net_optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = self._calc_loss(batch, gamma)
        loss_t.backward()
        self.net_optimizer.step()
        self.Q_loss_list.append(loss_t.item())

    def _calc_loss(self, batch, gamma):
        states, actions, rewards, next_states, dones = batch

        states_v = torch.tensor(states).float()
        next_states_v = torch.tensor(next_states).float()
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).float()
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = self.net(states_v).to(device)
        state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).to(device).squeeze(-1)
        next_state_values = self.net_target(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = (next_state_values * gamma + rewards_v).to(device)
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def save(self, directory, name):
        torch.save(self.net.state_dict(), '%s/%s_net.pth' % (directory, name))
        torch.save(self.net_target.state_dict(), '%s/%s_net_target.pth' % (directory, name))

    def load(self, directory, name):
        print("DIR={} NAME={}".format(directory, name))
        try:
            self.net.load_state_dict(torch.load('%s/%s_net.pth' % (directory, name)))
            self.net_target.load_state_dict(torch.load('%s/%s_net_target.pth' % (directory, name)))

            print("Models loaded")
        except:
            print("No models to load")

    def truncate_loss_lists(self):
        if len(self.Q_loss_list) > self.max_loss_list:
            self.Q_loss_list.pop(0)

