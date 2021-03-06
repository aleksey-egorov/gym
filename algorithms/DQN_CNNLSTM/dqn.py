import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from DQN_CNNLSTM.models import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN_CNNLSTM:
    def __init__(self, env, input_shape, n_actions, batch_size):
        self.env = env
        self.batch_size = batch_size
        self.Q_loss_list = []

        self.net = Network(input_shape, n_actions, batch_size)
        self.net_target = Network(input_shape, n_actions, batch_size)

        self.target_update_interval = 1000
        self.max_loss_list = 100

    def set_optimizers(self, lr):
        self.lr = lr
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.Tensor(state)
            statem = state
            for i in range(self.batch_size - 1):
                statem = torch.cat((statem, state), 0)

            state_v = statem.float().to(device)
            q_vals_v = self.net(state_v)
            act_v = torch.argmax(q_vals_v[0])
            action = act_v.item()
        return action

    def update(self, buffer, t, batch_size, gamma):
        if buffer.len() > batch_size:
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

        states_v = torch.tensor(states).float().to(device)
        next_states_v = torch.tensor(next_states).float().to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).float().to(device)
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

