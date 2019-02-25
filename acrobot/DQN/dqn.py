import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from DQN.models import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer

        self.net = Network(env.observation_space.shape, env.action_space.n)
        self.tgt_net = Network(env.observation_space.shape, env.action_space.n)

    def set_optimizers(self, lr):
        self.lr = lr
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = self.net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action

    def update(self, buffer, t, batch_size, gamma):

        if t % 1000 == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())

        self.net_optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = self._calc_loss(batch, gamma)
        loss_t.backward()
        self.net_optimizer.step()

    def _calc_loss(self, batch, gamma):
        states, actions, rewards, next_states, dones = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)
