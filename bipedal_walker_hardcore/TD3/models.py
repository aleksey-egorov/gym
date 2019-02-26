import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, actor_config, action_low, action_high):
        super(Actor, self).__init__()

        self.model = FullyConnected(actor_config).create()
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.action_range = self.action_high - self.action_low

        print("ACTOR={}".format(self.model))

    def forward(self, state):
        a = self.model(state)
        return a


class Critic(nn.Module):
    def __init__(self, critic_config):
        super(Critic, self).__init__()

        self.model = FullyConnected(critic_config).create()

        print("CRITIC={}".format(self.model))

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = self.model(state_action)
        return q


class FullyConnected(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dropout_default = 0.2

    def create(self):
        layers = []
        for layer in self.config:
            layers.append(nn.Linear(layer['dim'][0], layer['dim'][1]))
            if layer['dropout'] == True:
                layers.append(nn.Dropout(self.dropout_default))

            if layer['activation'] == 'relu':
                layers.append(nn.ReLU())
            if layer['activation'] == 'elu':
                layers.append(nn.ELU())
            if layer['activation'] == 'tanh':
                layers.append(nn.Tanh())
            if layer['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

