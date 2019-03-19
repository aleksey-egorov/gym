import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, config, num_outputs, device, std=0.0):
        super(ActorCritic, self).__init__()

        #self.critic = nn.Sequential(
        #    nn.Linear(num_inputs, hidden_size),
        # #   nn.ReLU(),
        #    nn.Linear(hidden_size, 1)
        #)

        #self.actor = nn.Sequential(
        #    nn.Linear(num_inputs, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, num_outputs),
        #)

        actor_config, critic_config = config

        self.actor = FullyConnected(actor_config).create().to(device)
        self.critic = FullyConnected(critic_config).create().to(device)

        print ("Actor: {} Critic: {} Device: {}".format(self.actor, self.critic, device))


        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std).to(device)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class FullyConnected():
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

