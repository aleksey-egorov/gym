import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(ActorCritic, self).__init__()


        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        ).to(device)

        conv_out_size = self._get_conv_out(num_inputs, device)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

        #self.log_std = nn.Parameter(torch.ones(1, num_outputs))

        print ("NETWORK: {} {} {} device: {}".format(self.conv, self.actor, self.critic, device))

    def _get_conv_out(self, shape, device):
        o = self.conv(torch.zeros(1, *shape).to(device))
        return int(np.prod(o.size()))

    #def forward(self, x):
    #    conv_out = self.conv(x).view(x.size()[0], -1)
    #    return self.fc(conv_out)

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.critic(conv_out)
        mu = self.actor(conv_out)
        mu_prob_v = F.log_softmax(mu, dim=1)

        #print ("MU: {}".format(mu_prob_v.shape))
        #print ("MU: {}".format(mu_prob_v.shape))

        #std = self.log_std.exp().expand_as(mu)
        #print ("MU: {}".format(mu.shape))
        dist = Categorical(mu_prob_v)
        return dist, value


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

