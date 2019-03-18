import torch
import torch.nn as nn


class A2C_Model(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size):
        super(A2C_Model, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_size, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(hidden_size, 1)

        print("MODELS: {} {} {} {}".format(self.base, self.mu, self.var, self.value))

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


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

