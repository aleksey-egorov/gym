import torch
import torch.nn as nn

import numpy as np



class Network(nn.Module):
    def __init__(self, fc_config, device):
        super(Network, self).__init__()

        self.model = FullyConnected(fc_config).create().to(device)
        print ("NETWORK={}".format(self.model))
        print ("Device: {}".format(device))


    def forward(self, x):
        #conv_out = self.conv(x).view(x.size()[0], -1)
        return self.model(x)


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



