import torch
import torch.nn as nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Network(nn.Module):
    def __init__(self, fc_config):
        super(Network, self).__init__()

        ''' self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)      
        
        conv_out_size = input_shape
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        ) '''

        self.model = FullyConnected(fc_config).create()
        print ("NETWORK={}".format(self.model))

    #def _get_conv_out(self, shape):
    #    o = self.conv(torch.zeros(1, *shape))
    #    return int(np.prod(o.size()))

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



