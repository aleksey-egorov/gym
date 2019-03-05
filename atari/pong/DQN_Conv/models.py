import torch
import torch.nn as nn



class Network(nn.Module):
    def __init__(self, config, device):
        super(Network, self).__init__()

        conv_config, fc_config = config

        self.conv = ConvNet(conv_config).create().to(device)
        self.fc = FullyConnected(fc_config).create().to(device)
        print ("NETWORK: {} {} device: {}".format(self.conv, self.fc, device))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


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

class ConvNet():

    def __init__(self, config):
        self.config = config

    def create(self):
        layers = []
        for layer in self.config:
            layers.append(nn.Conv2d(layer['dim'][0], layer['dim'][1], kernel_size=layer['kernel'], stride=layer['stride']))

            if layer['activation'] == 'relu':
                layers.append(nn.ReLU())

            if layer['batch_norm'] == True:
                layers.append(nn.BatchNorm2d(layer['dim'][1]))

        return nn.Sequential(*layers)

