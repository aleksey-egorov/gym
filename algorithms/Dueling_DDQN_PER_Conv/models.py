import torch
import torch.nn as nn
import numpy as np


class DuelingNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, config, device):
        super(DuelingNetwork, self).__init__()

        conv_config, adv_config, val_config = config

        conv_config[0]['dim'][0] = input_shape[0]
        adv_config[-1]['dim'][1] = n_actions

        self.conv = ConvNet(conv_config).create().to(device)
        conv_out_size = self._get_conv_out(input_shape, device)
        adv_config[0]['dim'][0] = conv_out_size
        val_config[0]['dim'][0] = conv_out_size

        self.adv_stream = FullyConnected(adv_config).create().to(device)
        self.val_stream = FullyConnected(val_config).create().to(device)

        print("NETWORK: {} {} {} device: {}".format(self.conv, self.adv_stream, self.val_stream, device))

    def _get_conv_out(self, shape, device):
        o = self.conv(torch.zeros(1, *shape).to(device))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)

        conv_out = self.conv(x).view(x.size()[0], -1)
        adv_out = self.adv_stream(conv_out)
        value_out = self.val_stream(conv_out)

        val = value_out.expand(batch_size, adv_out.size(1))
        output = val + adv_out - adv_out.mean(dim=1, keepdim=True)

        return output


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


