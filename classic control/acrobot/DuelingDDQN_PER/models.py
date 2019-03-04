import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    def __init__(self, config, device):
        super(DuelingNetwork, self).__init__()

        fc_config, adv_stream_config, value_stream_config = config

        self.fc = FullyConnected(fc_config).create().to(device)
        self.adv_stream = FullyConnected(adv_stream_config).create().to(device)
        self.value_stream = FullyConnected(value_stream_config).create().to(device)

        print ("NETWORK: {} {} {}".format(self.fc, self.adv_stream, self.value_stream))
        print ("Device: {}".format(device))

    def forward(self, x):
        batch_size = x.size(0)

        fc_out = self.fc(x)
        adv_out = self.adv_stream(fc_out)
        value_out = self.value_stream(fc_out)
        #print("FC: {}   ADV: {}   VAL: {}".format(fc_out.shape, adv_out.shape, value_out.shape))

        val = value_out.expand(batch_size, adv_out.size(1))
        output = value_out + adv_out - torch.mean(adv_out, dim=1, keepdim=True)

        output = val + adv_out - adv_out.mean(dim=1, keepdim=True)

        #print("OUT: {}".format(output.shape))
        #out = self.aggr(output)
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



