import torch.nn as nn



class Network(nn.Module):
    def __init__(self, config, device):
        super(Network, self).__init__()

        self.net = FullyConnected(config).create().to(device)

        print ("NETWORK: {} Device: {}".format(self.net, device))

        #self.net = nn.Sequential(
        #    nn.Linear(input_size, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, n_actions)
        #)

    def forward(self, x):
        return self.net(x)



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



