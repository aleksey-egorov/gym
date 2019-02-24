import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, actor_config, max_action):
        super(Actor, self).__init__()

        self.model = FullyConnected(actor_config).create()
        self.max_action = max_action

        print("ACTOR={}".format(self.model))

    def forward(self, state):
        a = self.model(state) * self.max_action
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
            if layer['activation'] == 'tanh':
                layers.append(nn.Tanh())

        return nn.Sequential(*layers)



