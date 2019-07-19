import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable

from DQN_CNNLSTM.utils import weights_init, norm_col_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, num_inputs, action_space, batch_size):
        super(Network, self).__init__()

        print("NUM INPUTS: {}".format(num_inputs))
        print("ACTION SPACE {}".format(action_space))

        self.batch_size = batch_size
        self.cx = Variable(torch.zeros(self.batch_size, 128)).to(device)
        self.hx = Variable(torch.zeros(self.batch_size, 128)).to(device)

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, padding=1).to(device)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1).to(device)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1).to(device)
        self.lrelu3 = nn.LeakyReLU(0.1)
        #self.conv4 = nn.Conv2d(64, 64, 3, stride=1).to(device)
        #self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(6400, 128).to(device)
        self.actor_linear = nn.Linear(128, action_space).to(device)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        #self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01).to(device)
        self.actor_linear.bias.data.fill_(0).to(device)

        self.lstm.bias_ih.data.fill_(0).to(device)
        self.lstm.bias_hh.data.fill_(0).to(device)


    def forward(self, state, type='train'):

        state = state.reshape(self.batch_size, 4, 84, 84)
        #print("STATE: {}".format(state.shape))
        conv1 = self.conv1(state)
        #print("CONV1: {}".format(conv1.shape))
        x = self.lrelu1(conv1)

        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        #x = self.lrelu4(self.conv4(x))

        #print("ACT STATE PRE LSTM: {}".format(x.shape))
        x = x.view(x.size(0), -1).to(device)
        hx = copy.copy(self.hx)
        cx = copy.copy(self.cx)

        self.hx, self.cx = self.lstm(x, (hx, cx))
        x = self.hx
        x = x.to(device)
        x = self.actor_linear(x)
        return x
