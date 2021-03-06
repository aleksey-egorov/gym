import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from TD3_PER_CNNLSTM.utils import weights_init, norm_col_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, num_inputs, action_space, batch_size):
        super().__init__()
        print("NUM INPUTS: {}".format(num_inputs))

        self.batch_size = batch_size
        self.cx = Variable(torch.zeros(self.batch_size, 256)).to(device)
        self.hx = Variable(torch.zeros(self.batch_size, 256)).to(device)

        self.cx_eval = Variable(torch.zeros(1, 256)).to(device)
        self.hx_eval = Variable(torch.zeros(1, 256)).to(device)

        self.conv1 = nn.Conv1d(num_inputs, 64, 3, stride=1, padding=1).to(device)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(64, 64, 3, stride=1, padding=1).to(device)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(64, 128, 3, stride=1, padding=1).to(device)
        self.lrelu3 = nn.LeakyReLU(0.1)
        #self.conv4 = nn.Conv1d(128, 128, 1, stride=1).to(device)
        #self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(3072, 256).to(device)
        self.actor_linear = nn.Linear(256, action_space).to(device)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        #self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, state, type='train'):
        #hx, cx = hid
        #state = state.unsqueeze(0)

        # print("STATE: {}".format(state.shape))
        #print("ACT STATE PRE CONV: {}".format(state.shape))
        x = self.lrelu1(self.conv1(state))
        #print("CRT STATE PRE CONV2: {}".format(x.shape))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        #x = self.lrelu4(self.conv4(x))

        #print("ACT STATE PRE LSTM: {}".format(x.shape))
        x = x.view(x.size(0), -1).to(device)
        hx = copy.copy(self.hx)
        cx = copy.copy(self.cx)

        self.hx, self.cx = self.lstm(x, (hx, cx))
        x = self.hx

        return F.softsign(self.actor_linear(x))


class Critic(nn.Module):
    def __init__(self, num_inputs, action_space, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.cxc = Variable(torch.zeros(self.batch_size, 256)).to(device)
        self.hxc = Variable(torch.zeros(self.batch_size, 256)).to(device)

        self.input_dim = num_inputs
        self.conv1 = nn.Conv1d(self.input_dim, 64, 3, stride=1, padding=1).to(device)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(64, 64, 3, stride=1, padding=1).to(device)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(64, 128, 3, stride=1, padding=1).to(device)
        self.lrelu3 = nn.LeakyReLU(0.1)
        #self.conv4 = nn.Conv1d(128, 128, 1, stride=1).to(device)
        #self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(3584, 256).to(device)
        self.critic_linear = nn.Linear(256, 1).to(device)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        #self.conv4.weight.data.mul_(lrelu_gain)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, state, action):

        #print("INPUT DIM: {}".format(self.input_dim))
        action = action.unsqueeze(2)
        action = torch.stack([action, action, action, action], dim=2).squeeze(3)
        #print("CRT STATE PRE CAT: {} {}".format(state.shape, action.shape))
        state_action = torch.cat([state, action], 2)

        #print("CRT STATE PRE CONV: {}".format(state_action.shape))
        x = self.lrelu1(self.conv1(state_action))
        #print("CRT STATE PRE CONV2: {}".format(state_action.shape))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        #x = self.lrelu4(self.conv4(x))

        #print("CRT STATE PRE LSTM: {}".format(x.shape))
        x = x.view(x.size(0), -1)
        hxc = copy.copy(self.hxc)
        cxc = copy.copy(self.cxc)

        self.hxc, self.cxc = self.lstm(x, (hxc, cxc))
        x = self.hxc

        return self.critic_linear(x)
