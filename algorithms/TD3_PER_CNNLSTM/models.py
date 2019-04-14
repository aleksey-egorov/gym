import torch
import torch.nn as nn
from torch.autograd import Variable

from TD3_PER_CNNLSTM.utils import weights_init, norm_col_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        print("NUM INPUTS: {}".format(num_inputs))
        self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(1600, 128)
        self.actor_linear = nn.Linear(128, action_space)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, state):
        #state = state.unsqueeze(0)
        print("STATE PRE: {}".format(state.shape))

        x = self.lrelu1(self.conv1(state))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))

        print("STATE PRE LSTM: {}".format(x.shape))
        x = x.view(x.size(0), -1).to(device)
        hx, cx = self.lstm(x, (self.hx, self.cx))
        x = hx

        return self.actor_linear(x)


class Critic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        self.conv1 = nn.Conv1d(num_inputs + action_space, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(1600, 128)
        self.critic_linear = nn.Linear(128, 1)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.cx = Variable(torch.zeros(1, 128))
        self.hx = Variable(torch.zeros(1, 128))

        self.train()

    def forward(self, state, action):
        state = state.unsqueeze(0)
        state_action = torch.cat([state, action], 1)

        x = self.lrelu1(self.conv1(state_action))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (self.hx, self.cx))
        x = hx

        return self.critic_linear(x)
