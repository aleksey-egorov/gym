import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable


from Dueling_DDQN_PER_CNNLSTM.utils import weights_init, norm_col_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DuelingNetwork(nn.Module):
    def __init__(self, num_inputs, action_space, batch_size):
        super(DuelingNetwork, self).__init__()

        print("NUM INPUTS: {}".format(num_inputs))
        print("ACTION SPACE {}".format(action_space))

        self.batch_size = batch_size
        self.cx = Variable(torch.zeros(self.batch_size, 256)).to(device)
        self.hx = Variable(torch.zeros(self.batch_size, 256)).to(device)

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, padding=1).to(device)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1).to(device)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1).to(device)
        self.lrelu3 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(6400, 256).to(device)
        self.adv_stream_1 = nn.Linear(256, 512).to(device)
        self.adv_stream_2 = nn.Linear(512, action_space).to(device)
        self.val_stream_1 = nn.Linear(256, 512).to(device)
        self.val_stream_2 = nn.Linear(512, 1).to(device)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)

        self.adv_stream_1.weight.data = norm_col_init(self.adv_stream_1.weight.data, 0.01).to(device)
        self.adv_stream_2.weight.data = norm_col_init(self.adv_stream_2.weight.data, 0.01).to(device)
        self.val_stream_1.weight.data = norm_col_init(self.val_stream_1.weight.data, 0.01).to(device)
        self.val_stream_2.weight.data = norm_col_init(self.val_stream_2.weight.data, 0.01).to(device)

        self.adv_stream_1.bias.data.fill_(0).to(device)
        self.adv_stream_2.bias.data.fill_(0).to(device)
        self.val_stream_1.bias.data.fill_(0).to(device)
        self.val_stream_2.bias.data.fill_(0).to(device)

        self.lstm.bias_ih.data.fill_(0).to(device)
        self.lstm.bias_hh.data.fill_(0).to(device)


    def forward(self, state, type='train'):
        #print("RAW STATE: {}".format(state.shape))

        state = state.reshape(self.batch_size, 4, 84, 84)
        #print("STATE: {}".format(state.shape))
        conv1 = self.conv1(state)
        #print("CONV1: {}".format(conv1.shape))
        x = self.lrelu1(conv1)
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))

        #print("ACT STATE PRE LSTM: {}".format(x.shape))
        conv_out = x.view(x.size(0), -1).to(device)

        hx = copy.copy(self.hx)
        cx = copy.copy(self.cx)
        self.hx, self.cx = self.lstm(conv_out, (hx, cx))
        lstm_out = self.hx
        lstm_out = lstm_out.to(device)

        #print("ACT STATE PAST LSTM: {}".format(lstm_out.shape))
        adv_out_1 = self.adv_stream_1(lstm_out)
        adv_out_2 = self.adv_stream_2(adv_out_1)
        val_out_1 = self.val_stream_1(lstm_out)
        val_out_2 = self.val_stream_2(val_out_1)

        val = val_out_2.expand(self.batch_size, adv_out_2.size(1))
        output = val + adv_out_2 - adv_out_2.mean(dim=1, keepdim=True)

        return output
