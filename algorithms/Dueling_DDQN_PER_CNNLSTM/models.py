import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from torch.autograd import Variable


from Dueling_DDQN_PER_CNNLSTM.utils import weights_init, norm_col_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DuelingNetwork(nn.Module):
    def __init__(self, num_inputs, action_space, batch_size):
        super(DuelingNetwork, self).__init__()

        print("NUM INPUTS: {}".format(num_inputs))
        print("ACTION SPACE {}".format(action_space))

        self.batch_size = batch_size
        self.cx = Variable(torch.zeros(self.batch_size, 512)).to(device)
        self.hx = Variable(torch.zeros(self.batch_size, 512)).to(device)


        #self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4, padding=1).to(device)
        #self.lrelu1 = nn.LeakyReLU(0.1)
        #self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1).to(device)
        #self.lrelu2 = nn.LeakyReLU(0.1)
        #self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1).to(device)
        #self.lrelu3 = nn.LeakyReLU(0.1)

        #self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, batch_first=True).to(device)
        #self.lstm = nn.LSTMCell(3200, 128).to(device)

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2).to(device)
        self.maxp1 = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1).to(device)
        self.maxp2 = nn.MaxPool2d(2, 2).to(device)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1).to(device)
        self.maxp3 = nn.MaxPool2d(2, 2).to(device)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1).to(device)
        self.maxp4 = nn.MaxPool2d(2, 2).to(device)

        self.lstm = nn.LSTMCell(1024, 512).to(device)
        self.adv_stream = nn.Linear(512, action_space).to(device)
        self.val_stream = nn.Linear(512, 1).to(device)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.adv_stream.weight.data = norm_col_init(self.adv_stream.weight.data, 0.01).to(device)
        self.val_stream.weight.data = norm_col_init(self.val_stream.weight.data, 0.01).to(device)
        self.adv_stream.bias.data.fill_(0).to(device)
        self.val_stream.bias.data.fill_(0).to(device)

        self.lstm.bias_ih.data.fill_(0).to(device)
        self.lstm.bias_hh.data.fill_(0).to(device)
        #self.lstm.weight_ih.data = norm_col_init(self.lstm.weight_ih.data, 1.0).to(device)
        #self.lstm.weight_hh.data = norm_col_init(self.lstm.weight_ih.data, 1.0).to(device)

        #print ("IH: {} HH: {}".format(self.lstm.weight_ih.data.shape, self.lstm.weight_hh.data.shape))
        #self.cx = self.lstm.weight_ih.data
        #self.hx = self.lstm.weight_hh.data


    def forward(self, state, type='train'):
        #print("RAW STATE: {}".format(state.shape))

        state = state.reshape(self.batch_size, 4, 84, 84)

        x = F.relu(self.maxp1(self.conv1(state)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        conv_out = x.view(x.size(0), -1)

        hx = copy.copy(self.hx)
        cx = copy.copy(self.cx)

        #print("HX: {} CX: {}".format(hx.shape, cx[0].shape))

        #print ("CONV_OUT: {}".format(conv_out.shape))
        self.cx, self.hx = self.lstm(conv_out, (cx, hx))
        lstm_out = self.cx

        #print("ACT STATE PAST LSTM: {}".format(lstm_out.shape))
        adv_out = self.adv_stream(lstm_out)
        val_out = self.val_stream(lstm_out)

        #print("VAL: {}".format(adv_out_2.shape))
        val = val_out #.expand(self.batch_size, adv_out_2.size(1))
        output = val + adv_out - adv_out.mean(dim=1, keepdim=True)

        #print("OUTPUT: {}".format(output.shape))

        return output
