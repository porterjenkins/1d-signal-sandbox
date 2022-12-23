import numpy as np
import torch
import torch.nn as nn


class AvgPoolConvNet(nn.Module):

    def __init__(self, in_channels: int):
        super(AvgPoolConvNet, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels = 4, kernel_size=8, stride=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels = 8, kernel_size=4, stride=2)
        self.pool = nn.AvgPool1d(kernel_size=22, stride=1)
        self.output = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        h = self.activation(self.conv1(x))
        print(h.shape)
        h = self.activation(self.conv2(h))
        print(h.shape)
        h = self.pool(h).squeeze(-1)
        print(h.shape)
        y_hat = self.output(h)

        if not self.training:
            y_hat = self.softmax(y_hat)

        return y_hat


class ConvDeconvNet(nn.Module):

    def __init__(self, in_channels: int):
        super(ConvDeconvNet, self).__init__()
        self.activation = nn.ReLU()
        # conv layers
        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=4, kernel_size=8, stride=2)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, stride=2)
        # deconv layers
        self.deconv0 = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=4, stride=2)
        self.deconv1 = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=4, stride=2)
        self.linear = nn.Linear(376, 2)
        # self.linear1 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # CONV BLOCKS
        h0 = self.activation(self.conv0(x))
        # print(h0.shape)
        h1 = self.activation(self.conv1(h0))
        # print(h1.shape)

        # DECONV BLOCKS
        h1 = self.activation(self.deconv1(h1))
        # print(h1.shape)
        h0 = self.activation(self.deconv0(h0))
        # print(h0.shape)

        # GLOBAL FEATURE CONCATENATION
        h = torch.cat([h0.flatten(1), h1.flatten(1)], dim=-1)
        # print(h.shape)

        # print(h.shape)
        y_hat = self.linear(h)

        if not self.training:
            y_hat = self.softmax(y_hat)

        return y_hat
