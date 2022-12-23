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


class ConvDeconv2xNet(nn.Module):

    def __init__(self, in_channels: int):
        super(ConvDeconv2xNet, self).__init__()
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

class ConvDeconv3xNet(nn.Module):

    num_layers = 3
    def __init__(self, in_channels: int):
        super(ConvDeconv3xNet, self).__init__()

        conv_shp, deconv_shp = self.get_shapes(self.num_layers, in_channels)
        self.activation = nn.ReLU()
        # conv layers
        self.conv0 = nn.Conv1d(in_channels=conv_shp[0][0], out_channels=conv_shp[0][1], kernel_size=3, stride=1)
        self.conv1 = nn.Conv1d(in_channels=conv_shp[1][0], out_channels=conv_shp[1][1], kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=conv_shp[2][0], out_channels=conv_shp[2][1], kernel_size=3, stride=1)
        # deconv layers
        self.deconv2 = nn.ConvTranspose1d(in_channels=deconv_shp[0][0], out_channels=deconv_shp[0][1], kernel_size=3, stride=1)
        self.deconv1 = nn.ConvTranspose1d(in_channels=deconv_shp[1][0], out_channels=deconv_shp[1][1], kernel_size=3, stride=1)
        self.deconv0 = nn.ConvTranspose1d(in_channels=deconv_shp[2][0], out_channels=deconv_shp[2][1], kernel_size=3, stride=1)

        #self.linear = nn.Linear(deconv_shp[0][1] + deconv_shp[1][1] + deconv_shp[2][1], 2)
        self.linear = nn.Linear(1360, 2)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_shapes(n_layers: int, in_channels: int):
        d = in_channels
        conv = []
        for i in range(n_layers):
            conv.append(
                (d, d*2)
            )
            d *= 2
        deconv = [(x[1], x[0]) for x in list(reversed(conv))]
        return conv, deconv
    def forward(self, x):
        # CONV BLOCKS
        h0 = self.activation(self.conv0(x))
        h1 = self.activation(self.conv1(h0))
        h2 = self.activation(self.conv2(h1))

        # DECONV BLOCKS
        h2 = self.activation(self.deconv2(h2))
        h1 = self.activation(self.deconv1(h1))
        h0 = self.activation(self.deconv0(h0))



        # GLOBAL FEATURE CONCATENATION
        h = torch.cat([h0.flatten(1), h1.flatten(1), h2.flatten(1)], dim=-1)
        # print(h.shape)

        # print(h.shape)
        y_hat = self.linear(h)

        if not self.training:
            y_hat = self.softmax(y_hat)

        return y_hat


if __name__ == "__main__":
    x = torch.randn(16, 2, 100)
    model = ConvDeconv3xNet(in_channels=2)
    y = model(x)