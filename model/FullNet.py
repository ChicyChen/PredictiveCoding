from PCNetwork import RNNModel, LSTMModel, AutoEncoder
from PreNet import CNNPreNet
from AfterNet import CNNAfterNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, encoded_space_dim=64, num_frame=10):
        super(SimpleNet, self).__init__()

        self.prenet = CNNPreNet()
        self.pcnet = AutoEncoder(encoded_space_dim)
        self.afternet = CNNAfterNet()
        self.encoded_space_dim = encoded_space_dim
        self.num_frame = num_frame

    # x is a batch of video sequences (N, f, 64, 64) for moving MINST
    def forward(self, x):

        x_real = x.permute(1, 0, 2, 3).contiguous()
        h_list = []
        z_list = []
        h_hat_list = []
        x_hat_list = []

        for t in range(self.num_frame):
            h = self.prenet(x_real[t])
            z, h_hat = self.pcnet(h)
            x_hat = self.afternet(h_hat)

        return h, h_hat, z, x_hat
