from PCNetwork import RNNModel, LSTMModel, AutoEncoder
from PreNet import CNNPreNet
from AfterNet import CNNAfterNet

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, encoded_space_dim):
        super(SimpleNet, self).__init__()

        self.prenet = CNNPreNet()
        self.pcnet = AutoEncoder(encoded_space_dim)
        self.afternet = CNNAfterNet()

    # x is a batch of video sequences (N, f, 64, 64) for moving MINST
    def forward(self, x):

        h = self.prenet(x)
        z, h_hat = self.pcnet(h)
        x_hat = self.afternet(h_hat)

        return h, h_hat, z, x_hat
