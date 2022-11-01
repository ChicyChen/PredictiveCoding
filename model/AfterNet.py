import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAfterNet(nn.Module):
    def __init__(self):
        super(CNNAfterNet, self).__init__()

        # DeConvolution 1
        self.decnn1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),)

        # DeConvolution 2
        self.decnn2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),)

    def forward(self, x):
       # DeConvolution 1
        out = self.decnn1(x)
        out = self.decnn2(out)

        return out
