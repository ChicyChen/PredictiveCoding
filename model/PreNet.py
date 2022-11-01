import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# A CNN model as PreNet for moving MINST due to low resolution of images
class CNNPreNet(nn.Module):
    def __init__(self):
        super(CNNPreNet, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        # (Moving MINST 32*32)

        # Convolution 2
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        # (Moving MINST 16*16)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        # Convolution 2
        out = self.cnn2(out)

        # Resize
        # Original size: (100, 32, 16, 16)
        # out.size(0): 100
        # New out size: (100, 32*16*16)
        # out = out.view(out.size(0), -1)

        return out
