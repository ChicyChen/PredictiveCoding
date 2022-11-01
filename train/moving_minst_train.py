from model.FullNet import SimpleNet
from process_data.MovingMNIST import MovingMNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path

# global variables declaration
encoded_space_dim = 64
batch_size = 100

# load data
print('==>>> Loading training data')

train_set = MovingMNIST(root='../dataset/moving_minst',
                        train=True, preprocess=False)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

print('==>>> training data loaded, total trainning batch number: {}'.format(
    len(train_loader)))

# define model
model = SimpleNet(encoded_space_dim)

# define loss, opt, etc
criterion = nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for seq, seq_target in train_loader:
    print('--- Sample')
    print('Input:  ', seq.shape)  # torch.Size([100, 10, 64, 64])
    print('Target: ', seq_target.shape)  # torch.Size([100, 10, 64, 64])
