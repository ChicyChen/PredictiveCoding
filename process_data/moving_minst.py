from MovingMNIST import MovingMNIST
import torch

train_set = MovingMNIST(root='../dataset/moving_minst',
                        train=True, preprocess=True)
test_set = MovingMNIST(root='../dataset/moving_minst',
                       train=False, preprocess=False)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

for seq, seq_target in train_loader:
    print('--- Sample')
    print('Input:  ', seq.shape)  # torch.Size([100, 10, 64, 64])
    print('Target: ', seq_target.shape)  # torch.Size([100, 10, 64, 64])
    break
