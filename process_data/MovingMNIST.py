# moving minst contains 10000 videos, each have 20 frames, with image size 64 by 64
# dataset: http://www.cs.toronto.edu/~nitish/unsupervised_video/
# code reference: https://github.com/tychovdo/MovingMNIST

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import os
import os.path
import errno

# datapath = "../dataset/moving_minst/mnist_test_seq.npy"
# a = np.load(datapath).swapaxes(0, 1)

# print(a.shape) # (10000, 20, 64, 64)
# print(a[0,0,:,:].shape) # (64, 64)
# print(np.max(a), np.min(a), a.dtype) # 255 0 uint8

# img = Image.fromarray(a[0,0,:,:])
# img.show()

class MovingMNIST(data.Dataset):

    """
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
            preprocess (bool, optional): If true, preprocess to split and store train
            and testdata.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root = '../dataset/moving_minst', train=True, split=1000, transform=None, target_transform=None, preprocess=False):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train 

        if preprocess:
            self.preprocess()

        print('Loading...')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

        print('Done!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part, each part contains 10 frames
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def preprocess(self):

        # process and save as torch files
        print('Processing...')

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        raw_data_path = os.path.join(self.root, 'mnist_test_seq.npy')
        raw_data = np.load(raw_data_path).swapaxes(0, 1)

        training_set = torch.from_numpy(raw_data[:-self.split])
        test_set = torch.from_numpy(raw_data[-self.split:])

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

