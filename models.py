## Import

import sys
import time
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from lshashpy3 import LSHash
from torch.utils.data import TensorDataset, DataLoader

from six.moves import urllib
opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Run this code cell to train MNIST neural network. Do not modify!
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MNISTNet(nn.Module):
    def __init__(self, hidden=128):
      super(MNISTNet, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.hidden = hidden
      self.fc1 = nn.Linear(28*28*1, self.hidden)
      self.fc2 = nn.Linear(self.hidden, 10)

    def forward(self, x):
      x = x.view(-1, 28*28)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)


class HashedMNISTNetDecimal(nn.Module):
    def __init__(self, num_hash_tables, hidden=128):
      super(HashedMNISTNetDecimal, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.num_hash_tables = num_hash_tables
      self.hidden = hidden
      self.fc1 = nn.Linear(num_hash_tables*1, self.hidden)
      self.fc2 = nn.Linear(self.hidden, 10)

    def forward(self, x):
      x = x.view(-1, self.num_hash_tables)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)


class HashedMNISTNetBinary(nn.Module):
    def __init__(self, num_hash_tables, bitsize, hidden=128):
      super(HashedMNISTNetBinary, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.hidden = hidden
      self.num_hash_tables = num_hash_tables
      self.bitsize = bitsize
      self.fc1 = nn.Linear(num_hash_tables*bitsize, self.hidden)
      self.fc2 = nn.Linear(self.hidden, 10)

    def forward(self, x):
      x = x.view(-1, self.num_hash_tables*self.bitsize)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)
