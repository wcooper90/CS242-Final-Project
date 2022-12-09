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
import json

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


from helpers import *
from models import MNISTNet, HashedMNISTNetDecimal, HashedMNISTNetBinary


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset1_mnist = datasets.MNIST('./data_mnist', train=True, download=True,
                    transform=transform)
dataset2_mnist = datasets.MNIST('./data_mnist', train=False,
                    transform=transform)


bitsizes = [1, 2, 4, 8, 16, 32]
num_hash_tables = [1, 5, 10, 15, 20, 30, 50, 100]

first_network_results = {}
for b in bitsizes:
    for n in num_hash_tables:
        print("*"*80)
        print("Bitsize: " + str(b) + ", num_hash_tables: " + str(n))
        model = HashedMNISTNetBinary(n, b)
        train_loader_hashed_mnist, test_loader_hashed_mnist, times = preprocess_binary(b, n, dataset1_mnist, dataset2_mnist)
        training_time, accuracy, num_params = run_hashed_model(train_loader_hashed_mnist, test_loader_hashed_mnist, model)
        if b not in first_network_results:
            first_network_results[b] = {}
        first_network_results[b][n] = {'preprocessing_time': sum(times), 'training_time': training_time,
                                        'accuracy': accuracy, 'num_params': num_params}
        print(first_network_results[b][n])
        print("*"*80)

        with open('results.txt', 'a') as f:
            f.write(json.dumps(first_network_results[b][n]))
            f.write('\n')


print(first_network_results)



# run_default(dataset1_mnist, dataset2_mnist)
#
# train_loader_hashed_mnist, test_loader_hashed_mnist, times = preprocess_decimal(8, 20, dataset1_mnist, dataset2_mnist)
# print('Total preprocessing time: '  + str(sum(times)) + ' seconds.')
# model = HashedMNISTNetDecimal(20)
# run_hashed_model(train_loader_hashed_mnist, test_loader_hashed_mnist, model)
#
# train_loader_hashed_mnist, test_loader_hashed_mnist, times = preprocess_binary(8, 20, dataset1_mnist, dataset2_mnist)
# print('Total preprocessing time: '  + str(sum(times)) + ' seconds.')
# model = HashedMNISTNetBinary(20, 8)
# training_time, accuracy, num_params = run_hashed_model(train_loader_hashed_mnist, test_loader_hashed_mnist, model)
# print('Total training time: '  + str(training_time) + ' seconds.')
# print('Number of model parameters: '  + str(num_params) + ' parameters.')
# print('Final accuracy: '  + str(accuracy) + '%')
