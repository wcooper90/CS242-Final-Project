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
from tqdm import tqdm


def train_mnist(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))




def test_mnist(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def run_default(dataset1_mnist, dataset2_mnist):

    model = MNISTNet()
    device = torch.device("cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader_mnist = torch.utils.data.DataLoader(dataset1_mnist)
    test_loader_mnist = torch.utils.data.DataLoader(dataset2_mnist)

    optimizer = optim.Adam(model.parameters())


    start_time = time.time()

    scheduler = StepLR(optimizer, step_size=1, gamma=.1)

    for epoch in range(1, 2):
        train_mnist(model, device, train_loader_mnist, optimizer, epoch)
        test_mnist(model, device, test_loader_mnist)
        scheduler.step()

    total_time = time.time() - start_time
    print('Total default network training time: {} seconds'.format(total_time))



def preprocess_decimal(bitsize, num_hash_tables, dataset1_mnist, dataset2_mnist):

    print("*"*40)
    print('Decimal LSH!')

    times = []

    start_time = time.time()
    hashed_dataset1_mnist = np.array([])
    lsh = LSHash(bitsize, 784, num_hash_tables)

    print("*"*40)
    print('Hashing dataset1_mnist... ')

    for i, value in tqdm(enumerate(dataset1_mnist)):
      lsh.index(value[0].flatten().numpy(), extra_data=(i, value[1]))

    print("*"*40)
    print('Hashing dataset2_mnist... ')

    for i, value in tqdm(enumerate(dataset2_mnist)):
      lsh.index(value[0].flatten().numpy(), extra_data=(i+len(dataset1_mnist), value[1]))


    total_time = time.time() - start_time
    times.append(total_time)
    # print('Total preprocessing time: {} seconds'.format(total_time))


    start_time = time.time()

    hashed_values = {}
    label_values = {}

    print("*"*40)
    print('Unpacking hash tables... ')

    for i, table in tqdm(enumerate(lsh.hash_tables)):
      for key in table.storage:
        decimal = int(key, 2)
        for value in table.storage[key]:
          index = value[1][0]
          label = value[1][1]

          curr_list = hashed_values.get(index, [])
          curr_list.append(decimal)

          if (label_values.get(index, -1) > 0) and label_values.get(index, -1) != label:
            print("BADBADBADBAD")
          label_values[index] = label

          hashed_values[index] = curr_list


    total_time = time.time() - start_time
    # print('Total preprocessing time (pt. 2): {} seconds'.format(total_time))
    times.append(total_time)


    start_time = time.time()

    hashed_train_x = np.array([hashed_values[0]])
    hashed_train_y = np.array([label_values[0]])

    for i in range(1, len(dataset1_mnist)):
      hashed_train_x = np.append(hashed_train_x, [hashed_values[i]], axis=0)
      hashed_train_y = np.append(hashed_train_y, label_values[i])

    hashed_test_x = np.array([hashed_values[len(dataset1_mnist)]])
    hashed_test_y = np.array([label_values[len(dataset1_mnist)]])

    for i in range(len(dataset1_mnist)+1, len(dataset1_mnist)+len(dataset2_mnist)):
      hashed_test_x = np.append(hashed_test_x, [hashed_values[i]], axis=0)
      hashed_test_y = np.append(hashed_test_y, label_values[i])


    total_time = time.time() - start_time
    # print('Total preprocessing time (pt. 3): {} seconds'.format(total_time))
    times.append(total_time)


    start_time = time.time()

    train_tensor_x = torch.Tensor(hashed_train_x) # transform to torch tensor
    train_tensor_y = torch.Tensor(hashed_train_y)

    # train_tensor_x = train_tensor_x.type(torch.LongTensor)
    train_tensor_y = train_tensor_y.type(torch.LongTensor)

    hashed_dataset1_mnist = TensorDataset(train_tensor_x, train_tensor_y) # create your datset

    test_tensor_x = torch.Tensor(hashed_test_x) # transform to torch tensor
    test_tensor_y = torch.Tensor(hashed_test_y)

    # test_tensor_x = test_tensor_x.type(torch.LongTensor)
    test_tensor_y = test_tensor_y.type(torch.LongTensor)

    hashed_dataset2_mnist = TensorDataset(test_tensor_x,test_tensor_y) # create your datset


    total_time = time.time() - start_time
    # print('Total preprocessing time (pt. 4): {} seconds'.format(total_time))
    times.append(total_time)


    train_loader_hashed_mnist = torch.utils.data.DataLoader(hashed_dataset1_mnist)
    test_loader_hashed_mnist = torch.utils.data.DataLoader(hashed_dataset2_mnist)

    return train_loader_hashed_mnist, test_loader_hashed_mnist, times


def run_hashed_model(train_loader_hashed_mnist, test_loader_hashed_mnist, model):

    accuracy = 0
    model = model
    device = torch.device("cpu")

    optimizer = optim.Adam(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=.1)

    start_time = time.time()
    for epoch in range(1, 5):
        train_mnist(model, device, train_loader_hashed_mnist, optimizer, epoch)
        accuracy = test_mnist(model, device, test_loader_hashed_mnist)
        scheduler.step()


    total_time = time.time() - start_time
    print('Total hashed network training time: {} seconds'.format(total_time))

    num_params = sum(p.numel() for p in model.parameters())
    return total_time, accuracy, num_params


def get_hash(lsh, img):
    hashes = []
    planes = lsh.uniform_planes

    for plane in planes:
        hash = lsh._hash(plane, img[0].flatten().numpy().tolist())

        hash = list(hash)

        for i, val in enumerate(hash):
            hash[i] = int(hash[i])

        hashes = hashes + hash

    return np.array(hashes)


def preprocess_binary(bitsize, num_hash_tables, dataset1_mnist, dataset2_mnist):
    times = []

    print("*"*40)
    print('Binary LSH!')

    lsh = LSHash(bitsize, 784, num_hash_tables)

    start_time = time.time()

    hashed_dataset_train = []
    mnist_labels_train = []

    hashed_dataset_test = []
    mnist_labels_test = []

    print("*"*40)
    print('Hashing dataset1_mnist... ')

    for image in tqdm(dataset1_mnist):
      hashed_dataset_train.append(get_hash(lsh, image))
      mnist_labels_train.append(image[1])

    print("*"*40)
    print('Hashing dataset1_mnist... ')

    for image in tqdm(dataset2_mnist):
      hashed_dataset_test.append(get_hash(lsh, image))
      mnist_labels_test.append(image[1])

    total_time = time.time() - start_time
    # print('Total preprocessing time (pt. 5): {} seconds'.format(total_time))
    times.append(total_time)

    hashed_dataset_train = np.array(hashed_dataset_train)
    mnist_labels_train = np.array(mnist_labels_train)

    hashed_dataset_test = np.array(hashed_dataset_test)
    mnist_labels_test = np.array(mnist_labels_test)


    start_time = time.time()

    train_tensor_x = torch.Tensor(hashed_dataset_train) # transform to torch tensor
    train_tensor_y = torch.Tensor(mnist_labels_train)

    # train_tensor_x = train_tensor_x.type(torch.LongTensor)
    train_tensor_y = train_tensor_y.type(torch.LongTensor)

    hashed_dataset1_mnist = TensorDataset(train_tensor_x, train_tensor_y) # create your datset

    test_tensor_x = torch.Tensor(hashed_dataset_test) # transform to torch tensor
    test_tensor_y = torch.Tensor(mnist_labels_test)

    # test_tensor_x = test_tensor_x.type(torch.LongTensor)
    test_tensor_y = test_tensor_y.type(torch.LongTensor)

    hashed_dataset2_mnist = TensorDataset(test_tensor_x,test_tensor_y) # create your datset


    total_time = time.time() - start_time
    # print('Total preprocessing time (pt. 6): {} seconds'.format(total_time))
    times.append(total_time)


    train_loader_hashed_mnist = torch.utils.data.DataLoader(hashed_dataset1_mnist)
    test_loader_hashed_mnist = torch.utils.data.DataLoader(hashed_dataset2_mnist)

    return train_loader_hashed_mnist, test_loader_hashed_mnist, times
