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

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gnn_helpers import create_ffn


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


class DeeperHashedMNISTNetBinary(nn.Module):
    def __init__(self, num_hash_tables, bitsize, hidden_1=512, hidden_2=512):
      super(DeeperHashedMNISTNetBinary, self).__init__()

      self.hidden_1 = hidden_1
      self.hidden_2 = hidden_2
      self.num_hash_tables = num_hash_tables
      self.bitsize = bitsize

      # linear layer (784 -> hidden_1)
      self.fc1 = nn.Linear(num_hash_tables*bitsize, self.hidden_1)
      # linear layer (n_hidden -> hidden_2)
      self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
      # linear layer (n_hidden -> 10)
      self.fc3 = nn.Linear(self.hidden_2,10)
      # dropout layer (p=0.2)
      # dropout prevents overfitting of data
      self.droput = nn.Dropout(0.2)


    def forward(self, x):
      x = x.view(-1, self.num_hash_tables*self.bitsize)
      # add hidden layer, with relu activation function
      x = self.fc1(x)
      x = F.relu(x)
      # add dropout layer
      x = self.droput(x)
      # add hidden layer, with relu activation function
      x = self.fc2(x)
      x = F.relu(x)
      # add dropout layer
      x = self.droput(x)
      # add output layer
      x = self.fc3(x)

      return F.log_softmax(x, dim=1)


# original GNN code from the following colab notebook:
# https://colab.research.google.com/drive/1qw1flWPXpm8MZkHlmVKetqgPj03Msjw-#scrollTo=j1aFCogG2XM6


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim].
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)



class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)
