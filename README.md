Adapting Locality Sensitive Hashing for Higher Efficiency and Safer Machine Learning

Final Project for Harvard CS 242: Computing at Scale

Collaborators: Justin Ye, Richard Lun, William Cooper

This repository contains the code with which we used to test the use of locality-sensitive hashing on
inputs of neural networks to speed up training. Below is a brief description of each important file, and directions to run the code. Requirements are located in requirements.txt.  


models.py: contains all models used in evaluation: MNISTNet, HashedMNISTNet, DeeperHashedMNISTNET, GNNNodeClassifier

helpers.py: helper functions for MLPs.

gnn_helpers.py: helper functions for both regular and hashed GNNs

systematic-parameter-tuning.py: run this file to see hashed MLPs in action.

baseline_gnn.py: run this file to train a fully connected neural network on Cora dataset paper data, and to train a GNN on the Cora dataset paper and citation data

hashed_gnn.py: run this file to train a fully connected neural network on Cora dataset paper data, and to train a GNN on hashed paper data values and citation data

CS242_Final_MNIST.ipynb: notebook containing code for our first HashedMNISTNet, as well as some other scratch functions we considered, for data augmentation or different hash functions
