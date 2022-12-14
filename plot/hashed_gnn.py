import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lshashpy3 import LSHash
from models import GNNNodeClassifier
from gnn_helpers import *
import time



papers, citations = retrieve_data()

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])


NUM_HASH_TABLES = 300
HASH_SIZE = 4

papers = hash_gnn_inputs(HASH_SIZE, NUM_HASH_TABLES, papers)

train_data, test_data = get_train_test_data(papers)


hidden_units = [32, 32]

learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]

# baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
# baseline_model.summary()
#
# history = run_experiment(baseline_model, x_train, y_train)
# display_learning_curves(history, 'baseline_nn')
# _, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
# print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
#
# new_instances = generate_random_instances(num_classes)
# logits = baseline_model.predict(new_instances)
# probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
# display_class_probabilities(probabilities)


# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
edges = citations[["source", "target"]].to_numpy().T
# Create an edge weights array of ones.
edge_weights = tf.ones(shape=edges.shape[1])
# Create a node features array of shape [num_nodes, num_features].
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
)
# Create graph info tuple with node_features, edges, and edge_weights.
graph_info = (node_features, edges, edge_weights)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)


gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

print("GNN output shape:", gnn_model([1, 10, 100]))

gnn_model.summary()


x_train = train_data.paper_id.to_numpy()
start = time.time()
history = run_experiment(gnn_model, x_train, y_train, learning_rate, num_epochs, batch_size)
end = time.time()
total_training_time = end - start
print("LSH GNN training time: " + str(total_training_time))

display_learning_curves(history, 'hashed_gnn')
x_test = test_data.paper_id.to_numpy()
_, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
