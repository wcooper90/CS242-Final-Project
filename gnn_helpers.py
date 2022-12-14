# original GNN code from the following colab notebook:
# https://colab.research.google.com/drive/1qw1flWPXpm8MZkHlmVKetqgPj03Msjw-#scrollTo=j1aFCogG2XM6


import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lshashpy3 import LSHash
from tqdm import tqdm


# retrieve cora dataset
def retrieve_data():
    zip_file = keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")
    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    print("Citations shape:", citations.shape)
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
    )
    print("Papers shape:", papers.shape)
    return papers, citations


# get_hash function, adapted from MLP get_hash function
def get_hash(lsh, values):
    hashes = []
    planes = lsh.uniform_planes
    for plane in planes:
        hash = lsh._hash(plane, values)
        hash = list(hash)
        for i, val in enumerate(hash):
            hash[i] = int(hash[i])
        hashes = hashes + hash
    return np.array(hashes)


# hash and format cora dataset feature vector values
def hash_gnn_inputs(hashsize, num_hash_tables, input):

    # input dimension will always be 1433 for this dataset
    lsh = LSHash(hashsize, 1433, num_hash_tables)
    bruh = input.values.tolist()
    hashed_values = {}
    label_values = {}
    for i, value in tqdm(enumerate(bruh)):
        hashed_values[value[0]] = get_hash(lsh, value[1:-1])
        label_values[value[0]] = value[-1]
    new_dict = {}
    new_dict['paper_id'] = []
    for i in range(num_hash_tables * hashsize):
        new_dict['term_' + str(i)] = []
    new_dict['subject'] = []
    for key in hashed_values.keys():
        new_dict['paper_id'].append(key)
        for i in range(num_hash_tables * hashsize):
            new_dict['term_' + str(i)].append(hashed_values[key][i])
        new_dict['subject'].append(label_values[key])
    hashed_papers = pd.DataFrame(new_dict)
    print(hashed_papers.head())
    return hashed_papers


def get_train_test_data(papers):
    train_data, test_data = [], []
    for _, group_data in papers.groupby("subject"):
        # Select around 50% of the dataset for training.
        random_selection = np.random.rand(len(group_data.index)) <= 0.5
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    return train_data, test_data


def run_experiment(model, x_train, y_train, learning_rate, num_epochs, batch_size):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )
    return history


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)


def display_learning_curves(history, fig_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.savefig(fig_name + '.png')


def create_baseline_model(hidden_units, num_classes, num_features, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")


def generate_random_instances(num_instances):
    token_probability = x_train.mean(axis=0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size=len(token_probability))
        instance = (probabilities <= token_probability).astype(int)
        instances.append(instance)

    return np.array(instances)


def display_class_probabilities(probabilities):
    for instance_idx, probs in enumerate(probabilities):
        print(f"Instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f"- {class_values[class_idx]}: {round(prob * 100, 2)}%")
