import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import json
from altair_saver import save



data = []

with open('formatted_results.txt', 'r') as f:
    b = f.readlines()

counter = 0
for line in b:
    data.append(json.loads(line[:-1]))

accuracy = []
counter = 0
bitsize = -1

training_time = []

preprocessing_time = []

for d in data:
    if counter == 0:
        accuracy.append([])
        bitsize += 1
    accuracy[bitsize].append(d['accuracy'])
    counter += 1
    if counter == 8:
        counter = 0
counter = 0
bitsize = -1
for d in data:
    if counter == 0:
        training_time.append([])
        bitsize += 1
    training_time[bitsize].append(round(d['training_time'], 2))
    counter += 1
    if counter == 8:
        counter = 0


counter = 0
bitsize = -1
for d in data:
    if counter == 0:
        preprocessing_time.append([])
        bitsize += 1
    preprocessing_time[bitsize].append(round(d['preprocessing_time'], 2))
    counter += 1
    if counter == 8:
        counter = 0

accuracy = np.array(accuracy)


print(accuracy)


training_time = np.array(training_time)
preprocessing_time = np.array(preprocessing_time)

total_computation_time = np.add(training_time, preprocessing_time)

normalized = np.divide(accuracy, total_computation_time)
print(normalized)

x = np.array([1, 2, 4, 8, 16, 32])
y = np.array([1, 5, 10, 15, 20, 30, 50, 100])

# Compute x^2 + y^2 across a 2D grid

hash_sizes, num_hash_tables = np.meshgrid(x, y)


# Convert this grid to columnar data expected by Altair
source = pd.DataFrame({'hash_sizes': hash_sizes.ravel(),
                     'num_hash_tables': num_hash_tables.ravel(),
                     'accuracy': accuracy.ravel()})

chart1 = alt.Chart(source).mark_rect().encode(
    x='hash_sizes:O',
    y='num_hash_tables:O',
    color='accuracy:Q'
)

save(chart1, 'accuracy.html', inline=True)


source = pd.DataFrame({'hash_sizes': hash_sizes.ravel(),
                     'num_hash_tables': num_hash_tables.ravel(),
                     'training_time': training_time.ravel()})

chart2 = alt.Chart(source).mark_rect().encode(
    x='hash_sizes:O',
    y='num_hash_tables:O',
    color='training_time:Q'
)

save(chart2, 'training.html', inline=True)

source = pd.DataFrame({'hash_sizes': hash_sizes.ravel(),
                     'num_hash_tables': num_hash_tables.ravel(),
                     'preprocessing_time': preprocessing_time.ravel()})

chart3 = alt.Chart(source).mark_rect().encode(
    x='hash_sizes:O',
    y='num_hash_tables:O',
    color='preprocessing_time:Q'
)

save(chart3, 'preprocessing.html', inline=True)


source = pd.DataFrame({'hash_sizes': hash_sizes.ravel(),
                     'num_hash_tables': num_hash_tables.ravel(),
                     'acc / time': normalized.ravel()})

chart4 = alt.Chart(source).mark_rect().encode(
    x='hash_sizes:O',
    y='num_hash_tables:O',
    color='acc / time:Q'
)

save(chart4, 'normalized.html', inline=True)

chart5 = alt.hconcat(chart2, chart3)

save(chart5, 'hehe.html', inline=True)
