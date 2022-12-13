import numpy as np
import matplotlib.pyplot as plt
import json


with open('formatted_results.txt', 'r') as f:
    b = f.readlines()

data = []
counter = 0
for line in b:
    data.append(json.loads(line[:-1]))

accuracy = []
counter = 0
bitsize = -1

training_time = []


for d in data:
    if counter == 0:
        accuracy.append([])
        bitsize += 1
    accuracy[bitsize].append(d['training_time'])
    counter += 1
    if counter == 8:
        counter = 0

print(accuracy)

for row in accuracy:
    training_time.append(row[5])

training_time = accuracy[3]
print(training_time)

accuracy = []
counter = 0
bitsize = -1


preprocessing_time = []

for d in data:
    if counter == 0:
        accuracy.append([])
        bitsize += 1
    accuracy[bitsize].append(d['preprocessing_time'])
    counter += 1
    if counter == 8:
        counter = 0


for row in accuracy:
    preprocessing_time.append(row[5])

preprocessing_time = accuracy[3]
print(preprocessing_time)

total_time = []
for i in range(len(training_time)):
    total_time.append(training_time[i] + preprocessing_time[i])

x = [1, 5, 10, 15, 20, 30, 50, 100]
default_network = [1105 for i in range(len(total_time))]

# Y-axis values
y1 = preprocessing_time

# Y-axis values
y2 = training_time

y3 = total_time

y4 = default_network

# Function to plot
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)

plt.title('Number of Hash Tables vs. Compute Time')
plt.xlabel('# of hash tables')
plt.ylabel('time (s)')

# Function add a legend
plt.legend(["preprocessing time", "training time", 'total time', 'non-LSH training time'], loc ="upper left")

# function to show the plot
plt.savefig('nothing.png')
