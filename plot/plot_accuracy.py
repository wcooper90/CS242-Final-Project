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
    accuracy[bitsize].append(d['accuracy'])
    counter += 1
    if counter == 8:
        counter = 0

y1 = accuracy[0]
y2 = accuracy[2]
y3 = accuracy[4]
y4 = [97 for i in range(len(accuracy[4]))]

x = [1, 5, 10, 15, 20, 30, 50, 100]

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)

plt.title('Number of Hash Tables vs. Accuracy')
plt.xlabel('# of hash tables')
plt.ylabel('accuracy (%)')

# Function add a legend
plt.legend(["hashsize=1", "hashsize=4", 'hashsize=16', 'non-LSH network'], loc ="lower right")

# function to show the plot
plt.savefig('accuracy.png')
