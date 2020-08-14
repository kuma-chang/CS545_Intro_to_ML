#########################
# Michael Chang
# PSU ID: 916711480
#########################

import sys
import numpy as np                              # Load the numpy libraries with alias 'np' 
import os                                       # Load the os libraries for geting file address
from k_means import K_means

range_of_input = 16
k = 10
repeat = 5

# getting the path for bot training ant test data
files = sys.argv[1:]
for file in files:
    if os.path.exists(file):
        print(os.path.basename(file)) 

train_data = np.genfromtxt(files[0], delimiter=',')
test_data = np.genfromtxt(files[1], delimiter=',')

p_train_data = train_data[:, :-1]
size_of_input = len(p_train_data[0])

print(p_train_data.shape)

model_table = []
for i in range(repeat):
    model = K_means(k, range_of_input, size_of_input)
    model.train(p_train_data)
    model_table.append((model.mse(p_train_data), model))
    model.mss()

best_run = 0
for i in range(len(model_table)):
    if i != 0:
        if model_table[best_run][0] > model_table[i][0]:
            best_run = i
print("Best run: ", best_run)
model_table[best_run][1].mss()