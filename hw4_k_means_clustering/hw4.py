#########################
# Michael Chang
# PSU ID: 916711480
#########################

import sys
import numpy as np                              # Load the numpy libraries with alias 'np' 
import os                                       # Load the os libraries for geting file address
from k_means import K_means
from cal_n_plot import *                        # Load the function for calculation and ploting that I implimented
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting

range_of_input = 16
number_of_classes = 10
k_list = [10, 30]
repeat = 5

train_file_name = '/optdigits/optdigits.train'
test_file_name = '/optdigits/optdigits.test'

train_data = np.genfromtxt(os.getcwd() + train_file_name, delimiter=',')
test_data = np.genfromtxt(os.getcwd() + test_file_name, delimiter=',')

p_train_data = train_data[:, :-1]
train_label = train_data[:, -1]
size_of_input = len(p_train_data[0])

p_test_data = test_data[:, :-1]
test_label = test_data[:, -1]

for k in k_list:
    model_table = []
    for i in range(repeat):
        print("Repeat: ", i+1)
        model = K_means(k, range_of_input, size_of_input)
        model.train(p_train_data)
        model_table.append((model.mse(p_train_data), model))

    best_run = 0
    for i in range(len(model_table)):
        if i != 0:
            if model_table[best_run][0] > model_table[i][0]:
                best_run = i
    
    print("<< k set to: {} >>".format(k))
    print("Best run: ", best_run)
    print("Average mean-square-error: ", model_table[best_run][0])
    print("Mean-square-separation: ", model_table[best_run][1].mss())
    print("Mean entropy: ", model_table[best_run][1].m_entropy(number_of_classes, train_label))
    print("Assigned label: ", model_table[best_run][1].class_label)

    accuracy = model_table[best_run][1].cal_accuracy(p_test_data, test_label)
    print("Accuracy:{:.2f}%".format(accuracy))
    model_table[best_run][1].save_images(k)
    save_confusion_matrix(k, model_table[best_run][1], p_test_data, test_label)