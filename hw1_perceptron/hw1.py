import numpy as np                              # Load the numpy libraries with alias 'np' 
import pandas as pd                             # Load the Pandas libraries with alias 'pd'
import os                                       # Load the os libraries for geting file address
from perceptron import Perceptron               # Load the Perceptron class that I implimented
from cal_n_plot import *                        # Load the function for calculation and ploting that I implimented
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from progress.bar import IncrementalBar         # Load the progress.bar libraries for progress bar


# Setting up the data's file names and tunible variables
train_file_name = '/mnist_train.csv'
val_file_name = '/mnist_validation.csv'
total_epoch = 50 #would have to be set to 50
learning_rates = [0.00001, 0.001, 0.1]

# Reading in all the data for both train and validation
print("Processing training data...")
t_data = np.genfromtxt(os.getcwd() + train_file_name, delimiter=',')
np.random.shuffle(t_data)       # Randomly shuffle the training data
print("Processing validation data...")
v_data = np.genfromtxt(os.getcwd() + val_file_name, delimiter=',')
print("Data process done")

# Setting up the labels and input data for training 
t_label = t_data[:,:1]
t_input_data = t_data/225
t_input_data[:,:1] = 1 #bias unit set to 1
t_data_size = len(t_data)

# Setting up the labels and input data for validation 
v_label = v_data[:,:1]
v_input_data = v_data/225
v_input_data[:,:1] = 1 #bias unit set to 1
v_data_size = len(v_data)



for learning_rate in learning_rates: # runs through three learning rate 0.00001->0.001->0.1

    # Setting up the model to hold ten perceptron one for each digit
    model = []
    for i in range(10):
        model.append(Perceptron())

    t_accuracy = []     # Setting up a table to store the accuracy using t_data through each epoch
    v_accuracy = []     # Setting up a table to store the accuracy using v_data through each epoch

    plot_title = 'Learning rate set to ' + str(learning_rate)
    bar = IncrementalBar(plot_title, max = total_epoch) # initiate the progress bar

    # Accuracy for epoch 0
    cal_accuracy(model, t_input_data, t_accuracy, t_label)
    cal_accuracy(model, v_input_data, v_accuracy, v_label)

    # Start training the model
    for epoch in range(total_epoch):
        for i in range(t_data_size):
            for j in range(10):
                if t_label[i] == j:     # Setting the target
                    target = 1
                else:
                    target = 0
                model[j].train(target, model[j].dot(t_input_data[i]), t_input_data[i], learning_rate)

        # Collect the accuracy for each epoch
        cal_accuracy(model, t_input_data, t_accuracy, t_label)
        cal_accuracy(model, v_input_data, v_accuracy, v_label)

        bar.next()  # updating the progress bar
    bar.finish()

    # Plot the confusion matrix and save file
    save_confusion_matrix(model, v_input_data, v_label, learning_rate) 

    # Plot the accuracy for both train and validation then save file
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%) ")
    plt.title(plot_title)
    plt.plot(t_accuracy, label = "train")
    plt.plot(v_accuracy, label = "validation")
    plt.legend()
    plt.savefig(save_file_name_switch(learning_rate))
    print(save_file_name_switch(learning_rate)+".png have been saved")
