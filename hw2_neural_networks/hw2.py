#########################
# Michael Chang
# PSU ID: 916711480
#########################
import numpy as np                              # Load the numpy libraries with alias 'np' 
import pandas as pd                             # Load the Pandas libraries with alias 'pd'
import os                                       # Load the os libraries for geting file address
from neural_net import Neural_net               # Load the Neural Network class that I implimented
from cal_n_plot import *                        # Load the function for calculation and ploting that I implimented
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from progress.bar import IncrementalBar         # Load the progress.bar libraries for progress bar


# Setting up the data's file names and tunable variables
train_file_name = '/mnist_train.csv'
val_file_name = '/mnist_validation.csv'
total_epoch = 50                        # Total number of epochs is set to 50
learning_rate = 0.1                     # Learning rate is set to 0.1
hidden_units = [20, 50, 100]            # Three different number of hidden units for Experiment 1
fix_hidden_units = 100                  # Fixed hidden units is set to 100 for Experiment 2 & 3
portions = [0.25, 0.5]                  # Two different potions, one quarter and one half for Experiment 2
momentums = [0.25, 0.5, 0.95]           # Three different momentums for Experiment 3
fix_momentum = 0                        # Fixed momentum is set to 0 for Experiment 1 & 2
experiment_falg = [True, True, True]    # Flag to control if each experiment would be run


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

if(experiment_falg[0]):
    for num_of_hidden in hidden_units: # runs through three number of hidden units 20->50->100

        model = Neural_net(785, num_of_hidden, 10)

        t_accuracy = []     # Setting up a table to store the accuracy using t_data through each epoch
        v_accuracy = []     # Setting up a table to store the accuracy using v_data through each epoch

        plot_title = 'Number of hidden units: ' + str(num_of_hidden)
        bar = IncrementalBar(plot_title, max = total_epoch) # initiate the progress bar

        # Accuracy for epoch 0
        model.cal_accuracy(t_input_data, t_label, t_accuracy)
        model.cal_accuracy(v_input_data, v_label, v_accuracy)

        # Start training the model
        for epoch in range(total_epoch):
            for i in range(t_data_size):
                model.train(t_input_data[i], t_label[i], learning_rate, fix_momentum)

            # Collect the accuracy for each epoch
            model.cal_accuracy(t_input_data, t_label, t_accuracy)
            model.cal_accuracy(v_input_data, v_label, v_accuracy)

            bar.next()  # updating the progress bar
        bar.finish()


        # Plot the confusion matrix and save file
        e1_save_confusion_matrix(model, v_input_data, v_label, num_of_hidden) 


        # Print out the last 5 accuracy for both train and validation
        print("Last five accuracy(train): ", t_accuracy[-5:])
        print("Last five accuracy(validation): ", v_accuracy[-5:])

        # Plot the accuracy for both train and validation then save file
        creat_n_save_plot(t_accuracy, v_accuracy, plot_title, e1_save_file_name_switch(num_of_hidden))



if(experiment_falg[1]):
    for portion in portions:
        p_data = data_portion(t_data, portion)
        np.random.shuffle(p_data)       # Randomly shuffle the training data

        # Setting up the labels and input data for training 
        p_t_label = p_data[:,:1]
        p_t_input_data = p_data/225
        p_t_input_data[:,:1] = 1 #bias unit set to 1
        p_t_data_size = len(p_data)


        model = Neural_net(785, fix_hidden_units, 10)

        t_accuracy = []     # Setting up a table to store the accuracy using t_data through each epoch
        v_accuracy = []     # Setting up a table to store the accuracy using v_data through each epoch

        plot_title = 'Portion of data set to: ' + str(portion)
        bar = IncrementalBar(plot_title, max = total_epoch) # initiate the progress bar

        # Accuracy for epoch 0
        model.cal_accuracy(p_t_input_data, p_t_label, t_accuracy)
        model.cal_accuracy(v_input_data, v_label, v_accuracy)

        # Start training the model
        for epoch in range(total_epoch):
            for i in range(p_t_data_size):
                model.train(p_t_input_data[i], p_t_label[i], learning_rate, fix_momentum)

            # Collect the accuracy for each epoch
            model.cal_accuracy(p_t_input_data, p_t_label, t_accuracy)
            model.cal_accuracy(v_input_data, v_label, v_accuracy)

            bar.next()  # updating the progress bar
        bar.finish()

        # Plot the confusion matrix and save file
        e2_save_confusion_matrix(model, v_input_data, v_label, portion) 

        # Print out the last 5 accuracy for both train and validation
        print("Last five accuracy(train): ", t_accuracy[-5:])
        print("Last five accuracy(validation): ", v_accuracy[-5:])

        # Plot the accuracy for both train and validation then save file
        creat_n_save_plot(t_accuracy, v_accuracy, plot_title, e2_save_file_name_switch(portion))
        



if(experiment_falg[2]):
    for momentum in momentums: # runs through three number of hidden units 20->50->100

        model = Neural_net(785, fix_hidden_units, 10)

        t_accuracy = []     # Setting up a table to store the accuracy using t_data through each epoch
        v_accuracy = []     # Setting up a table to store the accuracy using v_data through each epoch

        plot_title = 'Momentum set to: ' + str(momentum)
        bar = IncrementalBar(plot_title, max = total_epoch) # initiate the progress bar

        # Accuracy for epoch 0
        model.cal_accuracy(t_input_data, t_label, t_accuracy)
        model.cal_accuracy(v_input_data, v_label, v_accuracy)

        # Start training the model
        for epoch in range(total_epoch):
            for i in range(t_data_size):
                model.train(t_input_data[i], t_label[i], learning_rate, momentum)

            # Collect the accuracy for each epoch
            model.cal_accuracy(t_input_data, t_label, t_accuracy)
            model.cal_accuracy(v_input_data, v_label, v_accuracy)

            bar.next()  # updating the progress bar
        bar.finish()

        # Plot the confusion matrix and save file
        e3_save_confusion_matrix(model, v_input_data, v_label, momentum) 

        # Print out the last 5 accuracy for both train and validation
        print("Last five accuracy(train): ", t_accuracy[-5:])
        print("Last five accuracy(validation): ", v_accuracy[-5:])

        # Plot the accuracy for both train and validation then save file
        creat_n_save_plot(t_accuracy, v_accuracy, plot_title, e3_save_file_name_switch(momentum))