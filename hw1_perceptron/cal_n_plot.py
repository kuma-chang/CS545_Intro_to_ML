import numpy as np                              # Load the numpy libraries with alias 'np' 
import pandas as pd                             # Load the Pandas libraries with alias 'pd'
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from sklearn.metrics import confusion_matrix    # Load the sklearn.metrics libraries for ploting the confusion matrix
import seaborn as sn                            # Load the seaborn libraries with alias 'sn' to generate heatmap for confusion matrix

# File name switch for accuracy plot
def save_file_name_switch(learning_rate): 
    switcher = { 
        0.00001: "plot_l_r_00001", 
        0.001: "plot_l_r_001", 
        0.1: "plot_l_r_1", 
    } 
    return switcher.get(learning_rate, "nothing")

# File name switch for confusion matrix heatmap
def cm_save_file_name_switch(learning_rate): 
    switcher = { 
        0.00001: "cm_l_r_00001", 
        0.001: "cm_l_r_001", 
        0.1: "cm_l_r_1", 
    } 
    return switcher.get(learning_rate, "nothing")

# Calculate the accuracy to the input data and store result in the accuracy table
def cal_accuracy(model, input_data, accuracy, label):
    hit = 0
    for i in range(len(input_data)):
        for j in range(10):
            dot_result = model[j].dot(input_data[i])
            if j == 0:
                prediction = 0
                record = dot_result
            if record < dot_result:
                record = dot_result
                prediction = j
        if prediction == label[i]:
            hit += 1
    accuracy.append(hit/len(input_data)*100)

# Generate the confusion matrix and then create a heatmap based on it and save 
def save_confusion_matrix(model, input_data, label, learning_rate):
    prediction_table = []
    for i in range(len(input_data)):
        for j in range(10):
            dot_result = model[j].dot(input_data[i])
            if j == 0:
                prediction = 0
                record = dot_result
            if record < dot_result:
                record = dot_result
                prediction = j
        prediction_table.append(prediction)     # Collect our model's prediction for each row of data

    cm = confusion_matrix(label, prediction_table) # Create the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10)) # Using Pandas to give index to our confusion matrix


    # Using Matplotlib and Seaborn to creat heatmap and save
    plt.figure(figsize = (15,12))
    sn.heatmap(df_cm,                                       # Our confusion matrix with index 
                annot=True,                                 # Show annotaion 
                cmap="YlGnBu",                              # Choose color scheme 
                annot_kws={"size": 16},
                cbar_kws={"orientation": "horizontal"},     # Let color bar be horizontal
                fmt='g',                                    # Annotation be in number
                linewidths=.5)

    """
    Due to Matplotlib 3.1.1 have a bug that would cut off the top and bottom of the heatmap plot
    found the folloing fix
    https://github.com/mwaskom/seaborn/issues/1773
    """
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values


    plot_title = 'Learning rate set to ' + str(learning_rate)
    plt.title(plot_title)
    plt.savefig(cm_save_file_name_switch(learning_rate))
    print(cm_save_file_name_switch(learning_rate)+".png have been saved")