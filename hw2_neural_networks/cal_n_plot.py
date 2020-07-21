import numpy as np                              # Load the numpy libraries with alias 'np' 
import pandas as pd                             # Load the Pandas libraries with alias 'pd'
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from sklearn.metrics import confusion_matrix    # Load the sklearn.metrics libraries for ploting the confusion matrix
import seaborn as sn                            # Load the seaborn libraries with alias 'sn' to generate heatmap for confusion matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##################################################
# File name switch for each experiment
##################################################
def e1_save_file_name_switch(num_of_hidden): 
    switcher = { 
        20: "h_20_accur_plot", 
        50: "h_50_accur_plot", 
        100: "h_100_accur_plot", 
    } 
    return switcher.get(num_of_hidden, "nothing")

def e1_cm_save_file_name_switch(num_of_hidden): 
    switcher = { 
        20: "h_20_cm", 
        50: "h_50_cm", 
        100: "h_100_cm", 
    } 
    return switcher.get(num_of_hidden, "nothing")

def e2_save_file_name_switch(num_of_hidden): 
    switcher = { 
        0.25: "p_25_accur_plot", 
        0.5: "p_5_accur_plot", 
    } 
    return switcher.get(num_of_hidden, "nothing")

def e2_cm_save_file_name_switch(num_of_hidden): 
    switcher = { 
        0.25: "p_25_cm", 
        0.5: "p_5_cm", 
    } 
    return switcher.get(num_of_hidden, "nothing")

def e3_save_file_name_switch(momentum): 
    switcher = { 
        0.25: "m_25_accur_plot", 
        0.5: "m_5_accur_plot", 
        0.95: "m_95_accur_plot", 
    } 
    return switcher.get(momentum, "nothing")

def e3_cm_save_file_name_switch(momentum): 
    switcher = { 
        0.25: "m_25_cm", 
        0.5: "m_5_cm", 
        0.95: "m_95_cm", 
    } 
    return switcher.get(momentum, "nothing")


##################################################
# Save confusion matrix for each experiment
##################################################
def e1_save_confusion_matrix(model, input_data, label, num_of_hidden):
    prediction_table = []
    for i in range(len(input_data)):
        model.forward_propagation(input_data[i])
        for j in range(10):
            if j == 0:
                prediction = 0
                record = model.output_units[j]
            if record < model.output_units[j]:
                record = model.output_units[j]
                prediction = j
        prediction_table.append(prediction)     # Collect our model's prediction for each row of data

    cm = confusion_matrix(label, prediction_table) # Create the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10)) # Using Pandas to give index to our confusion matrix
    plot_title = 'Number of hidden units: ' + str(num_of_hidden)

    creat_n_save_heatmap(df_cm, plot_title, e1_cm_save_file_name_switch(num_of_hidden))

def e2_save_confusion_matrix(model, input_data, label, portion):
    prediction_table = []
    for i in range(len(input_data)):
        model.forward_propagation(input_data[i])
        for j in range(10):
            if j == 0:
                prediction = 0
                record = model.output_units[j]
            if record < model.output_units[j]:
                record = model.output_units[j]
                prediction = j
        prediction_table.append(prediction)     # Collect our model's prediction for each row of data

    cm = confusion_matrix(label, prediction_table) # Create the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10)) # Using Pandas to give index to our confusion matrix
    plot_title = 'Portion of data set to: ' + str(portion)

    creat_n_save_heatmap(df_cm, plot_title, e2_cm_save_file_name_switch(portion))

def e3_save_confusion_matrix(model, input_data, label, momentum):
    prediction_table = []
    for i in range(len(input_data)):
        model.forward_propagation(input_data[i])
        for j in range(10):
            if j == 0:
                prediction = 0
                record = model.output_units[j]
            if record < model.output_units[j]:
                record = model.output_units[j]
                prediction = j
        prediction_table.append(prediction)     # Collect our model's prediction for each row of data

    cm = confusion_matrix(label, prediction_table) # Create the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10)) # Using Pandas to give index to our confusion matrix
    plot_title = 'Momentum set to: ' + str(momentum)

    creat_n_save_heatmap(df_cm, plot_title, e3_cm_save_file_name_switch(momentum))


# Plot the accuracy for both train and validation then save file
def creat_n_save_plot(t_accuracy, v_accuracy, plot_title, file_name):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%) ")
    plt.title(plot_title)
    plt.plot(t_accuracy, label = "train")
    plt.plot(v_accuracy, label = "validation")
    plt.legend()
    plt.savefig(file_name)
    print(file_name+".png have been saved")

def creat_n_save_heatmap(df_cm, plot_title, file_name):
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


    plt.title(plot_title)
    plt.savefig(file_name)
    print(file_name+".png have been saved")

# Dividing the data in to respectively portion, whild approximately balanced
def data_portion(input_data, portion):
    s_input_data = input_data[input_data[:,0].argsort()]
    s_input_data = np.split(s_input_data, 10)
    portion_index = int(len(input_data)*portion/10)
    for i in range(10):
        np.random.shuffle(s_input_data[i]) 
        if i == 0:
            o_input_data = s_input_data[i][:portion_index]
        else:
            o_input_data = np.concatenate((o_input_data, s_input_data[i][:portion_index]))
    
    return o_input_data


