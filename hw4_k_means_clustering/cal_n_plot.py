import numpy as np                              # Load the numpy libraries with alias 'np' 
import pandas as pd                             # Load the Pandas libraries with alias 'pd'
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from sklearn.metrics import confusion_matrix    # Load the sklearn.metrics libraries for ploting the confusion matrix
import seaborn as sn                            # Load the seaborn libraries with alias 'sn' to generate heatmap for confusion matrix

def cm_save_file_name_switch(k): 
    switcher = { 
        10: "k_10_cm", 
        30: "k_30_cm", 
    } 
    return switcher.get(k, "nothing")

def save_confusion_matrix(k, model, p_test_data, test_label):
    prediction_table = []
    for data_index in range(len(p_test_data)):
        distances =[]
        class_result = 0
        for i in range(len(model.centers)):
            distances.append(np.sum(np.square(p_test_data[data_index] - model.centers[i])))
        for j in range(len(distances)):
            if distances[class_result] > distances[j]:
                class_result = j
        prediction_table.append(model.class_label[class_result])


    cm = confusion_matrix(test_label, prediction_table) # Create the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10)) # Using Pandas to give index to our confusion matrix
    plot_title = 'K set to: ' + str(k)

    creat_n_save_heatmap(df_cm, plot_title, cm_save_file_name_switch(k))

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


    plt.title(plot_title, fontsize=40)
    plt.savefig(file_name, bbox_inches = 'tight')
    print(file_name+".png have been saved")