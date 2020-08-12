#########################
# Michael Chang
# PSU ID: 916711480
#########################
import sys
import numpy as np                              # Load the numpy libraries with alias 'np' 
import os                                       # Load the os libraries for geting file address
import random



# getting the path for bot training ant test data
files = sys.argv[1:]
for file in files:
    if os.path.exists(file):
        print(os.path.basename(file)) 

train_data = np.genfromtxt(files[0])
test_data = np.genfromtxt(files[1])

s_train_data = train_data[train_data[:,-1].argsort()]

train_class_label = []
train_class_label.append(s_train_data[0, -1])
class_flag = s_train_data[0, -1]
number_of_classes = 1
data_count_table = []
data_count = 0
for x in s_train_data:
    if class_flag != x[-1]:
        class_flag = x[-1]
        train_class_label.append(x[-1])
        number_of_classes += 1
        data_count_table.append(data_count)
        data_count = 1
    else:
        data_count += 1
data_count_table.append(len(s_train_data)-sum(data_count_table))

number_of_attributes = len(s_train_data[0]) - 1

class_prob_table = []
for i in range(number_of_classes):
    class_prob_table.append(data_count_table[i]/sum(data_count_table))
print("Class label:\t", train_class_label)
print("Data cout:\t", data_count_table)
print("Number of classes: ", number_of_classes, "\nNumber of attributes: ", number_of_attributes)
print("Class probability: \n", class_prob_table)


d_train_data = []
for i in range(number_of_classes):
    d_train_data.append(np.zeros((data_count_table[i], number_of_attributes)))
    # print(d_train_data[i].shape)

copy_count = 0
for i in range(number_of_classes):
    for j in range(data_count_table[i]):
        d_train_data[i][j] = s_train_data[copy_count][:number_of_attributes]
        copy_count += 1
    d_train_data[i] = d_train_data[i].T

m_and_s_table = np.zeros((number_of_classes, number_of_attributes, 2))



for i in range(number_of_classes):
    for j in range(number_of_attributes):
        m_and_s_table[i][j][0] = np.sum(d_train_data[i][j])/len(d_train_data[i][j]) 
        m_and_s_table[i][j][1] = np.sqrt(np.sum(np.square(d_train_data[i][j]-m_and_s_table[i][j][0]))/len(d_train_data[i][j]))
        if m_and_s_table[i][j][1] < 0.01:
            m_and_s_table[i][j][1] = 0.01
        print("Class %d, attribute %d, mean = %.2f, std = %.2f" % 
        (train_class_label[i], 
                j+1, 
                m_and_s_table[i][j][0], 
            m_and_s_table[i][j][1]))
    print()





def normal_distribution(x, mean, std):
    return 1/(std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * std**2))
# print(normal_distribution(5.2, 4.8, 1.8))
# print(normal_distribution(6.3, 7.1, 2.0))
# print(normal_distribution(5.3, 4.7, 2.5))
# print(normal_distribution(6.3, 4.2, 3.7))


def classifier(prediction_table):
    index = 0
    predictions = []
    for i in range(number_of_classes):
        if i == 0:
            predictions.append((i, prediction_table[i]))
        elif predictions[0][1] < prediction_table[i]:
            predictions = []
            predictions.append((i, prediction_table[i]))
        elif predictions[0][1] == prediction_table[i]:
            predictions.append((i, prediction_table[i]))

    return predictions

def cal_accuracy(true_class, predictions, accuracy_table, tie_flag):
    for i in range(number_of_classes):
        if train_class_label[i] == true_class:
            true_index = i
    if tie_flag == 0:
        if predictions[0][0] == true_index:
            accuracy_table.append(1)
        else:
            accuracy_table.append(0)
    else:
        hit_flag = False
        for i in range(len(predictions)):
            if predictions[i][0] == true_index:
                hit_flag = True
        if hit_flag:
            accuracy_table.append(1/len(predictions))
        else:
            accuracy_table.append(0)


accuracy_table = []
for data_id in range(len(test_data)):
    true_class = test_data[data_id][-1]
    normal_dis_table = np.zeros((number_of_classes, number_of_attributes+1))
    prediction_table = np.zeros(number_of_classes) 
    for i in range(number_of_classes):
        for j in range(number_of_attributes+1):
            if j < number_of_attributes:
                normal_dis_table[i][j] = normal_distribution(test_data[data_id][j], m_and_s_table[i][j][0], m_and_s_table[i][j][1])
            else:
                normal_dis_table[i][j] = class_prob_table[i]
        prediction_table[i] = np.sum(np.log(normal_dis_table[i]))
    predictions = classifier(prediction_table)
    
    if len(predictions) < 2:
        prediction = train_class_label[predictions[0][0]]
        probability = predictions[0][1]
        cal_accuracy(true_class, predictions, accuracy_table, 0)
        accuracy = accuracy_table[data_id]
    else:
        random_pick = random.randint(0,len(predictions))
        prediction = train_class_label[predictions[random_pick][0]]
        probability = class_prob_table[predictions[random_pick][0]]
        cal_accuracy(true_class, predictions, accuracy_table, 0)
        accuracy = accuracy_table[data_id]
    

    print("ID=%5d, predicted=%3d, probability=%.4f, true=%3d, accuracy=%4.2f" % (data_id+1, prediction, probability, true_class, accuracy))


overall_accuracy = sum(accuracy_table)/len(accuracy_table)
print("Classification accuracy=%6.4f" % (overall_accuracy))