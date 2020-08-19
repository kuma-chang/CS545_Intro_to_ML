import numpy as np
import random
import math
import matplotlib.pyplot as plt                 # Load the matplotlib libraries with alias 'plt' for ploting
from sklearn.metrics import confusion_matrix    # Load the sklearn.metrics libraries for ploting the confusion matrix
import seaborn as sn                            # Load the seaborn libraries with alias 'sn' to generate heatmap for confusion matrix

def k_10_file_name_switch(center_index): 
    switcher = { 
        0: "k_10_center_0", 
        1: "k_10_center_1", 
        2: "k_10_center_2", 
        3: "k_10_center_3", 
        4: "k_10_center_4", 
        5: "k_10_center_5", 
        6: "k_10_center_6", 
        7: "k_10_center_7", 
        8: "k_10_center_8", 
        9: "k_10_center_9", 
    } 
    return switcher.get(center_index, "nothing")

def k_30_file_name_switch(center_index): 
    switcher = { 
        0: "k_30_center_0", 
        1: "k_30_center_1", 
        2: "k_30_center_2", 
        3: "k_30_center_3", 
        4: "k_30_center_4", 
        5: "k_30_center_5", 
        6: "k_30_center_6", 
        7: "k_30_center_7", 
        8: "k_30_center_8", 
        9: "k_30_center_9", 
        10: "k_30_center_10", 
        11: "k_30_center_11", 
        12: "k_30_center_12", 
        13: "k_30_center_13", 
        14: "k_30_center_14", 
        15: "k_30_center_15", 
        16: "k_30_center_16", 
        17: "k_30_center_17", 
        18: "k_30_center_18", 
        19: "k_30_center_19", 
        20: "k_30_center_20", 
        21: "k_30_center_21", 
        22: "k_30_center_22", 
        23: "k_30_center_23", 
        24: "k_30_center_24", 
        25: "k_30_center_25", 
        26: "k_30_center_26", 
        27: "k_30_center_27", 
        28: "k_30_center_28", 
        29: "k_30_center_29", 
    } 
    return switcher.get(center_index, "nothing")

class K_means:
    def __init__(self, k, range_of_input, size_of_input):
        self.centers_prev = np.zeros((k, size_of_input))
        self.centers = np.random.randint(range_of_input+1, size=(k, size_of_input))
        self.clusters = []
        self.class_label = []
        for x in range(len(self.centers)):
            self.clusters.append([])
            self.class_label.append(0)
    
    def print_centers(self):
        for i in range(len(self.centers)):
            print("Center", i, ":", self.centers[i], '\n')
    
    def train(self, p_train_data):
        print("Training", end="")
        while np.array_equal(self.centers_prev, self.centers) == False:
            print('.', end="", flush=True)
            self.clusters = []
            for x in range(len(self.centers)):
                self.clusters.append([])
            for data_index in range(len(p_train_data)):
                distances =[]
                cluster_assign = 0
                for i in range(len(self.centers)):
                    distances.append(np.sum(np.square(p_train_data[data_index] - self.centers[i])))
                for j in range(len(distances)):
                    if j != 0:
                        if distances[cluster_assign] > distances[j]:
                            cluster_assign = j
                self.clusters[cluster_assign].append(data_index)

            for i in range(len(self.clusters)):
                self.centers_prev[i] = self.centers[i]
                cluster_sum = np.zeros(len(p_train_data[0]))
                if self.clusters[i]:
                    for j in range(len(self.clusters[i])):
                        cluster_sum = np.add(cluster_sum, p_train_data[self.clusters[i][j]])
                    self.centers[i] = cluster_sum/len(self.clusters[i])
                else:
                    self.centers[i] = p_train_data[random.randint(0, len(p_train_data))]
        print()
        # for i in range(len(self.clusters)):
        #     print(i, len(self.clusters[i]))
        #     print(clusters[i])

    def mse(self, p_train_data):
        mse_table = []
        for i in range(len(self.clusters)):
            cluster_d_square_sum = np.zeros(len(p_train_data[0]))
            if self.clusters[i]:
                for j in range(len(self.clusters[i])):
                    cluster_d_square_sum = np.add(cluster_d_square_sum, np.square(p_train_data[self.clusters[i][j]] - self.centers[i]))
                mse_table.append(sum(cluster_d_square_sum)/len(self.clusters[i]))
            else:
                mse_table.append(0)
            
        mse = sum(mse_table)/len(mse_table)

        return mse

    def mss(self):
        mss_sum = 0
        for i in range(len(self.centers)-1):
            for j in range(i+1, len(self.centers)):
                mss_sum += np.sum(np.square(self.centers[i] - self.centers[j]))
        k = len(self.centers)
        mss = mss_sum/(k*(k-1)/2)

        return mss

    def m_entropy(self, number_of_classes, train_label):
        entropy_table = []
        mean_entropy = 0
        for i in range(len(self.clusters)):
            entropy_count = []
            for x in range(number_of_classes):
                entropy_count.append(0)
            for j in range(len(self.clusters[i])):
                entropy_count[int(train_label[self.clusters[i][j]])] += 1
            # print(entropy_count)
            self.assign_label(i, entropy_count)
            entropy_sum = 0
            for j in range(len(entropy_count)):
                if entropy_count[j]:
                    portion = entropy_count[j]/sum(entropy_count)
                    entropy_sum += (portion)*math.log2(portion)
            entropy_table.append(-entropy_sum)
        # print(entropy_table)

        for i in range(len(entropy_table)):
            mean_entropy += len(self.clusters[i])/len(train_label)*entropy_table[i]
        
        return mean_entropy
    
    def assign_label(self, i, entropy_count):
        label = 0
        for j in range(len(entropy_count)):
            if entropy_count[label] < entropy_count[j]:
                label = j
        self.class_label[i] = label

    def cal_accuracy(self, p_test_data, test_label):
        hit = 0
        for data_index in range(len(p_test_data)):
            distances =[]
            class_result = 0
            for i in range(len(self.centers)):
                distances.append(np.sum(np.square(p_test_data[data_index] - self.centers[i])))
            for j in range(len(distances)):
                if distances[class_result] > distances[j]:
                    class_result = j
            if self.class_label[class_result] == test_label[data_index]:
                hit += 1
        accuracy = hit/len(test_label)*100
        return accuracy

    def save_images(self, k):
        for i in range(len(self.centers)):
            if k == 10:
                file_name = k_10_file_name_switch(i)
            elif k == 30:
                file_name = k_30_file_name_switch(i)
            plot_title = "K = " + str(k) + " center no." + str(i) + ", assigned label: " + str(self.class_label[i])
            im = self.centers[i]
            im = im.reshape(8, 8)
            plt.figure()
            plt.gray()
            plt.imshow(im)
            plt.axis("off")
            plt.title(plot_title, fontsize = 20)
            plt.savefig(file_name, bbox_inches = 'tight')



