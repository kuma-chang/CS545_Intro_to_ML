import numpy as np
import random


class K_means:
    def __init__(self, k, range_of_input, size_of_input):
        self.centers_prev = np.zeros((k, size_of_input))
        self.centers = np.random.randint(range_of_input+1, size=(k, size_of_input))
        self.clusters = []
        for x in range(len(self.centers)):
            self.clusters.append([])
    
    def print_centers(self):
        for i in range(len(self.centers)):
            print("Center", i, ":", self.centers[i], '\n')
    
    def train(self, p_train_data):
        while np.array_equal(self.centers_prev, self.centers) == False:
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
        for i in range(len(self.clusters)):
            print(i, len(self.clusters[i]))
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
        print(mse)
        return mse

    def mss(self):
        mss_sum = 0
        for i in range(len(self.centers)-1):
            for j in range(i+1, len(self.centers)):
                mss_sum += np.sum(np.square(self.centers[i] - self.centers[j]))
        k = len(self.centers)
        mss = mss_sum/(k*(k-1)/2)
        print(mss)
        return mss

    def m_entropy(self):
        print("hello Yinzi!")
