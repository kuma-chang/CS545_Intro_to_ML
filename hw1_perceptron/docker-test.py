print("docker test 1")
print("docker test 2")
print("docker test 3")
print("docker test 4")
print("docker test 5")
import os                                       # Load the os libraries for geting file address
import numpy as np                              # Load the numpy libraries with alias 'np' 


train_file_name = '/mnist_train.csv'
val_file_name = '/mnist_validation.csv'


print("Processing validation data...")
v_data = np.genfromtxt(os.getcwd() + val_file_name, delimiter=',')
print("Data process done")

print("Processing training data...")
t_data = np.genfromtxt(os.getcwd() + train_file_name, delimiter=',')
print("now shuffle")
np.random.shuffle(t_data)       # Randomly shuffle the training data
print("Processing validation data...")
