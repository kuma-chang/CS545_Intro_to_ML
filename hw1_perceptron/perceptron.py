import numpy as np                          # Load the numpy libraries with alias 'np' 


class Perceptron:
    # Generate initial small random weights between [-0.5,0.5]
    def __init__(self):
        self.weights = np.random.rand(785)-0.5
        self.weights = self.weights/10
    
    # Prints out all the weights for checking while developing
    def print_weights(self):
        print(self.weights)

    # Return its own weights and the input data's dot product
    def dot(self, input_data):
        return np.dot(self.weights, input_data)

    # Using the input data to update our weights in order to train this perceptron
    def train(self, target, dot_result, input_data, learning_rate):
        if dot_result > 0:
            y = 1
        else:
            y = 0
        self.weights = np.add(self.weights, learning_rate*(target-y)*input_data)
