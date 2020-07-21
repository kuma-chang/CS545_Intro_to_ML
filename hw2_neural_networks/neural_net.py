import numpy as np                                    # Load the numpy libraries with alias 'np' 
from cal_n_plot import sigmoid                        # Load the function for calculation and ploting that I implimented

class Neural_net:
    # Generate initial small random weights between [-0.05,0.05]
    def __init__(self, num_of_input, num_of_hidden, num_of_output):
        self.i_to_h_weights = np.random.rand(num_of_hidden, num_of_input)-0.5
        self.i_to_h_weights = self.i_to_h_weights/10
        self.h_to_o_weights = np.random.rand(num_of_output, num_of_hidden+1)-0.5
        self.h_to_o_weights = self.h_to_o_weights/10
        self.hidden_units = np.zeros(num_of_hidden+1)
        self.hidden_units[0] = 1
        self.output_units = np.zeros(num_of_output)
        self.update_i_to_h = np.zeros((num_of_hidden, num_of_input))
        self.update_h_to_o = np.zeros((num_of_output, num_of_hidden+1))
    
    # Prints out all the weights for checking while developing
    def print_weights(self):
        print("\nWeights from input layer to hidden layer: ")
        print(self.i_to_h_weights)
        print("\nWeights from hidden layer to output layer: ")
        print(self.h_to_o_weights)
    
    # Prints out all the units for checking while developing
    def print_units(self):
        print("\nHidden units: ")
        print(self.hidden_units)
        print("\nOutput units: ")
        print(self.output_units)

    # Calculate the accuracy according to the given input and store the result into the given table
    def cal_accuracy(self, input_data, input_label, accuracy_table):
        hit = 0
        for i in range(len(input_data)):
            self.forward_propagation(input_data[i])
            for j in range(len(self.output_units)):
                if j == 0:
                    prediction = 0
                    record = self.output_units[j]
                if record < self.output_units[j]:
                    record = self.output_units[j]
                    prediction = j
            if prediction == input_label[i]:
                hit += 1
        accuracy_table.append(hit/len(input_data)*100)

    # Forward propergation for this neural net
    def forward_propagation(self, input_data):
        self.hidden_units[1:] = sigmoid(np.dot(self.i_to_h_weights, input_data))
        self.output_units = sigmoid(np.dot(self.h_to_o_weights, self.hidden_units))

    def back_propergation(self, input_data, target, learning_rate, momentum):
        target_array = np.ones(len(self.output_units))/10
        target_array[int(target)] = 0.9

        error_output = np.multiply(np.multiply(self.output_units, (1-self.output_units)), (target_array - self.output_units))
        error_hidden = np.multiply(np.multiply(self.hidden_units, (1-self.hidden_units)), np.dot(error_output, self.h_to_o_weights))

        for i in range(len(self.output_units)):
            self.update_h_to_o[i] = (np.multiply(learning_rate, np.multiply(error_output[i], self.hidden_units))) + np.multiply(momentum, self.update_h_to_o[i])
        self.h_to_o_weights = self.h_to_o_weights + self.update_h_to_o

        for i in range(len(self.hidden_units)-1):
            self.update_i_to_h[i] = (np.multiply(learning_rate, np.multiply(error_hidden[i+1], input_data))) + np.multiply(momentum, self.update_i_to_h[i])
        self.i_to_h_weights = self.i_to_h_weights + self.update_i_to_h

        # print("\nError output:\n", error_output)
        # print("\nError hidden:\n", error_hidden)
        # self.print_weights()


    # Using the input data to update our weights in order to train this perceptron
    def train(self, input_data, target, learning_rate, momentum):
        self.forward_propagation(input_data)
        self.back_propergation(input_data, target, learning_rate, momentum)
        


