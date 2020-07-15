## HW1 Perceptron
#### Michael Chang

In this assignment we will try and write and train our own 10 perceptron that will, as a group, learn to classify handwritten digits (28*28 pixel), each perceptron corresponds to a digit, 0 to 9. We will use the given training data to help each perceptron learn by tuning the weights with the following learning algorithm:

<img src="https://render.githubusercontent.com/render/math?math=\Delta w_{i} = \eta(t^{k} - y^{k})x_{1}">

We will train the model in three different learning rates: 0.00001, 0.001, 0.1 And 50 epochs for each learning rate.