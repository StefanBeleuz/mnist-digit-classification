# MNIST Dataset
The [dataset](http://deeplearning.net/data/mnist/mnist.pkl.gz) is split in training set, validation set and test set. 
Each of these three sets contains two vectors of equal length:
* A set of digits written as a vector of length 784. The digits from the dataset have the shape 28x28 pixels and 
are represented as a vector. Each pixel from the matrix has a value between 0 and 1, where 0 represents white, 
1 represents black and the value between 0 and 1 is a shade of grey.
* A label for each element from the first vector: a number between 0 and 9 representing the digit from the image.

# Single-layer perceptron

The classification algorithm is based on 10 perceptrons. Each of these 10 perceptrons is
trained to classify images that represent only one digit. For example, the first perceptron will be trained
to output value 1 for the digit 0 and value 0 for every other digit.

After each perceptron has been successfully trained, the input will be fed to each perceptron and the class will be 
given by the perceptron who has the biggest net input.

#### Results
I obtained an accuracy of 87.69% on the test set after 10 iterations and a learning rate of 0.1. 


# Feedforward network (Multi-layer perceptron)
* 3 layers: 
    * input: 784 neurons 
    * hidden: 100 neurons 
    * output: 10 neurons 
* activation function:
    * sigmoid (for neurons from hidden layer)
    * softmax (for neurons from output layer)
* cost function: cross-entropy
* weights are initialized as Gaussian random variables with mean 0 and standard deviation sqrt(1/n<sub>in</sub>) where 
n<sub>in</sub> is the total number of connection that go into the neuron
* weights are updated using L2 regularization + momentum

The class is given by the neuron’s number from the final layer with the greatest output value.

#### Results
I obtained an accuracy of 98.14% on the test set, using these hyper-parameters:
* size of the mini batch = 10
* epochs = 30
* learning rate = 0.1
* regularization parameter = 5.0
* momentum co-efficient (friction) = 0.9

# Bibliography
http://neuralnetworksanddeeplearning.com/