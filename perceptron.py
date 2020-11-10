import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 784  # images of 28x28 pixels


def activation(z):
    if z > 0:
        return 1
    else:
        return 0


class Perceptron:

    def __init__(self, digit, epochs=10, learning_rate=0.1):
        self.digit = digit
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(INPUT_SIZE)
        self.bias = 0

    def train(self, training_inputs, labels):
        while self.epochs > 0:
            for x, t in zip(training_inputs, labels):
                t = int(self.digit == t)  # target (expected output)
                z = np.dot(x, self.weights) + self.bias  # net input
                output = activation(z)  # classify sample
                self.weights = self.weights + (t - output) * x * self.learning_rate  # adjust weights
                self.bias = self.bias + (t - output) * self.learning_rate  # adjust bias
            self.epochs -= 1


def show_image(image):
    plt.imshow(image.reshape((28, 28)))
    plt.show()


def train_perceptrons(inputs, labels):
    perceptrons = []
    for _digit in range(10):
        perceptron = Perceptron(_digit)
        perceptron.train(inputs, labels)
        perceptrons.append(perceptron)

    return perceptrons


def test_perceptrons(perceptrons, inputs, labels):
    correct_outputs = 0
    for _input, _label in zip(inputs, labels):
        net_inputs = []
        for perceptron in perceptrons:
            net_input = np.dot(perceptron.weights, _input) + perceptron.bias
            net_inputs.append(net_input)
        if np.argmax(net_inputs) == _label:
            correct_outputs += 1

    return correct_outputs / len(inputs) * 100  # accuracy


if __name__ == '__main__':
    with gzip.open('src/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin')

    train_inputs, train_labels = train_set
    valid_inputs, valid_labels = valid_set
    test_inputs, test_labels = test_set

    # train perceptrons
    _perceptrons = train_perceptrons(train_inputs, train_labels)

    # validate classification
    print('Accuracy for validation: {0}%'.format(test_perceptrons(_perceptrons, valid_inputs, valid_labels)))

    # test classification
    print('Accuracy for testing: {0}%'.format(test_perceptrons(_perceptrons, test_inputs, test_labels)))
