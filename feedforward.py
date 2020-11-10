import gzip
import pickle
import random

import numpy as np


def load_data(file):
    with gzip.open(file, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin')

    train_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    train_labels = [np.eye(10)[y].reshape(10, 1) for y in train_set[1]]  # one-hot representation
    train_set = list(zip(train_inputs, train_labels))

    valid_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    valid_set = list(zip(valid_inputs, valid_set[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_set = list(zip(test_inputs, test_set[1]))

    return train_set, valid_set, test_set


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def soft_max(z):
    e = np.exp(z)
    return e / np.sum(e, axis=0)


class FeedforwardNetwork:
    def __init__(self, layers):
        self.layers = layers
        # weights between input-hidden, hidden-hidden and hidden-output layers
        self.weights = [np.random.randn(j, k) / np.sqrt(k) for k, j in zip(layers[:-1], layers[1:])]
        # biases for neurons in hidden and output layers
        self.biases = [np.random.randn(k, 1) for k in layers[1:]]

    def mini_batch(self, train_set, mini_batch_size, epochs, learning_rate, reg_param, friction, test_set=None):
        initial_epochs = epochs
        while epochs > 0:
            random.shuffle(train_set)
            mini_batches = [train_set[i:i + mini_batch_size] for i in range(0, len(train_set), mini_batch_size)]
            for mini_batch in mini_batches:
                # apply gradient descent for each mini batch
                d_w = [np.zeros(w.shape) for w in self.weights]
                d_b = [np.zeros(b.shape) for b in self.biases]
                momentum = [np.zeros(w.shape) for w in self.weights]
                for x, t in mini_batch:
                    nabla_c = self.backpropagation(x, t)  # nabla_C = (d_C/d_w, d_C/d_b)
                    d_w = [dw + ndw for dw, ndw in zip(d_w, nabla_c[0])]
                    d_b = [db + ndb for db, ndb in zip(d_b, nabla_c[1])]
                    momentum = [friction * v - learning_rate * dw for v, dw in zip(momentum, nabla_c[0])]
                m = len(mini_batch)
                n = len(train_set)
                self.weights = [v + (1 - learning_rate * (reg_param / n)) * w - (learning_rate / m) * dw
                                for w, dw, v in zip(self.weights, d_w, momentum)]  # using L2 regularization + momentum
                self.biases = [b - (learning_rate / m) * db for b, db in zip(self.biases, d_b)]
            if test_set:
                print('Accuracy after epoch {0}: {1}%'.format(initial_epochs - epochs + 1, self.get_accuracy(test_set)))
            epochs -= 1

    def backpropagation(self, x, t):
        d_w = [np.zeros(w.shape) for w in self.weights]
        d_b = [np.zeros(b.shape) for b in self.biases]
        # compute y^l and z^l
        y = x
        ys = [y]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, y) + b
            zs.append(z)
            y = sigmoid(z)
            ys.append(y)
        # use softmax activation function for output layer
        ys[-1] = soft_max(zs[-1])
        # compute error for output layer (using cross-entropy)
        error_L = ys[-1] - t
        d_b[-1] = error_L
        d_w[-1] = np.dot(error_L, ys[-2].transpose())
        # process layers below
        error_l = error_L
        for l in range(2, len(self.layers)):
            # compute error for previous layer
            error_l = np.dot(self.weights[-l + 1].transpose(), error_l) * sigmoid_derivative(zs[-l])
            # compute the gradient for weights in current layer
            d_w[-l] = np.dot(error_l, ys[-l - 1].transpose())
            # compute the gradient for biases in current layer
            d_b[-l] = error_l

        return d_w, d_b

    def get_accuracy(self, test_set):
        correct_outputs = 0
        for x, t in test_set:
            if np.argmax(self.feedforward(x)) == t:
                correct_outputs += 1
        return correct_outputs / len(test_set) * 100

    def feedforward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x


if __name__ == '__main__':
    _train_set, _valid_set, _test_set = load_data('src/mnist.pkl.gz')
    neural_network = FeedforwardNetwork([784, 100, 10])
    neural_network.mini_batch(_train_set, 10, 30, 0.1, 5, 0.9, _test_set)
