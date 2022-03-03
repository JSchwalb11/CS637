import numpy as np
import re
from loss import loss


class Layer:
    def __init__(self, in_shape, out_shape, activation, init_weight_type, momentum=1, learning_rate=0.001, trainable=True, loss_type=None):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activation = activation
        self.init_weight_type = init_weight_type
        self.loss_type = loss_type

        search = re.search(('gaussian'), self.init_weight_type)
        if search:
            search1 = re.search('var=', init_weight_type)
            if search1:
                self.var = int(self.init_weight_type[search1.end():])
            else:
                self.var = 1
            self.init_weight_type = 'gaussian'


        self.init_activations()
        self.bias = np.zeros(out_shape)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.trainable = trainable
        self.head = None
        self.tail = None

        if loss_type is not None:
            self.loss = loss(loss_type=loss_type).loss_func
            self.dloss_dyi = loss(loss_type=loss_type).dloss_dyi
            self.init_weights()
            self.weight_dict['identity']()
        else:
            self.init_weights()
            self.weight_dict[self.init_weight_type]()

        self.output = np.zeros(out_shape)
        self.dzx = np.zeros(out_shape)
        self.dwx = np.zeros((in_shape, out_shape))
        self.dax = np.zeros(out_shape)

    def foward(self, inputs):
        z = np.dot(self.weights.T, inputs) + self.bias
        self.output = self.activation_dict[self.activation]['func'](z)
        """if np.sum(self.output) == 0:
            breakpoint()"""
        return self.output

    def backward(self, y_true):
        if self.tail == None:  # must be last activated layer in model
            return

        if self.activation == 'softmax':
            self.dzx = self.activation_dict[self.activation]['derivative'](self.tail.output, y_true)
        elif self.tail.loss_type == 'hinge':
            self.dzx = self.tail.dloss_dyi(self.tail.output, y_true)
        else:
            a = self.dax
            b = self.activation_dict[self.activation]['derivative'](self.output)[np.newaxis].T
            if a.shape == b.shape:
                self.dzx = np.multiply(a, b)
            else:
                b = self.activation_dict[self.activation]['derivative'](self.output).T
                assert self.dzx.shape == self.dax.shape
                self.dzx = np.multiply(a, b)

        if self.trainable == True:
            try:
                a = self.head.output[np.newaxis].T
                b = self.dzx[np.newaxis]
                self.dwx = np.dot(a, b)
            except ValueError:
                a = self.head.output[np.newaxis].T
                b = self.dzx
                self.dwx = np.dot(a, b)

            delta = self.momentum * self.learning_rate * self.dwx
            """if np.sum(delta) == 0:
                breakpoint()"""
            self.weights = self.weights - delta

        self.head.dax = np.dot(self.weights, self.dzx)

    def sigmoid_derivative(self, mat):
        return self.sigmoid(mat) * (1 - self.sigmoid(mat))

    def sigmoid(self, mat):
        a = 1 / (1 + np.exp(-mat))
        return a

    def relu_derivative(self, mat):
        return (mat > 0) * 1.0

    def relu(self, mat):
        return np.maximum(mat, 0)

    def softmax_derivative(self, y_pred, y_true):
        return y_pred - y_true

    def softmax(self, mat):
        a = np.zeros_like(mat)
        sum_si = np.sum(np.exp(mat))

        for i in range(0, mat.shape[0]):
            si = np.exp(mat[i])
            a[i] = si / sum_si

        return a

    def no_activation_derivative(self, mat):
        return np.zeros_like(mat)

    def no_activation(self, mat):
        return mat

    def weight_init_random(self):
        weights = np.random.rand(self.in_shape, self.out_shape)
        self.weights = weights

    def weight_init_gaussian(self):
        weights = np.random.normal(0, self.var, size=(self.in_shape, self.out_shape))
        self.weights = weights

    def main_diag_init(self):
        weights = np.zeros(shape=(self.in_shape, self.out_shape))
        min_dim = min(weights.shape[0], weights.shape[1])
        for i in range(0, min_dim):
            weights[i, i] = 1

        self.weights = weights

    def identity_init(self):
        weights = np.zeros(shape=(self.in_shape, self.in_shape))
        for i in range(0, self.in_shape):
            weights[i, i] = 1

        self.weights = weights

    def init_activations(self):
        self.activation_dict = {'sigmoid': {'func': self.sigmoid, 'derivative': self.sigmoid_derivative},
                                'relu': {'func': self.relu, 'derivative': self.relu_derivative},
                                'softmax': {'func': self.softmax, 'derivative': self.softmax_derivative},
                                'none': {'func': self.no_activation, 'derivative': self.no_activation_derivative}}

    def init_weights(self):
        self.weight_dict = {'random': self.weight_init_random,
                            'gaussian': self.weight_init_gaussian,
                            'diagonal': self.main_diag_init,
                            'identity': self.identity_init}

if __name__ == '__main__':
    pass




