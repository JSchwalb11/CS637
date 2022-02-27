import numpy as np
import sys
from loss import loss


class model:

    def __init__(self, layer_dim, layer_activation, loss_type, loss_params):
        """

        :param layer_dim: list of num_nerons for each layer
        :param layer_activation: list of activation for each layer
        :param loss: loss function
        """
        self.layer_dim = layer_dim
        self.layer_activations = layer_activation
        self.loss = loss(loss_type=loss_type, loss_params=loss_params)
        self.weights = []  # init first weight to 1
        self.layers = []
        self.trace = []
        self.loss_history = []
        self.loss_history_dloss_dyi = []
        self.dloss_dzx = []
        self.dloss_dax = []
        self.dloss_dwx = []
        self.y_pred = None

        assert len(self.layer_dim) == len(self.layer_activations)  # input layer excluded in activation functions
        assert len(self.layer_dim) >= 2

        for i in range(0, len(self.layer_dim)):
            self.add_layer(num_neuron=self.layer_dim[i], activation=self.layer_activations[i])
            # if i == 0:
            # self.weights.append(np.ones(shape=(self.layer_dim[0])))
            if i > 0:
                self.weights.append(self.init_weights(size=(self.layer_dim[i - 1], self.layer_dim[i])))

    def add_layer(self, num_neuron, activation):
        _in = np.zeros((num_neuron, 1))

        if activation == 'relu':
            act = self.relu(_in)

        elif activation == 'sigmoid':
            act = self.sigmoid(_in)

        elif activation == 'softmax':
            act = self.softmax(_in)

        else:
            act = self.no_activation(_in)

        self.layers.append((_in, activation))

    def softmax(self, mat):
        if mat.ndim > 2:
            print("Error in Softmax input matrix, too many columns.")
            sys.exit(-1)

        a = np.zeros_like(mat)
        sum_si = np.sum(np.exp(mat))

        for i in range(0, mat.shape[0]):
            si = np.exp(mat[i])
            a[i] = si / sum_si

        return a

    def sigmoid(self, mat):
        if mat.ndim > 2:
            print("Error in Sigmoid input matrix, too many columns.")
            sys.exit(-1)

        a = 1 / (1 + np.exp(-mat))

        return a

    def relu(self, mat):
        if mat.ndim > 2:
            print("Error in Relu input matrix, too many columns.")
            sys.exit(-1)

        a = np.zeros_like(mat)

        for i in range(0, mat.shape[0]):
            if mat[i] >= 0:
                a[i] = mat[i]
            else:
                a[i] = 0

        return a

    def no_activation(self, mat):
        if mat.ndim > 2:
            print("Error in No_Activation input matrix, too many columns.")
            sys.exit(-1)

        return mat

    def init_weights(self, size):
        weights = np.zeros(shape=size)
        min_dim = min(weights.shape[0], weights.shape[1])
        for i in range(0, min_dim):
            weights[i, i] = 1

        return weights

    def foward_pass(self, data_point, y_true):
        """
        z1 = np.dot(w1.T, x)
        a1 = relu(z1)
        z2 = np.dot(w2.T, a1)
        a2 = sigmoid(z2)
        z3 = np.dot(w3.T, a2)
        yhat = softmax(z3)
        """

        # set first layer to the input data
        self.datapoint = data_point

        l1 = (self.datapoint, self.layers[0][1])
        self.layers[0] = l1

        for i in range(0, len(self.layer_dim) - 1):
            zx = self.step_forward(self.layers[i][0], self.weights[i])
            # activated = self.layers[i-1][1](aggregate)
            if self.layers[i][1] == 'relu':
                ax = self.relu(zx)

            elif self.layers[i][1] == 'sigmoid':
                ax = self.sigmoid(zx)

            elif self.layers[i][1] == 'softmax':
                ax = self.softmax(zx)

            else:
                ax = self.no_activation(zx)

            self.layers[i + 1] = (ax, self.layers[i + 1][1])

        for layer in self.layers:
            self.trace.append(layer)

        self.y_pred = self.layers[len(self.layers)-1][0]
        self.loss_history.append(self.loss.loss_func(y_pred=self.y_pred, y_true=y_true))

    def backward_pass(self, y_pred, y_true):
        #1
        dloss_dz3 = self.loss.dloss_dyi(y_pred=y_pred)

        #2
        a = self.layers[len(self.layers) - 2][0]
        b = dloss_dz3.T
        dloss_dw3 = np.dot(a, b)

        #3
        a = self.weights[len(self.weights) - 1]
        b = dloss_dz3
        dloss_da2 = np.dot(a, b)

        #4 sigmoid
        a = dloss_da2
        b = self.layers[len(self.layers) - 2][0] * (1 - self.layers[len(self.layers) - 2][0])
        #dloss_dz2 = np.dot(a, b)
        dloss_dz2 = a*b

        #5
        a = self.layers[len(self.layers) - 3][0]
        b = dloss_dz2.T
        dloss_dw2 = np.dot(a, b)

        #6
        a = self.weights[len(self.weights) - 2]
        b = dloss_dz2
        dloss_da1 = np.dot(a, b)

        #7 relu
        a = dloss_da1
        b = np.zeros_like(a)
        for i in range(0, len(self.datapoint) - 1):
            if self.datapoint[i] < 0:
                b[i] = 0
            else:
                b[i] = 1

        dloss_dz1 = a * b

        #8
        a = self.datapoint
        b = dloss_dz1.T
        dloss_dw1 = np.dot(a, b)

        print()

        """
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                self.dloss_dzx.append(y_pred - y_true)
                a = self.layers[i][0]
                b = self.dloss_dzx[(len(self.layers) - 1) - i].T
                self.dloss_dax = np.dot(a, b)
                print()

            else:
                self.dloss_dzx.append()
        print()"""


    def step_forward(self, previous_layer, weights):
        return np.dot(weights.T, previous_layer)


    def dump_model_summary(self):
        return self.trace
