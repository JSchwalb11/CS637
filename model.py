import numpy as np
import sys

class model:

    def __init__(self, layer_dim, layer_activation, loss):
        """

        :param layer_dim: list of num_nerons for each layer
        :param layer_activation: list of activation for each layer
        :param loss: loss function
        """
        self.layer_dim = layer_dim
        self.layer_activations = layer_activation
        self.loss = loss
        self.weights = [] # init first weight to 1
        self.layers = []
        self.trace = []


        assert len(self.layer_dim) == len(self.layer_activations) # input layer excluded in activation functions
        assert len(self.layer_dim) >= 2

        for i in range(0, len(self.layer_dim)):
            self.add_layer(num_neuron=self.layer_dim[i], activation=self.layer_activations[i])
            #if i == 0:
                #self.weights.append(np.ones(shape=(self.layer_dim[0])))
            if i > 0:
                self.weights.append(self.init_weights(size=(self.layer_dim[i-1], self.layer_dim[i])))


    def add_layer(self, num_neuron, activation):
        _in = np.zeros((num_neuron,))

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
        if mat.ndim > 1:
            print("Error in Relu input matrix, too many columns.")
            sys.exit(-1)

        a = np.zeros_like(mat)
        sum_si = np.sum(np.exp(mat))

        for i in range(0, mat.shape[0]):
            si = np.exp(mat[i])
            a[i] = si / sum_si

        return a

    def sigmoid(self, mat):
        if mat.ndim > 1:
            print("Error in Relu input matrix, too many columns.")
            sys.exit(-1)

        a = 1 / (1 + np.exp(-mat))

        return a

    def relu(self, mat):
        if mat.ndim > 1:
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
        if mat.ndim > 1:
            print("Error in Relu input matrix, too many columns.")
            sys.exit(-1)

        return mat

    def init_weights(self, size):
        weights = np.zeros(shape=size)
        min_dim = min(weights.shape[0], weights.shape[1])
        for i in range(0, min_dim):
            weights[i, i] = 1

        return weights

    def foward_pass(self, data_point):
        """
        z1 = np.dot(w1.T, x)
        a1 = relu(z1)
        z2 = np.dot(w2.T, a1)
        a2 = sigmoid(z2)
        z3 = np.dot(w3.T, a2)
        yhat = softmax(z3)
        """

        # set first layer to the input data
        l1 = (data_point, 'relu')
        self.layers[0] = l1

        for i in range(0, len(self.layer_dim) - 1):
            zx = self.step(self.layers[i][0], self.weights[i])
            #activated = self.layers[i-1][1](aggregate)
            if self.layers[i][1] == 'relu':
                ax = self.relu(zx)

            elif self.layers[i][1] == 'sigmoid':
                ax = self.sigmoid(zx)

            else:
                ax = self.no_activation(zx)

            self.layers[i+1] = (ax, self.layers[i+1][1])

            self.trace.append(zx) # used to trace intermittent values


        y_pred = self.softmax(self.layers[-1][0])

        print()

    def step(self, previous_layer, weights):
        return np.dot(weights.T, previous_layer)

    """
    class loss:
        def __init__(self, loss):
            if loss == 'categorical_crossentropy':
                self.loss = self.categorical_crossentropy()
            else:
                pass

        def categorical_crossentropy(self, y_pred, y_true):
            assert len(y_pred) == len(y_true)
            
            for i in range(0, y_pred):
                losses = y_true[i] * np.log(y_pred[i])
            sum_losses = - np.sum(losses)
            
            return sum_losses
            
    """

    def categorical_crossentropy(self, y_pred, y_true):
        assert len(y_pred) == len(y_true)

        for i in range(0, y_pred):
            losses = y_true[i] * np.log(y_pred[i])
        sum_losses = - np.sum(losses)

        return sum_losses