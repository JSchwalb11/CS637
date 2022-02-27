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
        self.fp_trace = []
        self.bp_trace = []
        self.bp_trace_test = []
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
            self.fp_trace.append(layer)

        self.y_pred = self.layers[len(self.layers)-1][0]
        self.loss_history.append(self.loss.loss_func(y_pred=self.y_pred, y_true=y_true))

    def backward_pass(self, y_pred):
        self.step_backward(y_pred=y_pred)

        weights_copy = self.weights.copy()

        num_steps = 2 * len(self.layers) - 1

        dloss_dzx = self.loss.dloss_dyi(y_pred=y_pred)
        self.loss_history_dloss_dyi.append(dloss_dzx)
        self.dloss_dzx.append(dloss_dzx)
        self.bp_trace.append(('Step {0} <dloss_dzx>'.format(2 * len(self.layers) - 1), dloss_dzx))
        self.layers.pop()

        for i in range(num_steps - 1, -1, -1):
            if i % 3 == 0:
                # dloss_dwx
                layer = self.layers.pop()
                a = layer[0]
                b = self.dloss_dzx.pop()
                dloss_dwx = np.dot(a, b.T)
                self.dloss_dwx.append(dloss_dwx)
                self.dloss_dzx.append(b) # put back for use in dloss_dax
                self.bp_trace.append(('Step {0} <dloss_dwx>'.format(i), dloss_dwx))
                self.layers.append(layer)

            elif i % 3 == 1:
                # dloss_dzx
                current_layer = self.layers.pop()
                next_layer = self.layers.pop()

                aggregate = current_layer[0]
                activation_type = next_layer[1]

                a = self.dloss_dax.pop()

                if activation_type == 'sigmoid':
                    b = self.sigmoid_derivative(aggregate) # sigmoid derivative, a function of the activated layer
                elif activation_type == 'relu':
                    b = self.relu_derivative(aggregate) # relu derivative, a function of the activation itself

                else:
                    b = 1 # derivative of identity function

                dloss_dzx = a * b
                self.dloss_dzx.append(dloss_dzx) # not matrix multiplication
                self.bp_trace.append(('Step {0} <dloss_dzx>'.format(i), dloss_dzx))
                self.layers.append(next_layer) # put the layer back for use in dloss_dwx
                # self.layers.append(current_layer) # toss current layer since we're done with it

            elif i % 3 == 2:
                # dloss_dax
                a = weights_copy.pop()
                b = self.dloss_dzx.pop()
                dloss_dax = np.dot(a, b)
                self.dloss_dax.append(dloss_dax)
                self.bp_trace.append(('Step {0} <dloss_dax>'.format(i), dloss_dax))
        print()


    def step_forward(self, previous_layer, weights):
        return np.dot(weights.T, previous_layer)

    def step_backward(self, y_pred):

        #7
        dloss_dz3 = self.loss.dloss_dyi(y_pred=y_pred)
        self.bp_trace_test.append(('Step {0} <dloss_dzx>'.format(str(7)), dloss_dz3))


        # 6
        a = self.layers[len(self.layers) - 2][0]
        b = dloss_dz3.T
        dloss_dw3 = np.dot(a, b)
        self.bp_trace_test.append(('Step {0} <dloss_dwx>'.format(str(6)), dloss_dw3))

        # 5
        a = self.weights[len(self.weights) - 1]
        b = dloss_dz3
        dloss_da2 = np.dot(a, b)
        self.bp_trace_test.append(('Step {0} <dloss_dax>'.format(str(5)), dloss_da2))

        # 4 sigmoid
        a = dloss_da2
        aggregate = self.layers[len(self.layers) - 2][0]
        b = aggregate * (1 - aggregate)
        # dloss_dz2 = np.dot(a, b)
        dloss_dz2 = a * b
        self.bp_trace_test.append(('Step {0} <dloss_dzx>'.format(str(4)), dloss_dz2))

        # 3
        a = self.layers[len(self.layers) - 3][0]
        b = dloss_dz2.T
        dloss_dw2 = np.dot(a, b)
        self.bp_trace_test.append(('Step {0} <dloss_dwx>'.format(str(3)), dloss_dw2))


        # 2
        a = self.weights[len(self.weights) - 2]
        b = dloss_dz2
        dloss_da1 = np.dot(a, b)
        self.bp_trace_test.append(('Step {0} <dloss_dax>'.format(str(2)), dloss_da1))


        # 1 relu
        aggregate = self.layers[len(self.layers) - 3][0]

        a = dloss_da1
        b = np.maximum(0, aggregate)

        dloss_dz1 = a * b
        self.bp_trace_test.append(('Step {0} <dloss_dzx>'.format(str(1)), dloss_dz1))


        # 0
        a = self.datapoint
        b = dloss_dz1.T
        dloss_dw1 = np.dot(a, b)
        self.bp_trace_test.append(('Step {0} <dloss_dwx>'.format(str(0)), dloss_dw1))


    def relu_derivative(self, mat):
        return np.maximum(mat, 0)

    def sigmoid_derivative(self, mat):
        return mat * (1 - mat)

    def dump_model_summary(self):
        return self.fp_trace
