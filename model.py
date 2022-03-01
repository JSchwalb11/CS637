import numpy as np
import sys
from loss import loss
import re


class model:

    def __init__(self, layer_dim, layer_activation, loss_type, loss_params, learning_rate, momentum, weight_initialization=None):
        """

        :param layer_dim: list of num_nerons for each layer
        :param layer_activation: list of activation for each layer
        :param loss: loss function
        """

        self.layer_dim = layer_dim
        self.layer_activations = layer_activation
        self.loss = loss(loss_type=loss_type, loss_params=loss_params)
        pattern = re.compile('var=')
        search = re.search(pattern, weight_initialization)
        if weight_initialization == 'random':
            self.weights_init = self.weight_init_random
        elif search:
            self.weight_init_var = float(weight_initialization[search.end():])
            self.weights_init = self.weight_init_gaussian
        else:
            self.weights_init = self.main_diag_init


        self.weights = []
        self.magnitude_change = []
        self.learning_rate = 1
        if learning_rate:
            self.learning_rate = learning_rate

        self.momentum = 1
        if momentum:
            self.momentum = momentum

        self.layers = []
        self.fp_trace = []
        self.bp_trace = []
        self.loss_val = []
        self.loss_history = []
        self.loss_history_dloss_dyi = []
        self.dloss_dzx = []
        self.dloss_dax = []
        self.dloss_dwx = []
        self.y_pred = None

        assert len(self.layer_dim) == len(self.layer_activations)  # input layer excluded in activation functions
        assert len(self.layer_dim) >= 2

        for i in range(0, len(self.layer_dim)):
            #self.add_layer(num_neuron=self.layer_dim[i], activation=self.layer_activations[i])

            if i > 0:
                self.weights.append(self.weights_init(size=(self.layer_dim[i - 1], self.layer_dim[i])))

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
        a = 1 / (1 + np.exp(-mat))
        return a

    def relu(self, mat):
        return np.maximum(mat, 0)

    def no_activation(self, mat):
        if mat.ndim > 2:
            print("Error in No_Activation input matrix, too many columns.")
            sys.exit(-1)

        return mat

    def main_diag_init(self, size):
        weights = np.zeros(shape=size)
        min_dim = min(weights.shape[0], weights.shape[1])
        for i in range(0, min_dim):
            weights[i, i] = 1

        return weights

    def weight_init_random(self, size):
        weights = np.random.rand(size[0]*size[1])
        weights = weights.reshape(size)
        #min_dim = min(weights.shape[0], weights.shape[1])
        #for i in range(0, min_dim):
        #    weights[i, i] = 1

        return weights

    def weight_init_gaussian(self, size):
        weights = np.random.normal(0, self.weight_init_var, size=size)
        #min_dim = min(weights.shape[0], weights.shape[1])
        #for i in range(0, min_dim):
        #    weights[i, i] = 1

        return weights

    def foward_pass(self, data_point, y_true, prediction=False):

        # set first layer to the input data
        self.datapoint = data_point
        #init_required = True

        #if len(self.weights) > 0:
        #    init_required = False

        #for i in range(0, len(self.layer_dim)):
        #    self.add_layer(num_neuron=self.layer_dim[i], activation=self.layer_activations[i])

        #    if i > 0 and init_required is True:
        #        self.weights.append(self.weights_init(size=(self.layer_dim[i - 1], self.layer_dim[i])))

        #if len(self.loss_history) > 0:
        #    self.flush_local_variables()

        self.reset_layers()

        l1 = (self.datapoint, self.layer_activations[0])
        self.layers[0] = l1

        for i in range(0, len(self.layer_dim) - 1):

            zx = self.step_forward(self.layers[i][0], self.weights[i])

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

        self.y_pred = self.layers[-1][0]

        l = self.loss.loss_func(y_pred=self.y_pred, y_true=y_true)
        tmp = np.sum(l)
        #self.loss_val.append(tmp)
        self.loss_history.append(tmp)

        if prediction==True:
            while len(self.layers) > 0:
                self.layers.pop()

        return self.y_pred

    def backward_pass(self, y_pred):
        weights_copy = self.weights.copy()
        num_steps = len(self.layers)

        dloss_dzx = None

        for i in range(0, num_steps):
            if i == 0:
                dloss_dzx = self.loss.dloss_dyi(y_pred=y_pred)
                self.dloss_dzx.append(dloss_dzx)
                self.bp_trace.append('Step {0} dloss_dzx {1}'.format(i, dloss_dzx))
                self.layers.pop()

            else:
                dloss_dzx = self.step_backward(idx=i, weights_copy=weights_copy, dloss_dzx=dloss_dzx)
                if dloss_dzx is not None:
                    self.layers.pop()

        # to match indexes with bp weights
        self.weights.reverse()

        self.magnitude_change = self.weights.copy()

        for i, (dwx, wx) in enumerate(zip(self.dloss_dwx, self.weights)):
            print(dwx)
            print()
            self.magnitude_change[i] = np.sqrt(np.sum(dwx)**2)
            self.weights[i] = wx - self.momentum*self.learning_rate*dwx

        # restore weights to appropriate layer indicies
        self.magnitude_change.reverse()
        self.weights.reverse()

    def step_forward(self, previous_layer, weights):
        return np.dot(weights.T, previous_layer)

    def step_backward(self, idx, weights_copy, dloss_dzx):


        """
        Calculate: dloss_dwx
        a: Aggregate value at current layer. Activated value of previous layer
            example: aggregate = weights * activation
        b: Change in loss with respect to the aggregate value at current layer.
            example: dloss_dzx = y_pred - y_true
        """
        current_layer = self.layers.pop()

        a = current_layer[0]
        b = dloss_dzx
        dloss_dwx = np.dot(a, b.T)

        self.dloss_dwx.append(dloss_dwx)
        #self.bp_trace['Step {0}'.format(idx)] = dict({'dloss_dwx': dloss_dwx})
        self.bp_trace.append('Step {0} dloss_dwx {1}'.format(idx, dloss_dwx))

        if len(self.layers) == 0:
            return None

        """
        Calculate: dloss_dax
        a: The weight we calculate a gradient for
            example: aggregate = weights * activation
        b: Change in loss with respect to the aggregate value at current layer.
            example: dloss_dzx = y_pred - y_true
        """

        a = weights_copy.pop()
        b = dloss_dzx
        dloss_dax = np.dot(a, b)

        self.dloss_dax.append(dloss_dax)

        #self.bp_trace['Step {0}'.format(idx)] = dict({'dloss_dax': dloss_dax})
        self.bp_trace.append('Step {0} dloss_dax {1}'.format(idx, dloss_dax))



        """
        Calculate: dloss_dzx
        a: Aggregate value at current layer.
            example: aggregate = weights * activation
        b: Derivative of activation function
            example: sigmoid(x)' = sigmoid(x) * (1 - sigmoid(x))   
        """

        next_layer = self.layers.pop()

        aggregate = current_layer[0]
        activation_type = next_layer[1]

        a = dloss_dax

        if activation_type == 'sigmoid':
            b = self.sigmoid_derivative(aggregate)
        elif activation_type == 'relu':
            b = self.relu_derivative(aggregate)
        else:
            b = 1

        dloss_dzx = a * b

        self.dloss_dzx.append(dloss_dzx)
        self.layers.append(next_layer)
        self.layers.append(current_layer)

        #self.bp_trace['Step {0}'.format(idx)] = dict({'dloss_dzx': dloss_dzx})
        self.bp_trace.append('Step {0} dloss_dzx {1}'.format(idx, dloss_dzx))


        return dloss_dzx

    def predict(self, sample, gth):
        y_pred = self.foward_pass(data_point=sample, y_true=gth, prediction=True)

        return np.argmax(y_pred)

    def fit(self, X, y, batch_size):
        X_chunks = [X[x:x + batch_size] for x in range(0, len(X), batch_size)]
        y_chunks = [y[x:x + batch_size] for x in range(0, len(y), batch_size)]

        for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):
            for (pt, label) in zip(X_chunk, y_chunk):
                self.foward_pass(data_point=pt, y_true=label)
                self.backward_pass(self.y_pred)
                pass
                #self.flush_local_variables()
                #print()

    def relu_derivative(self, mat):
        return np.maximum(mat, 0)

    def sigmoid_derivative(self, mat):
        return mat * (1 - mat)

    def dump_model_fp_summary(self):
        return self.fp_trace

    def dump_model_weight_summary(self):
        string = ''
        for i, weight in enumerate(self.weights):
            magnitude = np.sqrt(np.sum(weight)**2)
            string += 'weights.shape {0} magnitude_change: {1} Absolute Magnitude: {2}\n'.format(weight.shape,
                                                                                                 self.magnitude_change[i],
                                                                                                 magnitude)

        return string

    def flush_local_variables(self):
        self.dloss_dwx = []
        self.dloss_dax = []
        self.dloss_dzx = []
        #self.loss_history = []
        self.fp_trace = []

    def reset_layers(self):
        for i in range(0, len(self.layer_dim)):
            self.add_layer(num_neuron=self.layer_dim[i], activation=self.layer_activations[i])
