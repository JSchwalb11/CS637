import numpy as np
from Layer import Layer

class Network():
    def __init__(self, layers, *args, **kwargs):
        self.layers = layers

        self.input_ptr = None
        self.fp_trace = []
        self.bp_trace = []

    def compile(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.tail = self.layers[i + 1]
                self.input_ptr = np.zeros(layer.in_shape)
                layer.head = self.input_ptr
            elif i + 1 == len(self.layers):
                layer.tail = None
                layer.head = self.layers[i - 1]
            else:
                layer.tail = self.layers[i + 1]
                layer.head = self.layers[i - 1]

    def foward_pass(self, inp_ptr):
        current_layer = self.layers[0]
        current_layer.foward(inp_ptr)

        i = 0
        while current_layer.tail is not None:
            current_layer = current_layer.tail
            o = current_layer.foward(current_layer.head.output)
            i += 1

            self.fp_trace.append(("Step {0}".format(i), o))
        return current_layer.output

    def backward_pass(self, y_true):
        current_layer = self.layers[-1]

        i = 0
        while current_layer.head is not None:
            if str(type(current_layer.head)) is not "<class \'Layer.Layer\'":
                break

            current_layer.backward(y_true)
            o = current_layer.dzx
            current_layer = current_layer.head
            i += 1

            self.bp_trace.append(("Step {0}".format(i), o))

    def predict(self, inp, hot_encoded=False):
        y_pred = self.foward_pass(inp)

        if hot_encoded == True:
            hot_encoded = np.zeros_like(y_pred).astype('uint8')
            hot_encoded[np.argmax(y_pred)] = 1
            y_pred = hot_encoded

        return y_pred

    def dloss_dyi(self, y_pred, y_true):
        a = self.layers[-1].dloss_dyi(y_pred=y_pred, y_true=y_true)
        return a

    def get_loss(self, y_pred, y_true):
        a = self.layers[-1].loss(y_pred=y_pred, y_true=y_true)
        return a


    def compile_model(self, *args, **kwargs):
        self.in_shape = kwargs.get('in_shape')
        self.k_class = kwargs.get('k_class')
        self.loss_type = kwargs.get('loss_type')
        self.dims = kwargs.get('dims')
        self.activations = kwargs.get('activations')
        self.momentum = kwargs.get('momentum')
        self.learning_rate = kwargs.get('learning_rate')
        self.weight_type = kwargs.get('weight_type')

        l0 = Layer(in_shape=self.in_shape,
                   out_shape=self.in_shape,
                   activation='none',
                   init_weight_type='identity')

        ln = Layer(in_shape=self.k_class,
                   out_shape=self.k_class,
                   activation='none',
                   init_weight_type='identity',
                   loss_type=self.loss_type,
                   trainable=False)

        assert len(self.dims) == len(self.activations)

        for i in range(0, len(self.dims)):
            if i == 0:
                lx = Layer(in_shape=self.in_shape,
                           out_shape=self.dims[i],
                           activation=self.activations[i],
                           loss_type=self.loss_type,
                           momentum=self.momentum,
                           learning_rate=self.learning_rate,
                           init_weight_type=self.weight_type)
            else:
                lx = Layer(in_shape=self.dims[i - 1],
                           out_shape=self.dims[i],
                           activation=self.activations[i],
                           loss_type=self.loss_type,
                           momentum=self.momentum,
                           learning_rate=self.learning_rate,
                           init_weight_type=self.weight_type)

            self.layers.append(lx)
        network = Network(self.layers)
        network.compile()

        return network


