import numpy as np
from Layer import Layer

class Network():
    def __init__(self, layers, *args, **kwargs):
        self.layers = layers
        self.y_pred = None
        self.loss = None

        self.input_ptr = None
        self.fp_trace = []
        self.bp_trace = []

    def compile(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.tail = self.layers[i + 1]
                #self.input_ptr = np.zeros(layer.in_shape)
                #layer.head = self.input_ptr
                layer.head = None
            elif i + 1 == len(self.layers):
                layer.tail = None
                layer.head = self.layers[i - 1]
            else:
                layer.tail = self.layers[i + 1]
                layer.head = self.layers[i - 1]

    def foward_pass(self, inp_ptr):
        current_layer = self.layers[0]
        #current_layer.output=
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
        while True:
            """if str(type(current_layer.head)) != "<class \'Layer.Layer\'>":
                break"""

            current_layer.backward(y_true)
            o = current_layer.dzx
            current_layer = current_layer.head
            i += 1

            self.bp_trace.append(("Step {0}".format(i), o))
            if current_layer is None:
                break
        #print("i=",i)

    def fit(self, inp, y_true):
        self.foward_pass(inp_ptr=inp)
        self.y_pred = self.predict(inp)  # self.layers[-1].output
        self.backward_pass(y_true)
        self.loss = self.get_loss(self.y_pred, y_true)

    def fit_batch(self, batch_x, batch_y):
        predictions = []
        batch_losses = []

        for i, inp in enumerate(batch_x):
            inp = inp.T
            predictions.append(self.foward_pass(inp))
            self.y_pred = self.predict(inp)
            batch_losses.append(self.get_loss(self.y_pred, batch_y[i]))

        self.loss = np.average(batch_losses)
        self.backward_pass(batch_y[0])

        return predictions

    def predict(self, inp, hot_encoded=False):
        self.y_pred = self.foward_pass(inp)
        return self.y_pred

    def predict_batch(self, batch_x, batch_y):
        predictions = []
        batch_losses = []

        for i, inp in enumerate(batch_x):
            self.y_pred = self.predict(inp)
            predictions.append(self.y_pred)  # self.layers[-1].output
            batch_losses.append(self.get_loss(self.y_pred, batch_y[i]))

        return predictions

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

        assert len(self.dims) == len(self.activations)

        first_layer = Layer(in_shape=self.dims[0],
                       out_shape=self.dims[0],
                       activation='none',
                       momentum=self.momentum,
                       learning_rate=self.learning_rate,
                       init_weight_type='identity',
                       trainable=False)
        self.layers.append(first_layer)

        for i in range(1, len(self.dims)):
            lx = Layer(in_shape=self.dims[i - 1],
                       out_shape=self.dims[i],
                       activation=self.activations[i],
                       momentum=self.momentum,
                       learning_rate=self.learning_rate,
                       init_weight_type=self.weight_type)
            self.layers.append(lx)

        final_layer = Layer(in_shape=self.dims[-1],
                            out_shape=self.dims[-1],
                            activation='none',
                            loss_type=self.loss_type,
                            momentum=self.momentum,
                            learning_rate=self.learning_rate,
                            init_weight_type=self.weight_type,
                            trainable=False)

        self.layers.append(final_layer)

        network = Network(self.layers)
        network.compile()

        return network


