import numpy as np
from Layer import Layer
from Network import Network
import copy

def get_mnist_data(hot_encoded=True):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if hot_encoded == True:
        y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
        y_train[np.arange(train_y.shape[0]), train_y] = 1
        y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
        y_test[np.arange(test_y.shape[0]), test_y] = 1

    return (train_X, y_train), (test_X, y_test)

class TestDNN:
    def setup_method(self, test_method):
        # configure self.attribute
        self.l00 = Layer(in_shape=4,
                        out_shape=4,
                        activation='none',
                        init_weight_type='identity')

        self.l0 = Layer(in_shape=4,
                   out_shape=4,
                   activation='relu',
                   init_weight_type='diagonal')

        self.l1 = Layer(in_shape=4,
                   out_shape=4,
                   activation='sigmoid',
                   init_weight_type='diagonal')

        self.l2 = Layer(in_shape=4,
                   out_shape=4,
                   activation='none',
                   init_weight_type='identity',
                   loss_type='hinge',
                   trainable=False)
        #=======================================
        self.l001 = Layer(in_shape=4,
                         out_shape=4,
                         activation='none',
                         init_weight_type='identity')

        self.l01 = Layer(in_shape=4,
                        out_shape=4,
                        activation='relu',
                        init_weight_type='diagonal')

        self.l11 = Layer(in_shape=4,
                        out_shape=4,
                        activation='sigmoid',
                        init_weight_type='diagonal')

        self.l21 = Layer(in_shape=4,
                        out_shape=4,
                        activation='none',
                        init_weight_type='identity',
                        loss_type='hinge',
                        trainable=False)

    def test_Network_predict(self):
        inp = np.asarray([1, 0, 1, 0])

        network = Network([self.l00, self.l0, self.l1, self.l2])
        network.compile()

        oN = network.predict(inp)

        o00 = self.l00.foward(inp)
        o0 = self.l0.foward(o00)
        o1 = self.l1.foward(o0)
        o2 = self.l2.foward(o1)

        for i, v in enumerate(oN):
            assert v == o2[i]

    def test_Network_hinge_loss_delta(self):
        inp = np.asarray([1, 0, 1, 0])
        y_true = np.asarray([0,0,1,0])

        network = Network([self.l00, self.l0, self.l1, self.l2])
        network.compile()

        oN = network.predict(inp)

        o00 = self.l00.foward(inp)
        o0 = self.l0.foward(o00)
        o1 = self.l1.foward(o0)
        o2 = self.l2.foward(o1)

        loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)

        loss_true = np.asarray([1,1,-3,1]).astype(np.float64)

        for i, v in enumerate(loss_delta):
            assert v == loss_true[i]

    def test_Network_hinge_loss(self):
        inp = np.asarray([1, 0, 1, 0])
        y_true = np.asarray([0,0,1,0])

        network = Network([self.l00, self.l0, self.l1, self.l2])
        network.compile()

        oN = network.predict(inp)

        o00 = self.l00.foward(inp)
        o0 = self.l0.foward(o00)
        o1 = self.l1.foward(o0)
        o2 = self.l2.foward(o1)

        loss = network.get_loss(y_pred=oN, y_true=y_true)

        loss_true = 2.53788284273999

        assert loss == loss_true

    def test_Network_backward_pass(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l00, self.l0, self.l1, self.l2])
        network.compile()

        network1 = Network([self.l001, self.l01, self.l11, self.l21])
        network1.compile()

        #network1 = copy.deepcopy(network)

        oN = network.predict(inp)
        y_true = np.asarray([0,0,1,0])

        #loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)

        self.l2.backward(y_true)
        self.l1.backward(y_true)
        self.l0.backward(y_true)
        self.l00.backward(y_true)

        dw00 = self.l00.dwx
        dw0 = self.l0.dwx
        dw1 = self.l1.dwx
        dw2 = self.l2.dwx

        oN1 = network1.predict(inp)
        #y_true = np.asarray([1, 0, 0, 0, 0])

        network1.backward_pass(y_true=y_true)
        #for layer in network.layers
        for j, layer in enumerate(network1.layers):
            print("Layer{0} {1}".format(j, layer.dwx))

        for i, dwx in enumerate([dw00, dw0, dw1, dw2]):
            assert (dwx == network1.layers[i].dwx).all() == True

    def test_backward_x3_passes(self):
        from sklearn.model_selection import train_test_split

        def get_mnist_data(hot_encoded=True):
            from keras.datasets import mnist
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
            if hot_encoded == True:
                y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
                y_train[np.arange(train_y.shape[0]), train_y] = 1
                y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
                y_test[np.arange(test_y.shape[0]), test_y] = 1

            # return (train_X.T, y_train), (test_X.T, y_test)
            return (train_X, y_train), (test_X, y_test)

        (train_X, train_y), (val_X, val_y) = get_mnist_data()
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
        val_X = val_X.reshape((val_X.shape[0], val_X.shape[1] * val_X.shape[2]))

        train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42,
                                                            shuffle=True)

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]

        train_X = train_X[:3, :]
        dims = [in_shape, 128, k_class]
        weight_type = 'diagonal'
        activation = ['relu', 'sigmoid', 'none']
        loss_type = 'hinge'
        lr = 0.01

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activation,
                              loss_type=loss_type,
                              momentum=0.9,
                              learning_rate=lr,
                              weight_type=weight_type)

        dw0 = []
        dw1 = []
        dw2 = []

        starting_weights = []
        for layer in network.layers:
            starting_weights.append(layer.weights)

        y_pred0 = network.foward_pass(train_X[0])
        network.backward_pass(train_y[0])
        for layer in network.layers:
            dw0.append(layer.weights[:1])

        """y_pred01 = network.foward_pass(train_X[0])
        network.backward_pass(train_y[0])
        for layer in network.layers:
            dw0.append(layer.weights[:1])"""

        y_pred1 = network.foward_pass(train_X[1])
        network.backward_pass(train_y[1])
        for layer in network.layers:
            dw1.append(layer.weights[:1])

        y_pred2 = network.foward_pass(train_X[2])
        network.backward_pass(train_y[2])
        for layer in network.layers:
            dw2.append(layer.weights[:1])

        breakpoint()

if __name__ == '__main__':
    TestDNN.setup_method(None)
