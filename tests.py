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

        sublist = 1000

        train_X = train_X[:sublist]
        y_train = y_train[:sublist]
        test_X = test_X[:sublist]
        y_test = y_test[:sublist]

    return (train_X, y_train), (test_X, y_test)

class TestDNN:
    def setup_method(self, test_method):
        # configure self.attribute
        self.l00 = Layer(in_shape=4,
                        out_shape=4,
                        activation='none',
                        init_weight_type='identity')

        self.l0 = Layer(in_shape=4,
                   out_shape=3,
                   activation='relu',
                   init_weight_type='diagonal')

        self.l1 = Layer(in_shape=3,
                   out_shape=2,
                   activation='sigmoid',
                   init_weight_type='diagonal')

        self.l2 = Layer(in_shape=2,
                   out_shape=5,
                   activation='softmax',
                   init_weight_type='diagonal')

        self.l3 = Layer(in_shape=5,
                   out_shape=5,
                   activation='none',
                   init_weight_type='identity',
                   loss_type='hinge',
                   trainable=False)
        #############################################
        self.l001 = Layer(in_shape=4,
                          out_shape=4,
                          activation='none',
                          init_weight_type='identity')

        self.l01 = Layer(in_shape=4,
                         out_shape=3,
                         activation='relu',
                         init_weight_type='diagonal')

        self.l11 = Layer(in_shape=3,
                         out_shape=2,
                         activation='sigmoid',
                         init_weight_type='diagonal')

        self.l21 = Layer(in_shape=2,
                         out_shape=6,
                         activation='relu',
                         init_weight_type='diagonal')

        self.l31 = Layer(in_shape=6,
                         out_shape=8,
                         activation='sigmoid',
                         init_weight_type='diagonal')

        self.l41 = Layer(in_shape=8,
                         out_shape=5,
                         activation='softmax',
                         init_weight_type='diagonal')

        self.l51 = Layer(in_shape=5,
                         out_shape=5,
                         activation='none',
                         init_weight_type='identity',
                         loss_type='hinge',
                         trainable=False)

    """def test_gaussian_init(self):
        variance = 1.1
        l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='gaussian var=1.1')
        assert variance == l.var"""

    def test_sigmoid(self):
        l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='diagonal')
        inp = np.asarray([1, 0])
        o = l.foward(inp)

        dot = np.dot(l.weights.T, inp) + l.bias

        a = 1 / (1 + np.exp(-inp))

        for i, v in enumerate([0.73105858, 0.5, 0.5]):
            assert int(1000*v) == int(1000*o[i])
        #assert o.all() == [0.73105858, 0.5, 0.5]

        for i, v in enumerate([[1., 0., 0.], [0., 1., 0.]]):
            for j, v2 in enumerate(v):
                assert l.weights[i, j] == v2

    def test_sigmoid_derivative(self):
        l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='diagonal')
        inp = np.asarray([1, 0])

        a = 1 / (1 + np.exp(-inp))

        o = a * (1 - a)

        o1 = l.activation_dict[l.activation]['derivative'](inp)

        for i, v in enumerate(o):
            assert int(1000 * v) == int(1000 * o1[i])
        # assert o.all() == [0.73105858, 0.5, 0.5]

        for i, v in enumerate([[1., 0., 0.], [0., 1., 0.]]):
            for j, v2 in enumerate(v):
                assert l.weights[i, j] == v2

    def test_layer_attachment(self):
        l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='diagonal')
        l1 = Layer(in_shape=3, out_shape=4, activation='sigmoid', init_weight_type='diagonal')

        inp = np.asarray([1, 0])
        o = l.foward(inp)

        o1 = l1.foward(o)

        #print("o1 = ", o1)

        for i, v in enumerate([0.67503753, 0.62245933, 0.62245933, 0.5]):
            assert int(1000*v) == int(1000*o1[i])

    def test_relu_derivative(self):
        l = Layer(in_shape=2, out_shape=3, activation='relu', init_weight_type='diagonal')

        inp = np.asarray([1, 0])

        o = (inp > 0) * 1.0
        o1 = l.activation_dict[l.activation]['derivative'](inp)

        for i, v in enumerate(o):
            assert v == o1[i]

    def test_relu(self):
        l = Layer(in_shape=2, out_shape=3, activation='relu', init_weight_type='diagonal')

        inp = np.asarray([1, 0])
        o = l.foward(inp)

        for i, v in enumerate([1., 0., 0.]):
            assert int(1000*v) == int(1000*o[i])

    def test_softmax(self):
        l = Layer(in_shape=2, out_shape=2, activation='softmax', init_weight_type='diagonal')

        inp = np.asarray([0.9, 0.1])

        o = l.foward(inp)

        #print("o = ", o)

        for i, v in enumerate([0.68997448, 0.31002552]):
            assert int(1000*v) == int(1000*o[i])

    def test_prediction(self):
        l = Layer(in_shape=2,
                  out_shape=2,
                  activation='softmax',
                  init_weight_type='diagonal',
                  loss_type='categorical_crossentropy')

        inp = np.asarray([0.9, 0.1])
        y_pred = l.foward(inp)

        for i, v in enumerate([0.68997448, 0.31002552]):
            assert int(1000*v) == int(1000*y_pred[i])

        hot_encoded = np.zeros_like(y_pred).astype('uint8')
        hot_encoded[np.argmax(y_pred)] = 1

        #print("hot_encoded = ", hot_encoded)

        y_true = [1, 0]
        for i, v in enumerate([1, 0]):
            assert v == y_true[i]

    def test_loss_correct_categorical_crossentropy_prediction(self):
        l = Layer(in_shape=2,
                  out_shape=2,
                  activation='softmax',
                  init_weight_type='diagonal',
                  loss_type='categorical_crossentropy')

        inp = np.asarray([0.9, 0.1])
        y_pred = l.foward(inp)
        y_true = [1, 0]

        loss = l.loss(y_pred=y_pred, y_true=y_true)
        #print("loss = ", loss)

        assert loss == 0.18554960830993195

    def test_loss_categorical_crossentropy_derivative(self):
        l = Layer(in_shape=2,
                  out_shape=2,
                  activation='softmax',
                  init_weight_type='diagonal',
                  loss_type='categorical_crossentropy')

        inp = np.asarray([0.9, 0.1])
        y_pred = l.foward(inp)
        y_true = np.asarray([1,0])

        loss_delta = y_pred - y_true

        dloss_dyi = l.dloss_dyi(y_pred=y_pred, y_true=y_true)
        #print(dloss_dyi)

        for i, v in enumerate(loss_delta):
            assert v == dloss_dyi[i]

    def test_loss_correct_hinge_prediction(self):
        l = Layer(in_shape=3,
                  out_shape=3,
                  activation='none',
                  init_weight_type='diagonal',
                  loss_type='hinge')

        inp = np.asarray([0.7, 0.1, 0.2])
        y_pred = l.foward(inp)
        y_true = [1, 0, 0]

        loss = l.loss(y_pred=y_pred, y_true=y_true)
        #print(loss)

        assert loss == 1.5

    def test_loss_hinge_derivative(self):
        l = Layer(in_shape=3,
                  out_shape=3,
                  activation='none',
                  init_weight_type='diagonal',
                  loss_type='hinge')

        inp = np.asarray([0.6, 0.3, 0.1])

        y_pred = l.foward(inp)
        y_true = np.asarray([1, 0, 0])

        loss_delta = l.dloss_dyi(y_pred=y_pred, y_true=y_true)

        for i, v in enumerate([-0.3333333333333333,  0., 0.]):
            assert v == loss_delta[i]

    def test_Network_init(self):
        raw_network = [self.l0, self.l1, self.l2, self.l3]

        network = Network(raw_network)

        assert len(raw_network) == len(network.layers)


    def test_Network_compile(self):
        raw_network = [self.l0, self.l1, self.l2, self.l3]

        network = Network(raw_network)
        network.compile()

        for i, ptr in enumerate(raw_network):
            assert ptr == network.layers[i]

    def test_Network_predict(self):
        inp = np.asarray([1, 0, 1, 0])

        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()

        oN = network.predict(inp)

        o0 = self.l0.foward(inp)
        o1 = self.l1.foward(o0)
        o2 = self.l2.foward(o1)
        o3 = self.l3.foward(o2)

        for i, v in enumerate(oN):
            assert v == o3[i]

    def test_Network_gradient(self):
        inp = np.asarray([1, 0, 1, 0])

        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()

        y_pred = network.predict(inp)

        y_true = np.asarray([1, 0, 0, 0, 0])

        loss_delta = self.l3.dloss_dyi(y_pred=y_pred, y_true=y_true)

        loss_delta1 = network.dloss_dyi(y_pred=y_pred, y_true=y_true)

        for i, v in enumerate(loss_delta):
            assert v == loss_delta1[i]

    def test_prediction_not_zero(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()
        #breakpoint()
        oN = network.predict(inp)
        assert oN.all() > 0

    def test_layer_backward_dwx(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()

        oN = network.predict(inp)
        y_true = np.asarray([1, 0, 0, 0, 0])

        loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)
        self.l2.backward(y_true)

        dwx = self.l2.dwx

        assert (dwx!=0).any() == True

    def test_layer_backward(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()

        oN = network.predict(inp)

        y_true = np.asarray([1, 0, 0, 0, 0])

        loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)

        before_update = self.l2.weights.copy()

        self.l2.backward(y_true)

        after_update = self.l2.weights
        dwx = self.l2.dwx

        for i, v in enumerate(before_update):
            for j, z in enumerate(v):
                a = after_update[i, j] + self.l2.momentum * self.l2.learning_rate * dwx[i, j]
                assert z == a

    def test_not_trainable(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l0, self.l1, self.l2, self.l3])
        network.compile()

        oN = network.predict(inp)
        y_true = np.asarray([1, 0, 0, 0, 0])

        loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)
        self.l3.backward(y_true)

        dw3 = self.l3.dwx

        assert (dw3 == 0).all() == True

    def test_Network_backward_pass(self):
        inp = np.asarray([1, 0, 1, 0])
        network = Network([self.l00, self.l0, self.l1, self.l2, self.l3])
        network.compile()

        oN = network.predict(inp)
        y_true = np.asarray([1, 0, 0, 0, 0])

        loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)

        self.l2.backward(y_true)
        self.l1.backward(y_true)
        self.l0.backward(y_true)


        dw0 = self.l0.dwx
        dw1 = self.l1.dwx
        dw2 = self.l2.dwx

        assert (dw2 != 0).any() == True
        assert (dw1 != 0).any() == True
        assert (dw0 != 0).any() == True

        def test_Network_backward_pass(self):
            inp = np.asarray([1, 0, 1, 0])
            network = Network([self.l00, self.l0, self.l1, self.l2, self.l3])
            network.compile()

            oN = network.predict(inp)
            y_true = np.asarray([1, 0, 0, 0, 0])

            #loss_delta = network.dloss_dyi(y_pred=oN, y_true=y_true)

            self.l2.backward(y_true)
            self.l1.backward(y_true)
            self.l0.backward(y_true)

            dw0 = self.l0.dwx
            dw1 = self.l1.dwx
            dw2 = self.l2.dwx

            oN1 = network.predict(inp)
            y_true = np.asarray([1, 0, 0, 0, 0])

            network.backward_pass(y_true=y_true)
            dw01 = self.l0.dwx
            dw11 = self.l1.dwx
            dw21 = self.l2.dwx

            assert (dw0 == dw01).all() == True
            assert (dw1 == dw11).all() == True
            assert (dw2 == dw21).all() == True



    def test_Network_backward_pass_extended(self):
        # inp = np.asarray([1, 0, 1, 0])
        x = np.asarray([[1, 0, 1, 0], [1, 0, 1, 1]])
        y = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        network0 = Network([self.l001, self.l01, self.l11, self.l21, self.l31, self.l41, self.l51])
        network1 = copy.deepcopy(network0)
        network0.compile()

        for i, inp in enumerate(x):
            network0.fit(inp, y[i])

            y_true = y[i]
            self.l41.backward(y_true)
            self.l31.backward(y_true)
            self.l21.backward(y_true)
            self.l11.backward(y_true)
            self.l01.backward(y_true)

            dw0 = self.l01.dwx
            dw1 = self.l11.dwx
            dw2 = self.l21.dwx
            dw3 = self.l31.dwx
            dw4 = self.l41.dwx

            oN1 = network1.predict(inp)
            network1.backward_pass(y_true=y_true)

            dw01 = self.l01.dwx
            dw11 = self.l11.dwx
            dw21 = self.l21.dwx
            dw31 = self.l31.dwx
            dw41 = self.l41.dwx

            assert (dw0 == dw01).all() == True
            assert (dw1 == dw11).all() == True
            assert (dw2 == dw21).all() == True
            assert (dw3 == dw31).all() == True
            assert (dw4 == dw41).all() == True

    def test_network_compile_model(self):
        x = np.asarray([[1, 0, 1, 0], [1, 0, 1, 1]])
        y = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        x = x.reshape(x.shape[0], x.shape[1], 1)

        in_shape = x.shape[1]
        k_class = y.shape[1]

        dims = [4, 3, 2, 6, 8]
        dims_extended = [4, 4, 3, 2, 6, 8, 8]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'softmax']
        loss_type = 'categorical_crossentropy'
        weight_type = 'gaussian'

        network = Network(layers=[])

        network.compile_model(in_shape = in_shape,
                                k_class = k_class,
                                dims = dims,
                                activations = activations,
                                loss_type = loss_type,
                                momentum = 1,
                                learning_rate = 0.001,
                                weight_type = weight_type)

        assert network != None


    def test_network_fit(self):
        x = np.asarray([[1, 0, 1, 0], [1, 0, 1, 1]])
        y = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        x = x.reshape(x.shape[0], x.shape[1])

        in_shape = x.shape[1]
        k_class = y.shape[1]

        dims = [4, 3, 2, 6, 5]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'softmax']
        loss_type = 'categorical_crossentropy'
        weight_type = 'diagonal'

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activations,
                              loss_type=loss_type,
                              momentum=1,
                              learning_rate=0.001,
                              weight_type=weight_type)

        epoch_loss = []

        for i in range(0, len(x)):
            network.fit(inp=x[i], y_true=y[i])
            loss = np.sum(network.get_loss(network.y_pred, y[i]))
            epoch_loss.append(loss)

        #print(epoch_loss)

        for i, v in enumerate([0.2993450529218126, 0.33443350891870804]):
            assert v == epoch_loss[i]


    def test_mnist_backpropogation_crossentropy_softmax(self):
        import time
        from matplotlib import pyplot as plt

        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]

        dims = [in_shape, 3, 2, 6, k_class]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'softmax']
        loss_type = 'categorical_crossentropy'
        # loss_type = 'hinge'
        weight_type = 'diagonal'

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activations,
                              loss_type=loss_type,
                              momentum=1,
                              learning_rate=0.01,
                              weight_type=weight_type)

        EPOCHS = 50
        BATCH_SIZE = 32

        X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
        y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

        lifetime_sum_losses = []

        for j in range(0, EPOCHS):
            print("Begin Epoch {0}".format(j))
            now = time.time()

            losses = []
            for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):

                for (pt, label) in zip(X_chunk, y_chunk):
                    network.fit(inp=pt, y_true=label)
                    y_pred = network.y_pred
                    loss = network.get_loss(y_pred, label)
                    v = losses[-3:] + [loss]
                    losses.append(np.mean(v))

                """plt.figure()
                t = np.arange(0, len(losses))
                plt.plot(t, losses)
                plt.title("Batch Loss")
                plt.show()"""
            elapsed_time = time.time() - now
            print("Time taken to complete epoch: {0}".format(elapsed_time))

            lifetime_sum_losses.append(np.sum(losses))
            if len(lifetime_sum_losses) > 1:
                loss_delta = lifetime_sum_losses[-2] - lifetime_sum_losses[-1]
                print("Loss Delta {0}\n".format(loss_delta))

                if np.isclose(loss_delta, 0):
                    print("Loss converged in {0} Epochs".format(j))
                    print("Loss = {0}".format(lifetime_sum_losses[-1]))
                    break


        plt.figure()
        t = np.arange(0, len(lifetime_sum_losses))
        plt.plot(t, lifetime_sum_losses)
        plt.title("Epoch Loss")
        plt.show()

    def test_mnist_backpropogation_hinge(self):
        import time
        from matplotlib import pyplot as plt

        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]

        dims = [in_shape, 3, 2, 6, k_class]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'none']
        #loss_type = 'categorical_crossentropy'
        loss_type = 'hinge'
        weight_type = 'diagonal'

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activations,
                              loss_type=loss_type,
                              momentum=1,
                              learning_rate=0.01,
                              weight_type=weight_type)

        EPOCHS = 50
        BATCH_SIZE = 32

        X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
        y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

        lifetime_sum_losses = []

        for j in range(0, EPOCHS):
            print("Begin Epoch {0}".format(j))
            now = time.time()

            losses = []
            for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):

                for (pt, label) in zip(X_chunk, y_chunk):
                    network.fit(inp=pt, y_true=label)
                    y_pred = network.y_pred
                    loss = network.get_loss(y_pred, label)
                    losses.append(loss)

                """plt.figure()
                t = np.arange(0, len(losses))
                plt.plot(t, losses)
                plt.title("Batch Loss")
                plt.show()"""
            elapsed_time = time.time() - now
            print("Time taken to complete epoch: {0}".format(elapsed_time))

            lifetime_sum_losses.append(np.sum(losses))
            if len(lifetime_sum_losses) > 1:
                loss_delta = lifetime_sum_losses[-2] - lifetime_sum_losses[-1]
                print("Loss Delta {0}\n".format(loss_delta))

                #if np.isclose(loss_delta, 0):
                if loss_delta == 0:
                    print("Loss converged in {0} Epochs".format(j))
                    print("Loss = {0}".format(lifetime_sum_losses[-1]))
                    break

    def test_mnist_fit_batch_backpropogation_crossentropy_softmax(self):
        import time
        from matplotlib import pyplot as plt

        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]

        dims = [in_shape, 3, 2, 6, k_class]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'softmax']
        loss_type = 'categorical_crossentropy'
        # loss_type = 'hinge'
        weight_type = 'diagonal'

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activations,
                              loss_type=loss_type,
                              momentum=1,
                              learning_rate=0.001,
                              weight_type=weight_type)

        EPOCHS = 50
        BATCH_SIZE = 32

        X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
        y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

        lifetime_sum_losses = []

        for j in range(0, EPOCHS):
            print("Begin Epoch {0}".format(j))
            now = time.time()

            losses = []
            for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):
                # for (pt, label) in zip(X_chunk, y_chunk):
                network.fit_batch(batch_x=X_chunk, batch_y=y_chunk)
                losses.append(network.loss)

                """plt.figure()
                t = np.arange(0, len(losses))
                plt.plot(t, losses)
                plt.title("Batch Loss")
                plt.show()"""
            elapsed_time = time.time() - now
            print("Time taken to complete epoch: {0}".format(elapsed_time))

            lifetime_sum_losses.append(np.sum(losses))
            if len(lifetime_sum_losses) > 1:
                loss_delta = np.average(lifetime_sum_losses[-3:]) - lifetime_sum_losses[-1]
                print("Loss Delta {0}\n".format(loss_delta))

                if loss_delta < 1e-4:
                    print("Loss converged in {0} Epochs".format(j))
                    print("Loss = {0}".format(lifetime_sum_losses[-1]))
                    break

        plt.figure()
        t = np.arange(0, len(lifetime_sum_losses))
        plt.plot(t, lifetime_sum_losses)
        plt.title("Epoch Loss")
        plt.show()

    def test_mnist_fit_batch_backpropogation_hinge(self):
        import time
        from matplotlib import pyplot as plt

        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]

        dims = [in_shape, 3, 2, 6, k_class]
        activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'none']
        # loss_type = 'categorical_crossentropy'
        loss_type = 'hinge'
        weight_type = 'diagonal'

        network = Network(layers=[])

        network.compile_model(in_shape=in_shape,
                              k_class=k_class,
                              dims=dims,
                              activations=activations,
                              loss_type=loss_type,
                              momentum=1,
                              learning_rate=0.01,
                              weight_type=weight_type)

        EPOCHS = 50
        BATCH_SIZE = 32

        X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
        y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

        lifetime_sum_losses = []

        for j in range(0, EPOCHS):
            print("Begin Epoch {0}".format(j))
            now = time.time()

            losses = []
            for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):
                # for (pt, label) in zip(X_chunk, y_chunk):
                network.fit_batch(batch_x=X_chunk, batch_y=y_chunk)
                losses.append(network.loss)

                """plt.figure()
                t = np.arange(0, len(losses))
                plt.plot(t, losses)
                plt.title("Batch Loss")
                plt.show()"""
            elapsed_time = time.time() - now
            print("Time taken to complete epoch: {0}".format(elapsed_time))

            lifetime_sum_losses.append(np.sum(losses))
            if len(lifetime_sum_losses) > 1:
                loss_delta = np.average(lifetime_sum_losses[-3:]) - lifetime_sum_losses[-1]
                print("Loss Delta {0}\n".format(loss_delta))

                if loss_delta < 1e-4:
                    print("Loss converged in {0} Epochs".format(j))
                    print("Loss = {0}".format(lifetime_sum_losses[-1]))
                    break

        plt.figure()
        t = np.arange(0, len(lifetime_sum_losses))
        plt.plot(t, lifetime_sum_losses)
        plt.title("Epoch Loss")
        plt.show()

if __name__ == '__main__':
    TestDNN.setup_method(None)
