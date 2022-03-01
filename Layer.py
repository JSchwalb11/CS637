import numpy as np

class Layer:
    def __init__(self, in_shape, out_shape, activation, init_weight_type):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activation = activation
        self.init_activations()
        self.bias = np.zeros(out_shape)
        self.init_weights()
        self.weight_dict[init_weight_type]()
        self.output = np.zeros(out_shape)

    def foward(self, inputs):
        a = np.dot(self.weights.T, inputs) + self.bias
        self.output = self.activation_dict[self.activation]['func'](a)
        return self.output

    def sigmoid_derivative(self, mat):
        return mat * (1 - mat)

    def sigmoid(self, mat):
        a = 1 / (1 + np.exp(-mat))
        return a

    def weight_init_random(self):
        weights = np.random.rand((self.in_shape, self.out_shape))
        self.weights = weights

    def weight_init_gaussian(self):
        weights = np.random.normal(0, 0.1, size=(self.in_shape, self.out_shape))
        self.weights = weights

    def main_diag_init(self):
        weights = np.zeros(shape=(self.in_shape, self.out_shape))
        min_dim = min(weights.shape[0], weights.shape[1])
        for i in range(0, min_dim):
            weights[i, i] = 1

        self.weights = weights

    def init_activations(self):
        self.activation_dict = {'sigmoid': {'func': self.sigmoid, 'derivative': self.sigmoid_derivative}}

    def init_weights(self):
        self.weight_dict = {'random': self.weight_init_random,
                            'gaussian': self.weight_init_gaussian,
                            'diagonal': self.main_diag_init}





if __name__ == '__main__':

    l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='diagonal')
    inp = np.asarray([1,0])
    o = l.foward(inp)

    dot = np.dot(l.weights.T, inp) + l.bias

    a = 1 / (1 + np.exp(-inp))

    print(o==a)

    #assert o == a





