import numpy as np
from Layer import Layer


def test_layer1():
    l = Layer(in_shape=2, out_shape=3, activation='sigmoid', init_weight_type='diagonal')
    inp = np.asarray([1, 0])
    o = l.foward(inp)

    dot = np.dot(l.weights.T, inp) + l.bias

    a = 1 / (1 + np.exp(-inp))

    print("o = ", o)
    print(l.weights)
    for i, v in enumerate([0.73105858, 0.5, 0.5]):
        assert int(1000*v) == int(1000*o[i])
    #assert o.all() == [0.73105858, 0.5, 0.5]

    for i, v in enumerate([[1., 0., 0.], [0., 1., 0.]]):
        for j, v2 in enumerate(v):
            assert l.weights[i, j] == v2

