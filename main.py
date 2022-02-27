import sys
import numpy as np
from model import model

def softmax(mat):
    if mat.ndim > 1:
        print("Error in Relu input matrix, too many columns.")
        sys.exit(-1)

    a = np.zeros_like(mat)
    sum_si = np.sum(np.exp(mat))

    for i in range(0, mat.shape[0]):
        si = np.exp(mat[i])
        a[i] = si/sum_si

    return a

def sigmoid(mat):
    if mat.ndim > 1:
        print("Error in Relu input matrix, too many columns.")
        sys.exit(-1)

    a = 1/(1+np.exp(-mat))

    return a

def relu(mat):
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

def init_weights(size):
    weights = np.zeros(shape=size)
    min_dim = min(weights.shape[0], weights.shape[1])
    for i in range(0, min_dim):
        weights[i,i] = 1

    return weights

if __name__ == '__main__':

    x = np.asarray([1,0,1,0])
    x = x.reshape(x.shape[0], 1)

    """
    w1_shape = (4,3)
    w2_shape = (3,2)
    w3_shape = (2,5)
    b1_shape = (3,1)
    b2_shape = (2,1)

    w1 = init_weights(w1_shape)
    w2 = init_weights(w2_shape)
    w3 = init_weights(w3_shape)

    b1 = np.zeros(b1_shape)
    b2 = np.zeros(b2_shape)

    # foward pass
    z1 = np.dot(w1.T, x)
    a1 = relu(z1)
    z2 = np.dot(w2.T, a1)
    a2 = sigmoid(z2)
    z3 = np.dot(w3.T, a2)
    yhat = softmax(z3)
    """

    y_gth = np.asarray([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]])
    layer_dim = [4, 3, 2, 5]  # including input and output layer
    layer_activation = ['relu', 'sigmoid', 'softmax', '']
    #loss = 'categorical_crossentropy'
    loss = 'hinge'
    loss_params = ['j=2']
    model = model(layer_dim=layer_dim, layer_activation=layer_activation, loss_type=loss, loss_params=[])
    model.foward_pass(data_point=x, y_true=y_gth[0])
    model.backward_pass(model.y_pred)
    for y_true in y_gth:
        loss = model.categorical_crossentropy(y_pred=model.y_pred, y_true=y_true)
        print("Single Sample Loss")
        print("y_true: {0}, y_pred: {1}, loss: {2}".format(y_true, model.y_pred, loss))

    print("Model Summary")
    print(model.dump_model_summary())


