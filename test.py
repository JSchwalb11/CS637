import numpy as np
from model import model

if __name__ == '__main__':
    x = np.asarray([[1, 0, 1, 0], [1, 0, 1, 1]])
    y = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

    x = x.reshape(x.shape[0], x.shape[1], 1)

    in_shape = x.shape[1]
    k_class = y.shape[1]

    train_X = x
    train_y = y

    layer_dim = [in_shape, 3, 6, k_class]
    layer_activation = ['relu', 'sigmoid', 'softmax', '']

    weight_initialization = 'gaussian var=2'
    loss = 'categorical_crossentropy'


    model = model(layer_dim=layer_dim,
                  layer_activation=layer_activation,
                  loss_type=loss,
                  loss_params=[],
                  learning_rate=0.005,
                  momentum=1,
                  weight_initialization=weight_initialization,
                  )

    for j in range(0, 2):
        for i in range(0, 2):
            #print(model.foward_pass(data_point=train_X[i], y_true=train_y[i]))
            model.foward_pass(data_point=train_X[i], y_true=train_y[i])

            print("Iter {0} Example {1} Loss: {2}".format(j, i, model.loss_history[-1]))
            #model.backward_pass(model.y_pred)
            #print(model.weights)
            y_pred = model.y_pred
            model.foward_pass(data_point=train_X[i], y_true=y_pred)
            print("Iter {0} Example {1} Loss: {2}".format(j, i, model.loss_history[-1]))
            #print(model.layers)

            print()


