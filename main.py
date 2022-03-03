import numpy as np
from Network import Network
from keras.datasets import mnist
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

def get_mnist_data(hot_encoded=True, sublist=1000):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if hot_encoded == True:
        y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
        y_train[np.arange(train_y.shape[0]), train_y] = 1
        y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
        y_test[np.arange(test_y.shape[0]), test_y] = 1

        if sublist:
            train_X = train_X[:sublist]
            y_train = y_train[:sublist]
            test_X = test_X[:sublist]
            y_test = y_test[:sublist]

    return (train_X, y_train), (test_X, y_test)


if __name__ == '__main__':
    if True:
        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]
    else:
        train_X = np.asarray([[1,0,1,0], [1,0,1,1]])
        train_y = np.asarray([[1,0,0,0,0], [0,0,0,0,1]])

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

    EPOCHS = 5
    BATCH_SIZE = 8

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


