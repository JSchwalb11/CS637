import numpy as np
from Network import Network
from keras.datasets import mnist
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

def get_mnist_data(hot_encoded=True):
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
    #activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'softmax']
    activations = ['relu', 'sigmoid', 'relu', 'sigmoid', 'none']
    loss_type = 'categorical_crossentropy'
    #loss_type = 'hinge'
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

    # np.seterr(invalid='ignore')
    # warnings.filterwarnings('ignore')

    argmax_train_y = []
    for item in train_y:
        argmax_train_y.append(np.argmax(item))

    EPOCHS = 5
    BATCH_SIZE = 32

    X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
    y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

    lifetime_epoch_loss = []
    lifetime_epoch_acc = []

    for j in range(0, EPOCHS):
        print("Begin Epoch {0}".format(j))
        now = time.time()

        epoch_loss = []
        epoch_y_pred = []
        for (X_chunk, y_chunk) in zip(X_chunks, y_chunks):
            batch_loss = []
            for (pt, label) in zip(X_chunk, y_chunk):
                network.fit(inp=pt, y_true=label)
                y_pred = network.y_pred
                loss = np.sum(network.get_loss(y_pred, label))

                batch_loss.append(loss)
                epoch_y_pred.append(y_pred)
                epoch_loss.append(loss)

            plt.figure()
            t = np.arange(0, batch_loss)
            plt.plot(t, batch_loss)
            plt.title("Loss per Batch")
            plt.show()
        elapsed_time = time.time() - now
        print("Time taken to complete epoch: {0}".format(elapsed_time))



        start = time.time()

        sum_epoch_loss = np.sum(epoch_loss)
        #average_epoch_acc = np.average(epoch_acc)
        argmax_epoch_y_pred = []
        for item in epoch_y_pred:
            argmax_epoch_y_pred.append(np.argmax(item))

        scored_epoch_acc = accuracy_score(argmax_epoch_y_pred, argmax_train_y)

        lifetime_epoch_loss.append(sum_epoch_loss)
        lifetime_epoch_acc.append(scored_epoch_acc)

        end = time.time()
        print("Time taken to complete metrics: {0}".format(end-start))
        print("Current Epoch Loss: {0}".format(lifetime_epoch_loss[len(lifetime_epoch_loss) - 1]))
        print("Training Acc: {0}".format(lifetime_epoch_acc[len(lifetime_epoch_acc) - 1]))


    plt.figure()
    t = np.arange(0, EPOCHS)
    plt.plot(t, lifetime_epoch_loss)
    plt.title("Total Loss per Epoch")
    plt.show()

    plt.figure()
    t = np.arange(0, EPOCHS)
    plt.plot(t, lifetime_epoch_acc)
    plt.title("Training Accuracy per Epoch")
    plt.show()


