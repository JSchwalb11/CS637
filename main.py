import numpy as np
from Network import Network
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

def get_mnist_data(hot_encoded=True):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if hot_encoded == True:
        y_train = np.zeros((train_y.shape[0], train_y.max() + 1), dtype=np.float32)
        y_train[np.arange(train_y.shape[0]), train_y] = 1
        y_test = np.zeros((test_y.shape[0], test_y.max() + 1), dtype=np.float32)
        y_test[np.arange(test_y.shape[0]), test_y] = 1

    #return (train_X.T, y_train), (test_X.T, y_test)
    return (train_X, y_train), (test_X, y_test)

def batch_hot_encode_to_argmax(batch):
    predictions = []

    for encoded in batch:
        predictions.append(np.argmax(encoded))

    return predictions


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    (train_X, train_y), (val_X, val_y) = get_mnist_data()
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    val_X = val_X.reshape((val_X.shape[0], val_X.shape[1] * val_X.shape[2]))

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42, shuffle=True)

    in_shape = train_X.shape[1]
    k_class = train_y.shape[1]

    dims = [in_shape, 3, 7, 110, k_class]
    weight_type = 'gaussian'
    losses = ['hinge']
    activations = [['relu', 'sigmoid', 'relu', 'sigmoid', 'none']]
    lrs = [0.01]
    network = Network(layers=[])

    network.compile_model(in_shape=in_shape,
                          k_class=k_class,
                          dims=dims,
                          activations=activations[0],
                          loss_type=losses[0],
                          momentum=1,
                          learning_rate=lrs[0],
                          weight_type=weight_type)

    EPOCHS = 30
    BATCH_SIZE = 64

    train_X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
    train_y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

    test_X_chunks = [test_X[x:x + BATCH_SIZE] for x in range(0, len(test_X), BATCH_SIZE)]
    test_y_chunks = [test_y[x:x + BATCH_SIZE] for x in range(0, len(test_y), BATCH_SIZE)]

    val_X_chunks = [val_X[x:x + BATCH_SIZE] for x in range(0, len(val_X), BATCH_SIZE)]
    val_y_chunks = [val_y[x:x + BATCH_SIZE] for x in range(0, len(val_y), BATCH_SIZE)]

    lifetime_sum_train_losses = []
    lifetime_sum_train_acc = []

    lifetime_sum_test_losses = []
    lifetime_sum_test_acc = []

    lifetime_sum_val_losses = []
    lifetime_sum_val_acc = []

    for j in range(0, EPOCHS):
        print("Begin Epoch {0}".format(j))
        now = time.time()

        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        val_acc = []
        val_losses = []

        for (X_chunk, y_chunk) in zip(train_X_chunks, train_y_chunks):
            predictions = network.fit_batch(batch_x=X_chunk, batch_y=y_chunk)
            y_pred = batch_hot_encode_to_argmax(predictions)
            y_true = batch_hot_encode_to_argmax(y_chunk)

            train_acc.append(accuracy_score(y_pred, y_true))
            train_losses.append(network.loss)

        for (X_chunk, y_chunk) in zip(test_X_chunks, test_y_chunks):
            #predictions = network.predict_batch(batch_x=X_chunk, batch_y=y_chunk)
            predictions = network.fit_batch(batch_x=X_chunk, batch_y=y_chunk)
            y_pred = batch_hot_encode_to_argmax(predictions)
            y_true = batch_hot_encode_to_argmax(y_chunk)

            test_acc.append(accuracy_score(y_pred, y_true))
            test_losses.append(network.loss)

        for (X_chunk, y_chunk) in zip(val_X_chunks, val_y_chunks):
            predictions = network.predict_batch(batch_x=X_chunk, batch_y=y_chunk)
            y_pred = batch_hot_encode_to_argmax(predictions)
            y_true = batch_hot_encode_to_argmax(y_chunk)

            val_acc.append(accuracy_score(y_pred, y_true))
            val_losses.append(network.loss)

        elapsed_time = time.time() - now
        print("Time taken to complete epoch: {0}".format(elapsed_time))

        lifetime_sum_train_losses.append(np.average(train_losses))
        lifetime_sum_test_losses.append(np.average(test_losses))

        lifetime_sum_train_acc.append(np.average(train_acc))
        lifetime_sum_test_acc.append(np.average(test_acc))

        lifetime_sum_val_acc.append(np.average(val_acc))
        lifetime_sum_val_losses.append(np.average(val_losses))

        if len(lifetime_sum_train_losses) > 1:
            loss_delta = np.average(lifetime_sum_train_losses[-3:]) - lifetime_sum_train_losses[-1]
            train_score = lifetime_sum_train_acc[-1]
            print("Loss Delta {0}".format(loss_delta))
            print("Training Accuracy {0}".format(train_score))
            print()

            if np.abs(loss_delta) < 1e-4:
                print("Loss converged in {0} Epochs".format(j))
                print("Loss = {0}".format(lifetime_sum_train_losses[-1]))
                break

        info = 'L={0}, W={1}, LR={2}, B={3}'.format(losses[0], weight_type, lrs[0], BATCH_SIZE)

        plt.figure()
        t = np.arange(0, len(lifetime_sum_train_losses), 1)
        plt.plot(t, lifetime_sum_train_losses, label="Training Loss")
        plt.plot(t, lifetime_sum_test_losses, label="Validation Loss")
        plt.plot(t, lifetime_sum_val_losses, label="Testing Loss")
        plt.title("Model: {0}\nLoss v Epoch".format(info))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        save_file = 't-LossVEpoch_L={0}_W={1}_LR={2}_B={3}.png'.format(losses[0], weight_type, lrs[0], BATCH_SIZE)
        plt.savefig(save_file)

        plt.figure()
        plt.plot(t, lifetime_sum_train_acc, label="Training Accuracy")
        plt.plot(t, lifetime_sum_test_acc, label="Validation Accuracy")
        plt.plot(t, lifetime_sum_val_acc, label="Testing Accuracy")
        plt.title("Model: {0}\nAccuracy v Epoch".format(info))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        save_file = 't-AccVEpoch_L={0}_W={1}_LR={2}_B={3}.png'.format(losses[0], weight_type, lrs[0], BATCH_SIZE)
        plt.savefig(save_file)
    """
    warnings.filterwarnings("ignore")

    (train_X, train_y), (val_X, val_y) = get_mnist_data()
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    val_X = val_X.reshape((val_X.shape[0], val_X.shape[1] * val_X.shape[2]))

    train_X, test_X, train_y,test_y = train_test_split(train_X, train_y, test_size=0.1, random_state=43, shuffle=True)

    in_shape = train_X.shape[1]
    k_class = train_y.shape[1]

    dims = [in_shape, 128, k_class]
    weight_type = 'gaussian'
    #losses = ['categorical_crossentropy', 'hinge']
    losses = ['hinge']
    activations = [['relu', 'sigmoid', 'softmax'], ['relu', 'sigmoid', 'none']]
    lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for loss_type in losses:
        if loss_type == 'hinge':
            activation = ['relu', 'sigmoid', 'none']
        else:
            activation = ['relu', 'sigmoid', 'softmax']
        for lr in lrs:
            network = Network(layers=[])

            network.compile_model(in_shape=in_shape,
                                  k_class=k_class,
                                  dims=dims,
                                  activations=activation,
                                  loss_type=loss_type,
                                  momentum=0.9,
                                  learning_rate=lr,
                                  weight_type=weight_type)

            EPOCHS = 20
            BATCH_SIZE = 16

            train_X_chunks = [train_X[x:x + BATCH_SIZE] for x in range(0, len(train_X), BATCH_SIZE)]
            train_y_chunks = [train_y[x:x + BATCH_SIZE] for x in range(0, len(train_y), BATCH_SIZE)]

            test_X_chunks = [test_X[x:x + BATCH_SIZE] for x in range(0, len(test_X), BATCH_SIZE)]
            test_y_chunks = [test_y[x:x + BATCH_SIZE] for x in range(0, len(test_y), BATCH_SIZE)]

            val_X_chunks = [val_X[x:x + BATCH_SIZE] for x in range(0, len(val_X), BATCH_SIZE)]
            val_y_chunks = [val_y[x:x + BATCH_SIZE] for x in range(0, len(val_y), BATCH_SIZE)]

            lifetime_sum_train_losses = []
            lifetime_sum_train_acc = []

            lifetime_sum_test_losses = []
            lifetime_sum_test_acc = []

            lifetime_sum_val_losses = []
            lifetime_sum_val_acc = []


            for j in range(0, EPOCHS):
                print("Begin Epoch {0}".format(j))
                now = time.time()

                train_losses = []
                test_losses = []
                train_acc = []
                test_acc = []
                val_acc = []
                val_losses = []

                for (X_chunk, y_chunk) in zip(train_X_chunks, train_y_chunks):
                    predictions = network.fit_batch(batch_x=X_chunk, batch_y=y_chunk)
                    y_pred = batch_hot_encode_to_argmax(predictions)
                    y_true = batch_hot_encode_to_argmax(y_chunk)

                    train_acc.append(accuracy_score(y_pred, y_true))
                    train_losses.append(network.loss)

                for (X_chunk, y_chunk) in zip(test_X_chunks, test_y_chunks):
                    predictions = network.predict_batch(batch_x=X_chunk, batch_y=y_chunk)
                    y_pred = batch_hot_encode_to_argmax(predictions)
                    y_true = batch_hot_encode_to_argmax(y_chunk)

                    test_acc.append(accuracy_score(y_pred, y_true))
                    test_losses.append(network.loss)

                for (X_chunk, y_chunk) in zip(val_X_chunks, val_y_chunks):
                    predictions = network.predict_batch(batch_x=X_chunk, batch_y=y_chunk)
                    y_pred = batch_hot_encode_to_argmax(predictions)
                    y_true = batch_hot_encode_to_argmax(y_chunk)

                    val_acc.append(accuracy_score(y_pred, y_true))
                    val_losses.append(network.loss)

                elapsed_time = time.time() - now
                print("Time taken to complete epoch: {0}".format(elapsed_time))

                lifetime_sum_train_losses.append(np.average(train_losses))
                lifetime_sum_test_losses.append(np.average(test_losses))

                lifetime_sum_train_acc.append(np.average(train_acc))
                lifetime_sum_test_acc.append(np.average(test_acc))

                lifetime_sum_val_acc.append(np.average(val_acc))
                lifetime_sum_val_losses.append(np.average(val_losses))

                if len(lifetime_sum_train_losses) > 1:
                    loss_delta = np.average(lifetime_sum_train_losses[-3:]) - lifetime_sum_train_losses[-1]
                    train_score = lifetime_sum_train_acc[-1]
                    print("Loss Delta {0}".format(loss_delta))
                    print("Training Accuracy {0}".format(train_score))
                    print()

                    if np.abs(loss_delta) < 1e-3:
                        print("Loss converged in {0} Epochs".format(j))
                        print("Loss = {0}".format(lifetime_sum_train_losses[-1]))
                        break

            info = 'L={0}, W={1}, LR={2}, B={3}'.format(loss_type, weight_type, lr, BATCH_SIZE)

            plt.figure()
            t = np.arange(0, len(lifetime_sum_train_losses), 1)
            plt.plot(t, lifetime_sum_train_losses, label="Training Loss")
            plt.plot(t, lifetime_sum_test_losses, label="Validation Loss")
            plt.plot(t, lifetime_sum_val_losses, label="Testing Loss")
            plt.title("Model: {0}\nLoss v Epoch".format(info))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
            save_file = 'LossVEpoch_L={0}_W={1}_LR={2}_B={3}.png'.format(loss_type, weight_type, lr, BATCH_SIZE)
            plt.savefig(save_file)

            plt.figure()
            plt.plot(t, lifetime_sum_train_acc, label="Training Accuracy")
            plt.plot(t, lifetime_sum_test_acc, label="Validation Accuracy")
            plt.plot(t, lifetime_sum_val_acc, label="Testing Accuracy")
            plt.title("Model: {0}\nAccuracy v Epoch".format(info))
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
            save_file = 'AccVEpoch_L={0}_W={1}_LR={2}_B={3}.png'.format(loss_type, weight_type, lr, BATCH_SIZE)
            plt.savefig(save_file)
            
    """


