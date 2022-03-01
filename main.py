import numpy as np
from model import model
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

    return (train_X, y_train), (test_X, y_test)

if __name__ == '__main__':
    if False:
        (train_X, train_y), (test_X, test_y) = get_mnist_data()

        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]*train_X.shape[2], 1))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2], 1))

        in_shape = train_X.shape[1]
        k_class = train_y.shape[1]


    x = np.asarray([[1,0,1,0], [1,0,1,1]])
    y = np.asarray([[1,0,0,0,0], [0,0,0,0,1]])

    x = x.reshape(x.shape[0], x.shape[1], 1)


    in_shape = x.shape[1]
    k_class = y.shape[1]
    

    train_X = x
    train_y = y


    #layer_dim = [4, 3, 2, 5]  # including input and output layer
    #layer_dim = [in_shape, 3, 6, 8, 2, 6, k_class]  # including input and output layer
    layer_dim = [in_shape, 3, 6, k_class]
    layer_activation = ['relu', 'sigmoid', 'softmax', '']
    #layer_activation = ['relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'softmax', '']
    #weight_initialization = 'gaussian var=0.01'
    weight_initialization = 'main_diag'
    loss = 'categorical_crossentropy'
    #loss = 'hinge'
    loss_params = ['j=2']

    model = model(layer_dim=layer_dim,
                  layer_activation=layer_activation,
                  loss_type=loss,
                  loss_params=[],
                  learning_rate=0.005,
                  momentum=1,
                  weight_initialization=weight_initialization,
                  )

    EPOCHS = 100
    BATCH_SIZE = 64

    lifetime_epoch_loss = []

    #np.seterr(invalid='ignore')
    #warnings.filterwarnings('ignore')

    for j in range(0, EPOCHS):
        epoch_loss = []

        print("Begin Epoch {0}".format(j))
        now = time.time()

        model.fit(X=train_X, y=train_y, batch_size=BATCH_SIZE)
        l = model.loss_history
        model.loss_history = []

        elapsed_time = time.time() - now
        print("Time taken to complete epoch: {0}".format(elapsed_time))

        start = time.time()

        sum_epoch_loss = np.sum(l)
        lifetime_epoch_loss.append(sum_epoch_loss)

        end = time.time()
        print("Time taken to complete metrics: {0}".format(end-start))
        print("Current Epoch Loss: {0}".format(lifetime_epoch_loss[len(lifetime_epoch_loss) - 1]))

        #print("Weights Before Epoch Iteration:\n{0}".format(weight_summary))
        print("Weights After Epoch Iteration:\n{0}".format(model.dump_model_weight_summary()))




    fig, axes = plt.subplots(3)
    t = np.arange(0, EPOCHS)
    axes[0].plot(t, lifetime_epoch_loss, color='red')
    axes[0].set_title("Average Loss per Epoch")
    #axes[1].plot(t, train_epoch_acc, color='blue')
    axes[1].set_title("Train Accuracy per Epoch")
    #axes[2].plot(t, val_epoch_acc, color='green')
    axes[2].set_title("Validation Accuracy per Epoch")


    plt.show()

    """
    print("Model Weight Summary (Before BP)")
    print(model.dump_model_weight_summary())
    model.foward_pass(data_point=x, y_true=y_gth[0])
    model.backward_pass(model.y_pred)

    #print("Model Summary")
    #print(model.dump_model_fp_summary())
    print("Model Weight Summary (After BP)")
    print(model.dump_model_weight_summary())
    """


