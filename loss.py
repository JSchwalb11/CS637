import numpy as np

class loss:
    def __init__(self, loss_type):
        if loss_type == 'categorical_crossentropy':
            self.loss_func = self.categorical_crossentropy
            self.dloss_dyi = self.dloss_dyi_categorical_crossentropy

        elif loss_type == 'hinge':
            self.loss_func = self.hinge
            self.dloss_dyi = self.dloss_dyi_hinge

        else:
            self.loss_func = self.euclidian_distance

    def categorical_crossentropy(self, y_pred, y_true):
        assert len(y_pred) == len(y_true)
        losses = np.zeros_like(y_pred)
        #y_pred = self.check_hot_encoding(y_pred)

        for i in range(0, len(y_pred)):
            losses[i] = (y_true[i]) * np.log(y_pred[i] + 1e-6)

        sum_losses = - np.sum(losses)
        avg_losses = (1/len(y_pred)) * sum_losses

        return avg_losses

    def dloss_dyi_categorical_crossentropy(self, y_pred, y_true):
        #return -np.log(y_pred)
        return y_pred - y_true

    def hinge(self, y_pred, y_true, j=2):
        assert len(y_pred) == len(y_true)

        yt = y_pred[np.argmax(y_true)]
        y_pred[np.argmax(y_true)] = 0
        yp = y_pred[np.argmax(y_pred)]

        return max(0, 1 + yt - yp)

    def dloss_dyi_hinge(self, y_pred, y_true):
        grad_input = np.where(y_pred * y_true < 1, -y_true / y_pred.size, 0)

        return grad_input

    def euclidian_distance(self, y_pred, y_true):
        return np.sqrt((y_pred - y_true)**2)

    def check_hot_encoding(self, y_pred):
        for v in y_pred:
            if int(v) != 0 or int(v) != 1:
                y_pred = self.hot_encode(y_pred)

        return y_pred

    def hot_encode(self, y_pred):
        hot_encoded = np.zeros_like(y_pred).astype('uint8')
        hot_encoded[np.argmax(y_pred)] = 1
        return hot_encoded