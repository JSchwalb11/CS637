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

        for i in range(0, len(y_pred)):
            try:
                losses[i] = (y_true[i]) * np.log(y_pred[i])
            except:
                losses[i] = (y_true[i]) * np.log(y_pred[i]._value)

        sum_losses = - np.sum(losses)
        avg_losses = (1/len(y_pred)) * sum_losses

        return avg_losses

    def dloss_dyi_categorical_crossentropy(self, y_pred, y_true):
        # this function is only called from a softmax layer,
        # so a vectorized derivative of both CCE and softmax is returned

        return y_pred - y_true

    def hinge(self, y_pred, y_true):
        assert len(y_pred) == len(y_true)

        ys = np.argmax(y_true)
        tmp = np.zeros_like(y_pred)

        for i, v in enumerate(y_pred):
            a = 1 + v - y_pred[ys]
            tmp[i] = max(a, 0)

        tmp[ys] = 0

        return np.sum(tmp)

    def dloss_dyi_hinge(self, y_pred, y_true):
        ys = np.argmax(y_true)
        tmp = np.zeros_like(y_pred)

        for i, v in enumerate(y_pred):
            a = 1 + v - y_pred[ys]
            if i != ys:
                if a > 0:
                    tmp[i] = 1
                    tmp[ys] -= 1
                else:
                    tmp[i] = 0
            else:
                pass

        s = tmp

        return s

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