import numpy as np

class loss:
    def __init__(self, loss_type, loss_params):
        if loss_type == 'categorical_crossentropy':
            self.loss_func = self.categorical_crossentropy
            self.dloss_dyi = self.dloss_dyi_categorical_crossentropy

        elif loss_type == 'hinge':
            self.loss_func = self.hinge
            self.dloss_dyi = self.dloss_dyi_hinge
            #self.j = loss_params[2:]

        else:
            self.loss_func = self.euclidian_distance

    def categorical_crossentropy(self, y_pred, y_true):
        assert len(y_pred) == len(y_true)
        losses = np.zeros_like(y_pred)

        for i in range(0, len(y_pred)):
            losses[i] = (y_true[i]) * np.log(y_pred[i] + 1e-6)

        sum_losses = - np.sum(losses)
        avg_losses = (1/len(y_pred)) * sum_losses

        return avg_losses

    def dloss_dyi_categorical_crossentropy(self, y_pred):
        return -np.log(y_pred)

    def hinge(self, y_pred, y_true, j=1):
        assert len(y_pred) == len(y_true)
        losses = np.zeros_like(y_pred)

        for i in range(0, len(y_pred)):
            losses[i] = max(0, 1 - j * y_pred[i])

        return losses

    def dloss_dyi_hinge(self, y_pred):
        change = np.zeros_like(y_pred)

        for i, pred in enumerate(y_pred):
            if pred < 1:
                change[i] = -1
            else:
                change[i] = 0

        return change


    def euclidian_distance(self, y_pred, y_true):
        return np.sqrt((y_pred - y_true)**2)