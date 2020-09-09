# example of a wgan for generating handwritten digits
from keras import backend
from keras.constraints import Constraint


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)
