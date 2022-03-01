import numpy as np
import sys
from loss import loss
import re

class model:

    def __init__(self, layer_dim, layer_activation, loss_type, loss_params, learning_rate, momentum, weight_initialization=None):
