import numpy as np
from active_functions import sigmod_function as sigmod
# from active_functions import identity_function as identity
from active_functions import softmax_function as softmax

class Network:
    def __init__(self):
        self.layers = []

    def addLayer(self, *layer):
        self.layers.extend(layer)

    def forward(self, x, n = 0):
        if n <= len(self.layers) - 1:
            return self.forward(self.calculate(self.layers[n], x), n + 1)
        else:
            return x

    def calculate(self, layer, z):
        return layer.active(np.dot(z, layer.weight) + layer.bias)

# 神经网络的一层
# weight: 权重
# bias: 偏置
# active: 该层的激活函数
class Layer:
    def __init__(self, weight, bias, active):
        if np.ndim(bias) != 1:
            raise Exception("The dimension of the bias is not 1.")
        weight_columns = weight.shape[0]
        weight_rows = weight.shape[1]
        bias_columns = bias.shape[0]
        if weight_rows != bias_columns:
            raise Exception("The number of weighted rows is different from the size of bias.")
        self.input_rows = bias_columns
        self.weight = weight
        self.bias = bias
        self.active = active
