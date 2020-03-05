import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle
from PIL import Image
from forward_network import Network, Layer
import active_functions

def get_data():
    (__, __), (xtt, ttt) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return xtt, ttt

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        data = pickle.load(f)
    layer1 = Layer(
        weight = data['W1'],
        bias = data['b1'],
        active = active_functions.sigmod_function)
    layer2 = Layer(
        weight = data['W2'],
        bias = data['b2'],
        active = active_functions.sigmod_function)
    layer3 = Layer(
        weight = data['W3'],
        bias = data['b3'],
        active = active_functions.softmax_function)
    network = Network()
    network.addLayer(layer1, layer2, layer3)
    return network

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = network.forward(x[i], 0)
    p = np.argmax(y)
    r = p == t[i]
    print("[{0}] Label: {1}, Predict: {2}, Result: {3}".format(i, t[i], p, r))
    if r:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)) + "%")