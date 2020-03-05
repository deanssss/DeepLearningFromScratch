import numpy as np

# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# sigmod函数
def sigmod_function(x):
    return 1 / (1 + np.exp(-x))

# ReLU函数
def relu_function(x):
    return np.maximum(0, x)

# 恒等函数
def identity_function(x):
    return x

# softmax函数
def softmax_function(x):
    max = np.max(x)
    exp_a = np.exp(x - max) # 溢出处理
    sum = np.sum(exp_a)
    return exp_a/sum