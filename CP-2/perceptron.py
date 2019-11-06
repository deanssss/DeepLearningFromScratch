# 使用感知机模拟门电路

import numpy as np

# 多输入-单层感知机构造器
def perceptron_creater(w, b):
    def perceptron(x):
        return 1 if np.sum(x * w) + b > 0 else 0
    return perceptron

# 两输入-两层感知机构造器
def two_l_two_in_perceptron_creater(perceptron1, perceptron2, combine):
    def two_l_perceptron(x):
        s1 = perceptron1(x)
        s2 = perceptron2(x)
        s  = np.array([s1, s2])
        return combine(s)
    return two_l_perceptron

wAND = np.array([0.5, 0.5])
AND = perceptron_creater(wAND, -0.7)
AND.__name__ = "AND"

wOR = np.array([0.5, 0.5])
OR = perceptron_creater(wOR, -0.2)
OR.__name__ = "OR"

wNAND = np.array([-0.5, -0.5])
NAND = perceptron_creater(wNAND, 0.7)
NAND.__name__ = "NAND"

XOR = two_l_two_in_perceptron_creater(NAND, OR, AND)
XOR.__name__ = "XOR"

# 测试
## 测试用例
x1 = np.array([1, 1])
x2 = np.array([1, 0])
x3 = np.array([0, 1])
x4 = np.array([0, 0])
cases = [x1, x2, x3, x4]

ops = [AND, OR, NAND, XOR]

def calcul(x, op):
    print("{0} {1} {2} = {3}".format(x[0], op.__name__, x[1], op(x)))

for op in ops:
    for x in cases:
        calcul(x, op)
    print("")