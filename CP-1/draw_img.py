import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('e:/workspace/DeepLearning/CP-1/sample.bmp') # 读取图像
plt.imshow(img) # 绘制图像

plt.show()