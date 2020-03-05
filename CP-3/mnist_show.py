import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 训练图像， 训练标签 测试图像，测试标签
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_test[9986]
label = t_test[9986]

print(label)
img = img.reshape(28, 28)
img_show(img)