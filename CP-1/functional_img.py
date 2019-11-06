import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label = "sin")
plt.plot(x, y2, label = "cos", linestyle = "--")

plt.xlabel("X") # X轴标签
plt.ylabel("Y") # Y轴标签

plt.title("sin & cos") 
plt.legend() # 显示图示标签

plt.show()