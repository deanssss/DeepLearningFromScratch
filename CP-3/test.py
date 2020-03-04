import numpy as np
import matplotlib.pylab as plt
import active_functions as act

x = np.arange(-5.0, 5.0, 0.1)
y1 = act.step_function(x)
y2 = act.sigmod_function(x)
y3 = act.relu_function(x)

plt.plot(x, y1, label="step function", linestyle="--")
plt.plot(x, y2, label="sigmod function", linestyle="-.")
plt.plot(x, y3, label="ReLU function")

plt.ylim(-0.1, 1.1)

plt.xlabel("X") # X轴标签
plt.ylabel("Y") # Y轴标签

plt.title("active functions") 
plt.legend() # 显示图示标签

plt.show()