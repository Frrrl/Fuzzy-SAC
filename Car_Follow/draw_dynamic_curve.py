# 为了探究小车加速度与刹车、油门以及当前车速的关系，特地做了此文件，便于设置模糊规则：

from Refined_Env import Ego_Car
import numpy as np
import matplotlib.pyplot as plt

velos = [0, 5, 10, 15, 20, 25, 30]
throttle = np.arange(-1, 1, 0.01)
car = Ego_Car()

for velo in velos:
    acc = []
    for action in throttle:
        car.reset([0, velo, 0])
        car.step(action)
        acc.append(car.a)
    plt.plot(throttle, acc, color='red', alpha=(velo+10)/50, label='v={}'.format(velo))
plt.legend(loc=0)
plt.grid(True)
plt.title('dynamic reference')
plt.xlabel('brake/throttle')
plt.ylabel('acceleration')
plt.show()



