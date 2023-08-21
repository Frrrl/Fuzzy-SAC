import numpy as np
import math




def data_smooth(data):  # 给一串data，用窗口滑动平均来进行数据平滑化
    """
    :param data: lists of the row data
    :return: lists of the smoothed data
    """
    res = []
    length = len(data)
    for i in range(length):
        if i < 5:  # 当i没有到达窗口长度时：
            res.append(np.mean(data[:2*i+1]))
        elif i >= length - 5:
            gap = i - length  # 代表光标的位置
            res.append(np.mean(data[2*gap+1:]))
        else:
            res.append(np.mean(data[i-5:i+5+1]))
    return res


data = [12, 15, 21, 35, 16, 25, 27, 29, 16, 35, 27, 42, 41, 47, 50, 38, 57, 56, 55]
print(data_smooth(data))