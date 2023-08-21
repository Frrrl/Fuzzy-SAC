import matplotlib.pyplot as plt
import numpy as np

"""
通过读取txt文件中的数据来画出曲线图
数据中包含两行，一行是reward，另一行是danger
同时，为了防止曲线的震荡，需要进行窗口的滑动平均处理
"""
smooth_k = 5  # 滑动窗口大小设置，即每k个数据取平均值计算

def cal_ave_var(nums):  # 给进10个数，给出他们的均值和方差
    ave = np.mean(nums)
    var = np.std(nums)
    up, down = ave + var, ave - var
    return ave, up, down


def data_smooth(data):  # 给一串data，用窗口滑动平均来进行数据平滑化
    """
    :param data: lists of the row data
    :return: lists of the smoothed data
    """
    res = []
    length = len(data)
    for i in range(length):
        if i < smooth_k:  # 当i没有到达窗口长度时：
            res.append(np.mean(data[:2*i+1]))
        elif i >= length - smooth_k:
            gap = i - length  # 代表光标的位置
            res.append(np.mean(data[2*gap+1:]))
        else:
            res.append(np.mean(data[i-smooth_k:i+smooth_k+1]))
    return res


def draw_variance(file_path, mycolor, plot_label):  # 给一个文件地址，然后画出其文件对应的折线图，并指定颜色
    label_dict = {
        'pure_RL': 'Soft Actor Critic',
        'pure_FLC': 'Fuzzy Logic Control',
        'simple_comp': 'Our Method without Action Fusion',
        'complex_comp': 'Our Method'
    }

    # 从文件中收集数据
    fi = open(file_path)
    R_data, D_data = [], []
    lines = len(fi.readlines()) // 2
    fi.seek(0)
    for i in range(lines):
        reward_data = fi.readline().strip('\n').split(',')
        danger_data = fi.readline().strip('\n').split(',')
        R_data.append(list(map(float, reward_data)))
        D_data.append(list(map(float, danger_data)))
    fi.close()

    # 对R_data和D_data中的数据进行log处理
    for j in range(len(R_data)):
        for k in range(len(R_data[j])):
            R_data[j][k] = -np.log10(-R_data[j][k])
            # D_data[j][k] = np.log(D_data[j][k])
    # 数据分割
    average1, variance_up1, variance_down1 = [], [], []
    average2, variance_up2, variance_down2 = [], [], []
    episode = list(range(0, lines*10, 10))
    for i in range(lines):
        ave1, up1, down1 = cal_ave_var(R_data[i])
        ave2, up2, down2 = cal_ave_var(D_data[i])
        average1.append(ave1)
        variance_up1.append(up1)
        variance_down1.append(down1)
        average2.append(ave2)
        variance_up2.append(up2)
        variance_down2.append(down2)
    average1 = data_smooth(average1)
    variance_up1 = data_smooth(variance_up1)
    variance_down1 = data_smooth(variance_down1)
    average2 = data_smooth(average2)
    variance_up2 = data_smooth(variance_up2)
    variance_down2 = data_smooth(variance_down2)
    # 横纵坐标的字体
    font1 = {'family': 'Times New Roman',
             'weight': 'heavy',
             'size': 30,
             }
    # label的字体
    font2 = {'family': 'Times New Roman',
             'weight': 'heavy',
             'size': 30,
             }
    # 坐标轴数字的字体
    label_font = {'family': 'Times New Roman',
             'weight': 'heavy',
             'size': 20,
    }
    # axes.plot(episode, average1, color=mycolor, label=label_dict[plot_label])
    # axes.fill_between(episode, variance_down1, variance_up1, facecolor=mycolor, alpha=0.3)
    # axes.legend(loc='lower right', prop=font2)
    # axes.grid(True)
    # # axes.set_title('Reward Curve (log10)')
    # axes.set_xlabel('Training Step', fontdict=font1)
    # axes.set_ylabel('Cumulative Reward', fontdict=font1)
    axes.plot(episode, average2, color=mycolor, label=label_dict[plot_label])
    axes.fill_between(episode, variance_down2, variance_up2, facecolor=mycolor, alpha=0.3)
    axes.legend(loc='upper right', prop=font2)
    axes.grid(True)
    # axes.set_title('Bad State Curve')
    axes.set_xlabel('Training Step', fontdict=font1)
    axes.set_ylabel('Bad State', fontdict=font1)
    plt.xticks(font=label_font)
    plt.yticks(font=label_font)

# <editor-folder, desc='打开文件'>
fi1 = './/Render_File.//Training_RL_data.txt'
fi2 = './/Render_File.//Training_FLC_data.txt'
fi3 = './/Render_File.//Training_Complex_Comp_data.txt'
fi4 = './/Render_File.//Training_Simple_Comp_data.txt'
fig, axes = plt.subplots(1, 1)
# </editor-folder>


# draw_variance(fi1, 'green', 'pure_RL')
# draw_variance(fi2, 'dodgerblue', 'pure_FLC')
draw_variance(fi3, 'red', 'complex_comp')
draw_variance(fi4, 'lightsalmon', 'simple_comp')


plt.show()
