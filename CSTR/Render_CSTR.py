import matplotlib.animation
from matplotlib import pyplot as plt
# plt.rc('font',family='Times New Roman')
from matplotlib.patches import Wedge, Arc
import numpy as np
from SAC_Agent import config

"""
用于渲染CSTR中温度的变化情况
需要的信息包括每一帧的炉温、目标炉温以及使用的冷却剂温度
第一张图为将温度转化为从红（高温）到蓝（低温）的渐变色形式（jet）
第二张图为控制曲线的动态图，实时展现控制效果
"""

mode = 'complex_comp'
max_frames = config['max_step']  # 总共渲染的步数

font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 30,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 20,
         }

# <editor-fold desc="读取存在Render_File中的文件">
txt_name = None
gif_name = None
if mode == 'pure_RL':
    txt_name = './/Render_File//RL_Agent_Test_Trace.txt'
    gif_name = './/Render_File//RL_Trace.gif'
elif mode == 'pure_FLC':
    txt_name = './/Render_File//FLC_Agent_Test_Trace.txt'
    gif_name = './/Render_File//FLC_Trace.gif'
elif mode == 'simple_comp':
    txt_name = './/Render_File//Simple_Compensation_Test_Trace.txt'
    gif_name = './/Render_File//Simple_Comp_Trace.gif'
else:
    txt_name = './/Render_File//Complex_Compensation_Test_Trace.txt'
    gif_name = './/Render_File//Complex_Comp_Trace.gif'
# </editor-fold>

# <editor-fold desc="读取文件中的数据内容，存在三个列表当中，并加入一个Time列表便于画第二张图">
fi = open(txt_name, 'r')
infos = fi.read().strip('\n').split('\n')
Times, Tempers, Targets, Coolers = [], [], [], []
for i in range(len(infos)):
    Times.append(i * 0.05)
    heat, tar, cooler = eval(infos[i])
    Tempers.append(heat)
    Targets.append(tar)
    Coolers.append(cooler)
fi.close()
# </editor-fold>

# <editor-fold desc="将数值转化为颜色">
datas = Tempers + Targets + Coolers
def color_map(data, cmap='jet'):
    """数值映射为颜色"""
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256 / cmo.N
    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    return cs, dmin, dmax
colors, dmin, dmax = color_map(datas)
def digits2color(digit):
    data_code = np.uint8(255 * (digit - dmin) / (dmax - dmin))
    return colors[data_code]
# </editor-fold>

'定义一帧当中画的内容，包含了两张图的画作内容'
def draw_one_frame(frame):
    ax[0].clear()  # 清空画框1
    ax[1].clear()  # 清空画框2
    # <editor-fold desc="画第一张图的代码">
    def draw_container():  # 在画框ax1中画出整个架构，包括反应釜、加入的物料框以及加入的冷却剂框
        # <editor-fold desc="画出所有黑色的画框">
        hat_rect = plt.Rectangle((4, 4.5), 4, 0.3, fill=False, edgecolor='black', linewidth=2)  # 反应釜盖子的方框
        mat_rect = plt.Rectangle((2, 5.1), 1, 1, fill=False, edgecolor='black', linewidth=2)  # 装物料的方框
        cool_rect = plt.Rectangle((2, 1.5), 1, 1, fill=False, edgecolor='black', linewidth=2)  # 装冷却剂的方框
        target_rect = plt.Rectangle((8, 2.5), 1, 0.2, fill=False, edgecolor='black', linewidth=2)  # 目标温度的方框
        small_semi = Arc((6, 2.5), 2, 2, theta1=180, theta2=360, color='black', linewidth=2)
        big_semi = Arc((6, 2.5), 3, 3, theta1=180, theta2=360, color='black', linewidth=2)
        patches = [hat_rect, mat_rect, cool_rect, target_rect, small_semi, big_semi]
        for patch in patches:
            ax[0].add_patch(patch)
        # 另外一些需要绘制的直线
        # xs = [[3, 5.5], [5.5, 5.5], [3, 4.7], [4.5, 4.5], [5, 5], [7, 7], [7.5, 7.5]]
        # ys = [[5.5, 5.5], [3, 5.5], [2.5, 2.5], [2.5, 4.5], [2.5, 4.5], [2.5, 4.5], [2.5, 4.5]]
        xs = [[3, 5.3], [5.3, 5.3], [3, 4.7], [4.5, 4.5], [5, 5], [7, 7], [7.5, 7.5]]
        ys = [[5.5, 5.5], [3, 5.5], [2.5, 2.5], [2.5, 4.5], [2.5, 4.5], [2.5, 4.5], [2.5, 4.5]]
        for i in range(len(xs)):
            ax[0].plot(xs[i], ys[i], color='black', linewidth=2)
        # </editor-fold>
        # <editor-fold desc="相关文字注解">
        ax[0].annotate('Material', (1.8, 4.7), font=font2)
        ax[0].annotate('Target', (7.9, 2.1), font=font2)
        ax[0].annotate('Cooler', (1.8, 1.1), font=font2)
        # </editor-fold>
    # <editor-fold desc="画出各种不同颜色的形状">
    mat_rect = plt.Rectangle((2, 5.1), 1, 1, color=digits2color(350.00))  # 装物料的方框
    cool_rect = plt.Rectangle((2, 1.5), 1, 1, color=digits2color(Coolers[frame]))  # 装冷却剂的方框
    target_rect = plt.Rectangle((8, 2.5), 1, 0.2, color=digits2color(Targets[frame]))  # 目标温度的方框
    matters_rect = plt.Rectangle((5, 2.5), 2, 1, color=digits2color(Tempers[frame]))  # 釜内物料的方框
    matters_circle = Wedge((6, 2.5), 1, 180, 360, color=digits2color(Tempers[frame]))  # 釜内物料的半圆
    patches = [mat_rect, cool_rect, target_rect, matters_rect, matters_circle]
    for patch in patches:
        ax[0].add_patch(patch)
    # </editor-fold>
    # <editor-fold desc="温度相关文字注解">
    material_temp = 350.00
    matters_temp = Tempers[frame]
    target_temp = Targets[frame]
    cooler_temp = Coolers[frame]
    ax[0].annotate('{}K'.format(material_temp), (1.8, 6.2), font=font2)
    ax[0].annotate('{}K'.format(matters_temp), (5.4, 3.7), font=font2)
    ax[0].annotate('{}K'.format(target_temp), (7.8, 2.8), font=font2)
    ax[0].annotate('{}K'.format(cooler_temp), (1.8, 2.6), font=font2)
    # </editor-fold>
    draw_container()
    # ax[0].set_title('Control with {}'.format(mode))
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([0, 10])
    ax[0].axis('off')  # 去除坐标轴
    ax[0].axis('equal')  # 让长宽显示一致
    # </editor-fold>

    # <editor-fold desc="画第二张图的代码">
    # ax[1].set_title('Control Curve with {}'.format(mode))
    ax[1].set_xlim(Times[frame]-1, Times[frame]+1)
    ax[1].set_ylim(Targets[frame]-1, Targets[frame]+1)
    ax[1].grid(True)
    ax[1].plot(Times, Targets, color='black', label='target')
    ax[1].plot(Times[:frame], Tempers[:frame], color='red', label='temperature')
    ax[1].legend(loc='upper right', prop=font1)

    # </editor-fold>


# <editor-fold desc="进行渲染工作的主函数">
fig, ax = plt.subplots(1, 2, figsize=(15, 6))  # 第一张为以颜色表示的反应炉温度，第二张为实时控制曲线图
ani = matplotlib.animation.FuncAnimation(fig, func=draw_one_frame, frames=200, interval=10)
ani.save(gif_name, writer='ffmpeg', fps=15)
plt.show()
# </editor-fold>
