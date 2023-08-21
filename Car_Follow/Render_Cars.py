import matplotlib.animation
from matplotlib import pyplot as plt
import numpy as np
from SAC_Agent import config

"""
用之前保存的txt文件来画出跟车环境的变化
txt文件每行包括三个内容，ego_car_x，lead_car_x和distance
"""
lane_length = 300  # 车道的长度
max_frames = config['max_step']  # 总共渲染的步数
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 30,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 20,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 25,
         }

mode = 'pure_FLC'

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

fi = open(txt_name, 'r')
infos = fi.read().strip('\n').split('\n')

x_ego, x_lead, dist, target, v_ego, v_lead, Times = [], [], [], [], [], [], []
for i in range(len(infos)):
    Times.append(i * 0.05)
    egox, leadx, dis, tar, ve, vl = eval(infos[i])
    x_ego.append(egox)
    x_lead.append(leadx)
    dist.append(dis)
    target.append(tar)
    v_ego.append(ve)
    v_lead.append(vl)
fi.close()

def draw_one_frame(frame):
    ax[0].clear()
    ax[1].clear()
    # <editor-fold desc="画第一张图的代码">
    # 画上主体的三个矩形
    road_rect = plt.Rectangle((-10, -2), lane_length, 5, color='gray', alpha=0.5)
    ego_rect = plt.Rectangle((x_ego[frame], 0), 2, 1, color='lime', label='ego car')
    lead_rect = plt.Rectangle((x_lead[frame], 0), 2, 1, color='blue', label='lead car')
    ax[0].add_patch(road_rect)
    ax[0].add_patch(ego_rect)
    ax[0].add_patch(lead_rect)
    ax[0].legend(loc='upper right', prop=font3)

    # 为了便于观察，每隔10m就在矩形两侧画上黑点
    x = np.arange(0, lane_length, 10)
    y1 = [-2 for _ in range(len(x))]
    y2 = [3 for _ in range(len(x))]
    ax[0].scatter(x, y1, color='black', s=20)
    ax[0].scatter(x, y2, color='black', s=20)

    # 相关文字注释
    distance = dist[frame]
    target_dist = target[frame]
    velo_ego = v_ego[frame]
    velo_lead = v_lead[frame]
    dist_txt_x = (x_ego[frame] + x_lead[frame]) / 2
    ax[0].annotate('distance: {:.2f}m'.format(distance), (dist_txt_x - 2.5, -4), font=font2)
    ax[0].annotate('target: {:.2f}m'.format(target_dist), (dist_txt_x - 2, -6), font=font2)
    ax[0].annotate('v2={:.2f}m/s'.format(velo_ego + 10), (x_ego[frame] - 2.5, 4), font=font2)
    ax[0].annotate('v1={:.2f}m/s'.format(velo_lead + 10), (x_lead[frame], 4), font=font2)

    # 设置坐标轴缩放比例
    x_left = x_ego[frame] - 10
    x_right = x_lead[frame] + 10
    x_interval = x_right - x_left
    y_down = -x_interval / 2
    y_up = - y_down
    ax[0].set_xlim([x_left, x_right])
    ax[0].set_ylim([y_down, y_up])
    # </editor-fold>

    # <editor-fold desc="画第二张图的代码">

    ax[1].set_xlim(Times[frame]-1, Times[frame]+1)
    ax[1].set_ylim(target[frame]-1, target[frame]+1)
    ax[1].grid(True)
    ax[1].plot(Times, target, color='black', label='target')
    ax[1].plot(Times[:frame], dist[:frame], color='red', label='distance')
    ax[1].legend(loc='upper right', prop=font1)
    # </editor-fold>

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ani = matplotlib.animation.FuncAnimation(fig, func=draw_one_frame, frames=200, interval=10)
ani.save(gif_name, writer='ffmpeg', fps=15)
plt.show()
