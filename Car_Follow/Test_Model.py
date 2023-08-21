from SAC_Agent import SAC, config
from Refined_Env import env_for_rl, env_for_comp, lead_policy
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt
import time as t
"""
用来比较和测试各种模型的效果
还需要将轨迹保存到txt文件当中，便于进行后续的图片渲染
值得注意的是，尾号为7的代表纯RL的模型，8代表用Action_Corrector的模型，9代表纯加法的模型
这里显示FLC的推理速度快于RL
"""


mode = 'pure_RL'

if mode == 'pure_RL':
    env = env_for_rl()
    model_num = 197
    agent = SAC(env)
    agent.load_nets('./', model_num)
    fo = open('.//Render_File//RL_Agent_Test_Trace.txt', 'w')  # 写模式打开一个文件

elif mode == 'pure_FLC':
    env = env_for_rl()
    agent = Fuzzy_Agent()
    fo = open('.//Render_File//FLC_Agent_Test_Trace.txt', 'w')  # 写模式打开一个文件

elif mode == 'simple_comp':
    env = env_for_comp('simple_comp')  # 代表使用combine policy
    model_num = 199
    agent = SAC(env)
    agent.load_nets('./', model_num)
    fo = open('.//Render_File//Simple_Compensation_Test_Trace.txt', 'w')  # 写模式打开一个文件

else:
    env = env_for_comp('complex_comp')  # 代表使用combine policy
    model_num = 198
    agent = SAC(env)
    agent.load_nets('./', model_num)
    fo = open('.//Render_File//Complex_Compensation_Test_Trace.txt', 'w')  # 写模式打开一个文件

start = t.time()
env.reset('test')
target, dist, time, velo, lead_velo, acc, Action = [], [], [], [], [], [], []
rewards = 0
for step in range(config['max_step']):
    '先对env中的数据进行采集'
    time.append(step / 20)
    target.append(env.target_dist)
    dist.append(env.dist)
    velo.append(env.ego_car.v)
    lead_velo.append(env.lead_car.v)
    '一个正常的step应该做的动作'
    state = env.get_state()
    action = agent.get_action_test(state)
    lead_action = lead_policy(step, config['max_step'], mode='test')
    reward, done_mask = env.step(action[0], lead_action)[1:3]
    '以下为为了绘图所添加的数据'
    acc.append(env.ego_car.a)
    Action.append(action)
    rewards += reward
    fo.write("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(env.ego_car.x, env.lead_car.x, env.dist,
                                                                  env.target_dist, env.ego_car.v, env.lead_car.v))
    if not done_mask:  # 如果done_mask是0，就退出
        break

finish = t.time()
print('运行时间：'+str(finish - start))

'画图'
fig, axes = plt.subplots(2, 2)
# 画四个图片
font1 = {'family': 'sans-serif',
         'weight': 'semibold',
         'size': 'medium',
         }
axes[0][0].set_title("change of distance", loc='center', color='black')
axes[0][0].plot(time, dist, color='olive', label='distance')
axes[0][0].plot(time, target, color='black', label='target_distance')
axes[0][0].legend(loc=0, prop=font1)
axes[0][0].grid(True)

axes[0][1].set_title("change of ego car acceleration", loc='center', color='black')
axes[0][1].plot(time, acc, color='tomato', label='acceleration')
axes[0][1].legend(loc=0, prop=font1)
axes[0][1].grid(True)

axes[1][0].set_title("change of velocity", loc='center', color='black')
axes[1][0].plot(time, lead_velo, color='blue', label='lead car velocity')
axes[1][0].plot(time, velo, color='green', label='ego car velocity')
axes[1][0].legend(loc=0, prop=font1)
axes[1][0].grid(True)

axes[1][1].set_title("Action of the Controller",loc='center', color='black')
axes[1][1].plot(time, Action, color='red', label='Agent_Action')
axes[1][1].legend(loc=0, prop=font1)
axes[1][1].grid(True)

print(rewards)
plt.show()
fo.close()















