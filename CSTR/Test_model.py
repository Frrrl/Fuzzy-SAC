from SAC_Agent import SAC, config
from CSTR_Env import env_for_rl, env_for_comp
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt
import numpy as np
import time as t
"""
用来比较和测试各种模型的效果
还需要将轨迹保存到txt文件当中，便于进行后续的图片渲染
值得注意的是，尾号为7的代表纯RL的模型，8代表用Action_Corrector的模型，9代表纯加法的模型
这里显示FLC的推理速度快于RL，是因为都采用了CPU进行计算的原因
控制性能指标设计为误差的平方和
"""

mode = 'simple_comp'
# <editor-fold desc="根据mode生成环境与智能体和相应的文件">
if mode == 'pure_RL':
    env = env_for_rl()
    model_num = 17
    agent = SAC(env)
    agent.load_nets('./', model_num)

elif mode == 'pure_FLC':
    env = env_for_rl()
    agent = Fuzzy_Agent()

elif mode == 'simple_comp':
    env = env_for_comp('simple_comp')  # 代表使用combine policy
    model_num = 49
    agent = SAC(env)
    # agent.load_nets('./', model_num)

else:
    env = env_for_comp('complex_comp')  # 代表使用combine policy
    model_num = 8
    agent = SAC(env)
    # agent.load_nets('./', model_num)
# </editor-fold>
start = t.time()

env.reset('test')
Time, Target, Temper, Cooler = [], [], [], []
rewards = 0
for step in range(config['max_step']):
    '先对env中的数据进行采集'
    Time.append(step / 20)  # 按秒来计算的训练时间
    Target.append(env.target + 350)  # 实际的Target温度
    Temper.append(env.x2 + 350)  # 实际的釜内温度
    '一个正常的step应该做的动作'
    state = env.get_state()
    action = agent.get_action(state)
    reward, done_mask = env.step(action[0], 'random')[1:3]
    '采集每一步制冷剂的温度'
    if mode == 'pure_RL' or mode == 'pure_FLC':
        cooler_temp = action[0] * 10 + 338
        Cooler.append(cooler_temp)
    else:
        real_act = np.clip(env.action_fusion(env.FLC_Action, action), -1, 1)
        cooler_temp = real_act[0] * 10 + 338
        Cooler.append(cooler_temp)
    rewards += reward
    if not done_mask:  # 如果done_mask是0，就退出
        break
end = t.time()
print('time is {}'.format(end-start))

'画图'
fig, axes = plt.subplots(1, 2)
# 画四个图片
font1 = {'family': 'sans-serif',
         'weight': 'semibold',
         'size': 'medium',
         }
axes[0].set_title("Control of Temperature", loc='center', color='black')
axes[0].plot(Time, Temper, color='olive', label='trace')
axes[0].plot(Time, Target, color='black', label='target')
axes[0].legend(loc=0, prop=font1)
axes[0].grid(True)

axes[1].set_title("Cooler Temperature", loc='center', color='black')
axes[1].plot(Time, Cooler, color='tomato', label='Cooler heat')
axes[1].legend(loc=0, prop=font1)
axes[1].grid(True)


print('reward:{}'.format(rewards))
plt.show()
















