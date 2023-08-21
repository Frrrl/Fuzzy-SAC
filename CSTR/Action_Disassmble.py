from SAC_Agent import SAC, config
from CSTR_Env import env_for_rl, env_for_comp
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt
import numpy as np
import time as t

"""
专门针对补偿控制方法的动作分解进行的实验
取出simple_comp与complex_comp的策略，并画出其中的原始FLC输出、RL输出以及总的输出
期望的效果是RL实现了对于模糊控制震荡的消除效果
"""

mode = 'complex_comp'
env = env_for_comp(mode)
agent = SAC(env)
# <editor-fold desc="根据mode去换神经网络模型">
if mode == 'simple_comp':
    model_num = 99
elif mode == 'complex_comp':
    model_num = 98
else:
    model_num = None
# </editor-fold>
agent.load_nets('./', model_num)


env.reset('test')
Time, FLC_Out, RL_Out, True_Out = [], [], [], []

for step in range(config['max_step']):
    state = env.get_state()
    action = agent.get_action_test(state)
    reward, done_mask = env.step(action[0], 'random')[1:3]

    Time.append(step / 20)
    FLC_Out.append(env.FLC_Action)
    RL_Out.append(action[0])
    real_act = np.clip(env.action_fusion(env.FLC_Action, action), -1, 1)
    True_Out.append(real_act)
    if not done_mask:  # 如果done_mask是0，就退出
        break

fig, axes = plt.subplots(1, 1)
font1 = {'family': 'sans-serif',
         'weight': 'semibold',
         'size': 'medium',
         }
axes[0].set_title("Action Disassemble", loc='center', color='black')
axes[0].plot(Time, FLC_Out, color='tomato', label='FLC Output')
axes[0].plot(Time, RL_Out, color='lightsteelblue', label='RL Output')
axes[0].plot(Time, True_Out, color='red', label='Final Output')
axes[0].legend(loc=0, prop=font1)
axes[0].grid(True)

plt.show()

