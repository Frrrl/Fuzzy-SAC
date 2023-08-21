from SAC_Agent import SAC, config
from CSTR_Env import env_for_rl, env_for_comp
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt
import random
import numpy as np

"""
之前的Test_Model文件没有对不同的模型控制效果进行对比
该文件为了实现对模型效果的对比，将任务环境都定义为同一个，切实地比较不同模型的性能
其中环境的初始条件应该相同，且任务的跟踪目标也要相同
整个任务指标设置为累计误差平方和
"""


def name_to_env_agent(mode):  # 根据mode名字，返回相应的环境和智能体与路径保存的文件
    if mode == 'pure_RL':
        env = env_for_rl()
        agent = SAC(env)
        agent.load_nets('./', 47)
        fo = open('.//Render_File//RL_Agent_Test_Trace.txt', 'w')
    elif mode == 'pure_FLC':
        env = env_for_rl()
        agent = Fuzzy_Agent()
        fo = open('.//Render_File//FLC_Agent_Test_Trace.txt', 'w')
    elif mode == 'simple_comp':
        env = env_for_comp('simple_comp')  # 代表使用combine policy
        agent = SAC(env)
        agent.load_nets('./', 49)
        fo = open('.//Render_File//Simple_Compensation_Test_Trace.txt', 'w')
    elif mode == 'complex_comp':
        env = env_for_comp('complex_comp')  # 代表使用combine policy
        agent = SAC(env)
        agent.load_nets('./', 48)
        fo = open('.//Render_File//Complex_Compensation_Test_Trace.txt', 'w')
    else:
        return None
    return env, agent, fo

def draw_trace(env, agent, targets, axes, name, fo):
    # 一个字典，用于给特定的agent上色
    color_dict = {
        'pure_RL': 'green',
        'pure_FLC': 'dodgerblue',
        'complex_comp': 'red',
        'simple_comp': 'lightsalmon'
    }
    # <editor-fold desc="采集数据">
    Time, Target, Temper, Cooler = [], [], [], []
    rewards = 0
    for step in range(config['max_step']):
        '先对env中的数据进行采集'
        Time.append(step / 20)
        Target.append(env.target + 350)  # 实际的Target温度
        Temper.append(env.x2 + 350)  # 实际的釜内温度
        '一个正常的step应该做的动作'
        state = env.get_state()
        action = agent.get_action_test(state)
        reward, done_mask = env.step(action[0], targets)[1:3]
        '采集每一步制冷剂的温度'
        if name == 'pure_RL' or name == 'pure_FLC':
            cooler_temp = action[0] * 10 + 338
            Cooler.append(cooler_temp)
        else:
            real_act = np.clip(env.action_fusion(env.FLC_Action, action), -1, 1)
            cooler_temp = real_act[0] * 10 + 338
            Cooler.append(cooler_temp)
        rewards += reward
        fo.write("{:.2f},{:.2f},{:.2f}\n".format(env.x2+350, env.target+350, cooler_temp))
        if not done_mask:  # 如果done_mask是0，就退出
            break
    print('{}:{}'.format(name, rewards))
    # </editor-fold>
    # <editor-fold desc="将数据标记在四张图上">
    axes[0].plot(Time, Temper, color=color_dict[name], label=name)
    axes[1].plot(Time, Cooler, color=color_dict[name], label=name)

    # </editor-fold>


def multi_model_compare(agent_li):  # 根据agents中的智能体集合，给出多个智能体的控制对比效果
    # 定义初始的[Ca, T, Target]与target
    para = [random.uniform(-0.5, 0.5), random.uniform(-5, 5), random.uniform(-5, 5)]
    targets = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)]
    # <editor-fold desc="设置两张图的各类初始定义">
    fig, axes = plt.subplots(1, 2)
    font1 = {'family': 'sans-serif',
             'weight': 'semibold',
             'size': 'medium',
             }
    axes[0].set_title("Control of Temperature", loc='center', color='black')
    axes[0].grid(True)
    axes[1].set_title("Cooler Temperature", loc='center', color='black')
    axes[1].grid(True)
    # </editor-fold>

    # 画target曲线
    times, Tars = [], []
    for step in range(config['max_step']):
        times.append(step / 20)
        Tars.append(targets[step//(config['max_step']//4)]+350)
    axes[0].plot(times, Tars, color='black', label='target temperature')
    for ele in agent_li:
        ele_env, ele_agent, fo = name_to_env_agent(ele)
        ele_env.reset(para)  # 这一步保证了各个不同的环境初始参数是一样的
        draw_trace(ele_env, ele_agent, targets, axes, ele, fo)
        fo.close()
    axes[0].legend(loc=0, prop=font1)
    axes[1].legend(loc=0, prop=font1)
    plt.show()


agent_list = ['pure_RL', 'pure_FLC', 'complex_comp', 'simple_comp']
multi_model_compare(agent_list)
