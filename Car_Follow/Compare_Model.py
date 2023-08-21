from SAC_Agent import SAC, config
from Refined_Env import env_for_rl, env_for_comp, Lead_Car
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt
import random

"""
之前的Test_Model文件没有对不同的模型控制效果进行对比
该文件为了实现对模型效果的对比，将任务环境都定义为同一个，切实地比较不同模型的性能
其中环境的初始条件应该相同，而且前车的行驶策略也要相同
"""


def generate_lead_policy():  # 将lead_policy以固定的以step为轴的列表形式传给每个环境
    policy_list = [None] * config['max_step']  # 乘以400个None
    policy_list[0] = 0
    change_node = [100, 200, 300]  # 表示分别在100，200,300的step处进行
    for node in change_node:
        policy_list[node] = random.uniform(-0.5, 0.5)
    return policy_list


def generate_target_and_lead_velo(target, policy, velo_init):  # 返回三个列表，一个是time，一个是target数据，一个是lead_velo数据
    car = Lead_Car()
    car.v = velo_init
    time, targets, velos = [], [], []
    for step in range(config['max_step']):
        time.append(step / 20)
        targets.append(target)
        velos.append(car.v)
        car.step(policy[step])
    return time, targets, velos


def name_to_env_agent(mode):  # 根据mode名字，返回相应的环境和智能体
    if mode == 'pure_RL':
        env = env_for_rl()
        agent = SAC(env)
        agent.load_nets('./', 197)
    elif mode == 'pure_FLC':
        env = env_for_rl()
        agent = Fuzzy_Agent()
    elif mode == 'simple_comp':
        env = env_for_comp('simple_comp')  # 代表使用combine policy
        agent = SAC(env)
        agent.load_nets('./', 199)
    elif mode == 'complex_comp':
        env = env_for_comp('complex_comp')  # 代表使用combine policy
        agent = SAC(env)
        agent.load_nets('./', 198)
    else:
        return None
    return env, agent


# 其中env已经reset完毕，主要的功能为画出env在agent的策略下跑出的轨迹, policy为前车的运动策略， axes为已经建好的4个画框
def draw_trace(env, agent, policy, axes, name):
    # 一个字典，用于给特定的agent上色
    color_dict = {
        'pure_RL': 'green',
        'pure_FLC': 'dodgerblue',
        'complex_comp': 'red',
        'simple_comp': 'lightsalmon'
    }
    label_dict = {
        'pure_RL': 'SAC',
        'pure_FLC': 'FLC',
        'simple_comp': 'DRLCFC without Action Fusion',
        'complex_comp': 'DRLCFC'
    }
    # <editor-fold desc="采集数据">
    dist, time, velo, acc, Action = [], [], [], [], []
    rewards = 0
    for step in range(config['max_step']):
        '先对env中的数据进行采集'
        time.append(step / 20)
        dist.append(env.dist)
        velo.append(env.ego_car.v)
        '一个正常的step应该做的动作'
        state = env.get_state()
        action = agent.get_action_test(state)
        lead_action = policy[step]
        reward, done_mask = env.step(action[0], lead_action)[1:3]
        '以下为为了绘图所添加的数据'
        acc.append(env.ego_car.a)
        Action.append(env.action)
        rewards += reward * 0.05
        if not done_mask:  # 如果done_mask是0，就退出
            break
    print(rewards)
    # </editor-fold>
    # <editor-fold desc="将数据标记在四张图上">
    axes[0][0].plot(time, dist, color=color_dict[name], label=label_dict[name]+':{:.2f}'.format(rewards))
    axes[0][1].plot(time, acc, color=color_dict[name], label=label_dict[name])
    axes[1][0].plot(time, velo, color=color_dict[name], label=label_dict[name])
    axes[1][1].plot(time, Action, color=color_dict[name], label=label_dict[name])
    # </editor-fold>


def multi_model_compare(agent_li):  # 根据agents中的智能体集合，给出多个智能体的路径对比效果
    # [[lead_acc, lead_velo, lead_x], [ego_acc, ego_velo, ego_x], target_dist]
    para = [[0, random.uniform(3, 5), random.uniform(6, 7)], [0, random.uniform(1, 2), 0], random.uniform(5, 6)]
    policy = generate_lead_policy()
    # <editor-fold desc="设置四张图的各类初始定义">
    fig, axes = plt.subplots(2, 2)
    font1 = {'family': 'sans-serif',
             'weight': 'semibold',
             'size': 'medium',
             }
    axes[0][0].set_title("control of distance", loc='center', color='black')
    axes[0][0].grid(True)
    axes[0][1].set_title("control of ego car acceleration", loc='center', color='black')
    axes[0][1].grid(True)
    axes[1][0].set_title("control of velocity", loc='center', color='black')
    axes[1][0].grid(True)
    axes[1][1].set_title("Action of the Controller", loc='center', color='black')
    axes[1][1].grid(True)
    # </editor-fold>

    # 画target曲线和elad_velo曲线
    times, targets, velos = generate_target_and_lead_velo(para[-1], policy, para[0][1])
    axes[0][0].plot(times, targets, color='black', label='target dist')
    axes[1][0].plot(times, velos, color='black', label='lead velocity')

    for ele in agent_li:
        ele_env, ele_agent = name_to_env_agent(ele)
        ele_env.reset(para)  # 这一步保证了各个不同的环境初始参数是一样的
        draw_trace(ele_env, ele_agent, policy, axes, ele)
    axes[0][0].legend(loc=0, prop=font1)
    axes[0][1].legend(loc=0, prop=font1)
    axes[1][0].legend(loc=0, prop=font1)
    axes[1][1].legend(loc=0, prop=font1)
    plt.show()


agent_list = ['pure_RL', 'pure_FLC', 'complex_comp']
multi_model_compare(agent_list)
