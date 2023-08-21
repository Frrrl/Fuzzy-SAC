from SAC_Agent import SAC, config
from CSTR_Env import env_for_rl, env_for_comp
import time as t
from Fuzzy_Controller import Fuzzy_Agent

"""
单纯给出7个回合下模型的reward输出，便于观察训练情况
每次调用都返回一个列表
1.创建一个列表
"""

test_repeats = 7

def test_with_fixed_env(agent, mode='complex_comp'):
    if mode == 'pure_RL' or mode == 'pure_FLC':
        env = env_for_rl()
    elif mode == 'simple_comp' or mode == 'complex_comp':
        env = env_for_comp(mode)
    else:
        raise ValueError('No mode named {}'.format(mode))
    rewards, dangers = [], []
    for _ in range(test_repeats):
        env.reset('test')
        score = 0
        danger_rate = 0
        error = env.x2 - env.target  # 给error一个初始值
        for step in range(config['max_step']):
            '以下为正常一个step该做的动作'
            state = env.get_state()
            action = agent.get_action(state)
            reward, done_mask = env.step(action[0], 'test')[1:3]
            score += reward
            delta_error = env.x2 - env.target - error  # 为后一时刻的error减去前一时刻的error
            if is_bad_state(error, delta_error):
                danger_rate += 1
            error = env.x2 - env.target
            # print(error, delta_error)
        rewards.append(score)
        dangers.append(danger_rate)
    return rewards, dangers


def is_bad_state(error, delta_error):  # 根据两个参数对是否为坏的状态进行判定，返回是和否
    if error > 0.5 and delta_error >= 0:  # 当前已经过热，但是还是往更热的方向走
        return True
    elif error < -0.5 and delta_error <= 0:  # 当前已经过冷，但是还是往更冷的方向去
        return True
    else:
        return False


# start = t.time()
# # env = env_for_rl()
# # agent = SAC(env)
# agent = Fuzzy_Agent()
# a, b = test_with_fixed_env(agent, mode='pure_FLC')
# print(a, b)
#
#
# finish = t.time()
# print('运行时间：'+str(finish - start))
