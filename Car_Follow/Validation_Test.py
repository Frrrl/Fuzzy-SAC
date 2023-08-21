from SAC_Agent import config, SAC
from Refined_Env import env_for_rl, env_for_comp, lead_policy
# import time as t
# from Fuzzy_Controller import Fuzzy_Agent
"""
单纯给出10个回合下模型的reward输出，便于观察训练情况
每次调用都返回一个列表
在这里显示FLC的推理速度不如RL
"""

test_repeats = 7


def test_with_fixed_env(agent, mode='complex_comp'):
    if mode == 'pure_RL' or mode == 'pure_FLC':
        env = env_for_rl()
    else:
        env = env_for_comp(mode)
    rewards, dangers = [], []
    for _ in range(test_repeats):
        env.reset('test')
        score = 0
        danger_rate = 0
        for step in range(config['max_step']):
            '以下为正常一个step该做的动作'
            state = env.get_state()
            action = agent.get_action_test(state)
            lead_action = lead_policy(step, config['max_step'], mode='test')
            reward, done_mask = env.step(action[0], lead_action)[1:3]
            score += reward
            if not done_mask and step < config['max_step']-1:  # 如果碰撞了，那就直接给负奖励
                danger_rate += 50
                break
            if is_bad_state(env.target_dist, env.dist-env.target_dist,
                            env.lead_car.v-env.ego_car.v, env.lead_car.a-env.ego_car.a):
                danger_rate += 1
        rewards.append(score)
        dangers.append(danger_rate)
    return rewards, dangers


def is_bad_state(target, error, velo_rela, acc_rela):  # 根据四个参数对是否为坏的状态进行判定，返回是和否
    if error > 0.3 * target and acc_rela > 0:  # acc_rela代表前车减去后车
        return True
    elif error < -0.3 * target and velo_rela < 0:
        return True
    else:
        return False


# start = t.time()
# agent = Fuzzy_Agent()
# a, b = test_with_fixed_env(agent, mode='pure_FLC')
# print(a, b)
# finish = t.time()
# print('运行时间：'+str(finish - start))
