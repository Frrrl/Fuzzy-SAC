from Refined_Env import env_for_rl, lead_policy
from SAC_Agent import SAC, config
from Validation_Test import test_with_fixed_env
import numpy as np
import torch
import os
import random

"""
用纯强化学习方法训练ego_car的策略
"""

fo = open('.//Render_File//Training_RL_data.txt', 'w')  # 把训练中的数据写入到txt文件当中

env = env_for_rl()
agent = SAC(env)
# agent.load_nets('./', 39)  # 让Agent读取模型
# <editor-fold desc="多线程和随机参数的选项">
torch.set_num_threads(config['num_threads'])
os.environ['MKL_NUM_THREADS'] = str(config['num_threads'])
if not config['random_seed']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
# </editor-fold>

for episode in range(config['max_episode']):
    env.reset('random')
    for step in range(config['max_step']):
        state = env.get_state()
        action = agent.get_action(state)
        lead_action = lead_policy(step, config['max_step'])
        next_state, reward, done_mask = env.step(action[0], lead_action)
        agent.push((state, action, reward, next_state, done_mask))
        if not done_mask:  # 如果done_mask是0，就退出
            break
        if agent.buffer.buffer_len() > 1000 and step % 2 == 0:  # 每2步训练一次
            agent.update()
        if step % 20 == 0:  # 一个episode中每20步测试一下环境，一定要在RL开始更新之前就进行测试
            reward_test_list, danger_list = test_with_fixed_env(agent, mode='pure_RL')
            fo.write(str(reward_test_list)[1:-1]+'\n')
            fo.write(str(danger_list)[1:-1] + '\n')
    if episode % 10 == 7:  # 每10回合存一次agent参数
        agent.save_nets('./', episode)
    score = 0 if agent.buffer.buffer_len() < 50 else reward_test_list[0]
    print("episode:{}, buffer_len:{}, score:{}".format(episode, agent.buffer.buffer_len(), score))

fo.close()
