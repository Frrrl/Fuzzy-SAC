from CSTR_Env import env_for_comp
from SAC_Agent import SAC, config
from Validation_Test import test_with_fixed_env
import numpy as np
import torch
import os
import random

'''
用RL简单补偿的方法对控制器进行训练
'''

fo = open('.//Render_File//Training_Simple_Comp_data.txt', 'w')  # 把训练中的数据写入到txt文件当中
env = env_for_comp(mode='simple_comp')   # 默认需要用到复杂的结合policy
agent = SAC(env)

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
        next_state, reward, done_mask = env.step(action[0], 'random')
        agent.push((state, action, reward, next_state, done_mask))
        if not done_mask:  # 如果done_mask是0，就退出
            break
        if agent.buffer.buffer_len() > 1200 and step % 5 == 0:  # 每20步更新一次网络
            agent.update()
        if step % 10 == 0:  # 一个episode中每10步测试一下环境，一定要在RL开始更新之前就进行测试
            reward_test_list, danger_list = test_with_fixed_env(agent, mode='simple_comp')
            fo.write(str(reward_test_list)[1:-1]+'\n')
            fo.write(str(danger_list)[1:-1] + '\n')
            # fo.write(str(effect_list)[1:-1] + '\n')
    if episode % 10 == 9:  # 每10回合存一次agent参数
        agent.save_nets('./', episode)
    score = danger_list[0]
    print("episode:{}, buffer_len:{}, score:{}".format(episode, agent.buffer.buffer_len(), score))
fo.close()

