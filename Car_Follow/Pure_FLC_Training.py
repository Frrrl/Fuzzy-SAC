from Refined_Env import env_for_rl, lead_policy
from SAC_Agent import config
from Validation_Test import test_with_fixed_env
from Fuzzy_Controller import Fuzzy_Agent

"""
一个文件，观察FLC在200个回合中的平均表现
"""

fo = open('.//Render_File//Training_FLC_data.txt', 'w')  # 把训练中的数据写入到txt文件当中

env = env_for_rl()
agent = Fuzzy_Agent()

for episode in range(config['max_episode']):
    env.reset('random')
    for step in range(config['max_step']):
        if step % 20 == 0:  # 一个episode中每20步测试一下环境，一定要在RL开始更新之前就进行测试
            reward_test_list, danger_list = test_with_fixed_env(agent, mode='pure_FLC')
            fo.write(str(reward_test_list)[1:-1]+'\n')
            fo.write(str(danger_list)[1:-1] + '\n')
    score = reward_test_list[0]
    print("episode:{}, score:{}".format(episode, score))

fo.close()
