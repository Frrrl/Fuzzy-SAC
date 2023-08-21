import math
import random
import numpy as np
from Fuzzy_Controller import Fuzzy_Agent
import matplotlib.pyplot as plt

"""
一个CSTR连续搅拌反应釜的环境建模与仿真环境
输入量为反应釜的物料浓度Ca和温度T，控制量为冷却剂温度Tc，控制量为反应釜的温度T
其中x1代表Ca-Caeq, x2代表T-Teq, u代表Tc-Tceq, 输出y代表T-Teq
"""

EnvConfig = {
    'Caeq': 0.5,  # 稳态工作点时的釜内物料浓度，单位mol/L
    'Teq': 350,  # 稳态时釜内物料温度，单位K
    'Tceq': 338,  # 稳态时加入的冷却剂温度，单位K
    'Ts': 0.05,  # 系统采样时间，单位为s
    'action_gain': 10,  # 输入动作的神经元在实际环境中被放大的倍数
    'max_step': 200,  # 一回合的最大持续时间，也就是10s
    'min_threshold': 0.5,  # 最小的误差值
    'max_threshold': 1  # 最大的误差值
}


class env_for_rl:
    def __init__(self):
        # 一些不变的量
        self.Ts = EnvConfig['Ts']  # 系统的采样时间
        self.Ca = 1  # 釜内反应物的浓度，初始值选择进入的物料的浓度
        self.T = 350  # 釜内物料的浓度，初始值选择进料的温度
        self.state_high = [0.5, 5, 5]
        self.action_high = [EnvConfig['action_gain']]
        # 一些变化的量
        self.x1 = 0.5  # 选取第一个状态量=Ca-Caeq
        self.x2 = 0  # 选取第二个状态量=T-Teq
        self.target = 0  # 即需要控制的温度目标与350K的差值, 即Tref - Teq
        self.N_Counter = 0  # 计步器
        self.reset('random')

    def reset(self, paras=None):  # paras包含了对x1/x2和target的重置
        _paras_ = self.set_init_paras(user_config=paras)
        self.x1 = _paras_[0]  # 重置第一个状态量
        self.x2 = _paras_[1]  # 重置第二个状态量
        self.target = _paras_[2]  # 重置目标
        self.N_Counter = 0
        return self.get_state()

    def step(self, action, targets=None):  # 给定了u的情况下，系统如何变化
        if self.N_Counter % (EnvConfig['max_step']//4) == 0 and targets:  # 有targets参数的情况下才换目标
            if targets == 'random':
                self.target = random.uniform(-5, 5)
            elif targets == 'test':
                self.target = random.uniform(-3, 3)
            else:
                self.target = targets[self.N_Counter//(EnvConfig['max_step']//4)]
        u = action * self.action_high[0]  # 实际传输给系统的u，也就是Tc-Tceq
        delta_x1 = 0.5 - self.x1 - 7.2 * 10**10 * math.exp(-8750 / (self.x2 + 350)) * (self.x1 + 0.5)
        self.x1 += delta_x1 * self.Ts
        delta_x2 = -self.x2 + 8.63 / 239 * 10**14 * math.exp(-8750 / (self.x2 + 350)) * (self.x1 + 0.5) + 500/239 * (u - self.x2 - 12)
        self.x2 += delta_x2 * self.Ts

        next_state = self.get_state()
        reward = -abs(self.x2 - self.target)
        done_mask = 1
        self.N_Counter += 1
        if self.N_Counter == EnvConfig['max_step']:
            done_mask = 0
        return next_state, reward, done_mask

    def get_state(self):
        Ca_rela = self.x1 / self.state_high[0]
        T_rela = self.x2 / self.state_high[1]
        Tar_rela = self.target / self.state_high[2]
        return np.array([Ca_rela, T_rela, Tar_rela], dtype=np.float32)

    @staticmethod
    def set_init_paras(user_config=None):  # 配置环境的初始状态
        if user_config is None:
            return [0, 0, 0]  # 默认的参数
        if user_config is 'random':
            x1 = random.uniform(-0.5, 0.5)
            x2 = random.uniform(-5, 5)
            target = random.uniform(-5, 5)
            return [x1, x2, target]
        if user_config is 'test':  # 代表进行测试时进行的随机性比较小的reset
            x1 = random.uniform(-0.2, 0.2)
            x2 = random.uniform(-3, 3)
            target = random.uniform(-3, 3)
            return [x1, x2, target]
        return user_config


class env_for_comp:  # 主要的变化是加入了一个FLC的动作作为状态空间
    def __init__(self, mode='complex_comp'):
        # 一些一个回合中不会变的量
        self.Ts = EnvConfig['Ts']  # 系统的采样时间
        self.Ca = 1  # 釜内反应物的浓度，初始值选择进入的物料的浓度
        self.T = 350  # 釜内物料的浓度，初始值选择进料的温度
        self.state_high = [0.5, 5, 5, 1]
        self.action_high = [EnvConfig['action_gain']]
        self.combine_policy = mode  # 决定融合策略的效果
        self.FLC = Fuzzy_Agent()  # 给自己加一个FLC控制器
        self.target = 0  # 即需要控制的温度目标与350K的差值, 即Tref - Teq
        # 一些持续变化的量
        self.x1 = 0.5  # 选取第一个状态量
        self.x2 = 0  # 选取第二个状态量
        self.FLC_Action = 0  # 特指FLC给出的Action，避免了二次计算
        self.N_Counter = 0  # 计步器
        self.reset('random')

    def reset(self, paras=None):  # paras包含了对x1/x2和target的重置
        _paras_ = self.set_init_paras(user_config=paras)
        self.x1 = _paras_[0]  # 重置第一个状态量
        self.x2 = _paras_[1]  # 重置第二个状态量
        self.target = _paras_[2]  # 重置目标
        self.N_Counter = 0
        self.FLC_Action = 0
        return self.get_state()

    def step(self, action, targets=None):  # 给定了u的情况下，系统如何变化，如果没有给targets，那就直接不换目标
        # 重置一下系统的target
        if self.N_Counter % (EnvConfig['max_step']//4) == 0 and targets:  # 有targets的情况下才换目标
            if targets == 'random':
                self.target = random.uniform(-5, 5)
            elif targets == 'test':
                self.target = random.uniform(-3, 3)
            else:
                self.target = targets[self.N_Counter//(EnvConfig['max_step']//4)]
        self.FLC_Action = self.FLC.get_action(self.get_state()[:-1])[0]
        Final_action = self.action_fusion(self.FLC_Action, action)  # 策略融合，此时已经将范围限制在了[-1~1]之间

        u = Final_action * self.action_high[0]  # 实际传输给系统的u，也就是Tc-Tceq
        delta_x1 = 0.5 - self.x1 - 7.2 * 10 ** 10 * math.exp(-8750 / (self.x2 + 350)) * (self.x1 + 0.5)
        self.x1 += delta_x1 * self.Ts
        delta_x2 = -self.x2 + 8.63 / 239 * 10 ** 14 * math.exp(-8750 / (self.x2 + 350)) * (
                    self.x1 + 0.5) + 500 / 239 * (u - self.x2 - 12)
        self.x2 += delta_x2 * self.Ts

        next_state = self.get_state()
        reward = -abs(self.x2 - self.target)
        done_mask = 1
        self.N_Counter += 1
        if self.N_Counter == EnvConfig['max_step']:
            done_mask = 0
        return next_state, reward, done_mask

    def get_state(self):
        Ca_rela = self.x1 / self.state_high[0]
        T_rela = self.x2 / self.state_high[1]
        Tar_rela = self.target / self.state_high[2]
        return np.array([Ca_rela, T_rela, Tar_rela, self.FLC_Action], dtype=np.float32)

    def action_fusion(self, u1t, at):  # 根据st的情况，给出一个两种控制器的按比例输出结果
        if self.combine_policy == 'simple_comp':
            final_act = np.clip(u1t + at, -1, 1)
        else:
            error = abs(self.x2 - self.target)
            if error <= EnvConfig['min_threshold']:
                cf = 0
            elif error >= EnvConfig['max_threshold']:
                cf = 1
            else:
                cf = (error - EnvConfig['min_threshold']) / (EnvConfig['max_threshold'] - EnvConfig['min_threshold'])
            ut = u1t + (1 - cf) * at  # 差得越大，使用FLC的成分就越多
            final_act = np.clip(ut, -1, 1)
        return final_act

    @staticmethod
    def set_init_paras(user_config=None):  # 配置环境的初始状态
        if user_config is None:
            return [0, 0, 0]  # 默认的参数
        if user_config is 'random':
            x1 = random.uniform(-0.5, 0.5)
            x2 = random.uniform(-5, 5)
            target = random.uniform(-5, 5)
            return [x1, x2, target]
        if user_config is 'test':  # 代表进行测试时进行的随机性比较小的reset
            x1 = random.uniform(-0.2, 0.2)
            x2 = random.uniform(-3, 3)
            target = random.uniform(-3, 3)
            return [x1, x2, target]
        return user_config
