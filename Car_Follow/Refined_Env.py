import random
import numpy as np
from Fuzzy_Controller import Fuzzy_Agent

"""
一个用了更加精细的建模手法进行建模的任务
主要在控制力上加入了非线性因素，在风阻力上也加入了非线性因素
考虑一个后轮驱动的车辆，重心位置偏后，在后2/5处，对应了相应的后轮轮胎载荷
模型主要针对ego_car，因为它是直接控制的量，而前车则建模上偏向于简化
"""

Env_Config = {
    'T': 0.05,  # 采样时间为0.05s
    'g': 9.98,  # 重力加速度
    'mass': 1500,  # 汽车重量，单位kg
    'mass_center': 0.4,  # 汽车重心与后轮的距离比上前后轮距离，用于计算后轮载荷
    'miu': 0.02,  # 车轮收到的滚动阻力系数
    'phi': 0.9,  # 道路的附着力系数
    'brake_gain': 15000,  # 刹车带来的制动力增益
    'throttle_gain': 10000,  # 油门带来的牵引力增益
    'velo_gain': 200,  # 速度对油门产生力矩的影响
    'wind_co': 0.5,  # 空气阻力的系数， wind_force=wind_co*v^2
    'lead_car_velo': 15,  # lead_car的最大初始速度
    'lead_car_x': [5, 15],  # lead_car的初始位置区间
    'ego_car_velo': 10,  # 初始化时ego_car的最大速度
    'target_range': [5, 15],  # 保持距离的目标变换区间
    'state_high': [20, 20, 10, 5],  # 状态空间的上界包括了距离误差、己车速度、相对速度以及相对加速度
    'state_low': [-10, 0, -10, -5],  # 状态空间的下界
    'max_step': 400  # 一个回合的最大步数
}


class Ego_Car:
    def __init__(self):
        # 有关车辆的动力学参数
        self.length = 3  # 车身的长度
        self.mass = Env_Config['mass']  # 车辆的质量
        self.grav = self.mass * Env_Config['g']  # 车辆自身受到的重力
        self.Fzr = self.grav * (1 - Env_Config['mass_center'])  # 后轮载荷
        self.Rx = Env_Config['miu'] * self.grav  # 车轮的滚动阻力
        self.max_brake_force = self.grav * Env_Config['phi']  # 车辆的最大制动力
        self.max_traction_force = self.Fzr * Env_Config['phi']  # 车辆的最大牵引力
        # 有关车辆的运动学参数
        self.x, self.v, self.a = 0, 0, 0  # 小车的状态量， x代表车头所在位置
        self.brake, self.throttle = 0, 0  # 小车的控制量

        self.reset()

    def reset(self, init_paras=None):  # 重置小车的各种初始状态
        if init_paras is None:
            init_paras = [0, 0, 0]  # 默认三个参数全部为0
        self.a, self.v, self.x = init_paras[0], init_paras[1], init_paras[2]
        self.brake, self.throttle = 0, 0  # 动作也置零

    def step(self, action):
        if action >= 0:
            self.throttle, self.brake = action, 0
        else:
            self.throttle, self.brake = 0, -action

        # 计算各种横向力
        Rolling_resist = Env_Config['miu'] * self.grav if self.v >= 0 else 0  # 滚动阻力
        wind_resist = Env_Config['wind_co'] * self.v ** 2  # 风阻力
        braking_force = np.clip(self.brake * Env_Config['brake_gain'], 0, self.max_brake_force)  # 制动力
        traction_force = np.clip(self.throttle * Env_Config['throttle_gain'] - self.v * Env_Config['velo_gain'],
                                 0, self.max_traction_force)  # 牵引力
        Force = traction_force - braking_force - Rolling_resist - wind_resist

        # 计算在力的影响下改变的加速度、速度和距离等变量
        self.a = Force / self.mass
        if self.a < 0 and self.v == 0:
            self.a = 0  # 在停车时向后的各种力全部消失
        self.x += self.v * Env_Config['T']  # 改变小车位移
        self.v = np.clip(self.v + self.a * Env_Config['T'], 0, None)  # 改变v，并将v进行限制，不能小于0


class Lead_Car:  # 比较粗略的模型，因为不需要控制前车，所以不考虑其运动学模型
    def __init__(self):
        self.length = 3  # 小车长度(m)
        self.a, self.v, self.x = 0, 0, 0  # 分别代表小车的加速度和速度以及位移，状态量
        self.reset()  # 初始化的时候就顺便重置了

    def reset(self, init_paras=None):  # 重置小车各种状态，主要需要改变小车位置
        if init_paras is None:
            init_paras = [0, 10, 10]
        self.a, self.v, self.x = init_paras[0], init_paras[1], init_paras[2]  # 三个状态设置为paras

    def step(self, action=None):  # action直接改变加速度，但是没有阻力和质量那一套运算了
        # 计算在力的影响下改变的加速度
        if action:
            self.a = action
        self.x += self.v * Env_Config['T']  # 改变小车位移
        self.v = np.clip(self.v + self.a * Env_Config['T'], 0, None)  # 改变v，并将v进行限制，不能小于0


# 用于给RL和FLC进行训练和测试的环境
class env_for_rl:  # 包含两个小车，并以ego_car为原点，观察lead_car相对于ego_car的速度、距离和加速度
    def __init__(self):
        self.ego_car, self.lead_car = Ego_Car(), Lead_Car()
        self.target_dist = 10
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length
        self.state_high = Env_Config['state_high']  # 距离误差、相对速度、相对加速度
        self.state_low = Env_Config['state_low']  #
        self.action_high = [1]
        self.action = 0  # 用于计算delta_action，特指施加在ego_car上的动作
        self.N_Counter = 0  # 用于计算当前的回合步数
        self.reset()

    def reset(self, paras=None):
        _paras_ = self.set_init_paras(user_config=paras)
        self.lead_car.reset(_paras_[0])
        self.ego_car.reset(_paras_[1])
        self.target_dist = _paras_[2]
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length
        self.action = 0
        self.N_Counter = 0
        return self.get_state()

    def step(self, action1, action2=None):
        acc_prev = self.ego_car.a  # 记录己车前一时刻的acc
        self.ego_car.step(action1)
        self.lead_car.step(action2)
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length
        acc_now = self.ego_car.a  # 记录此时的acc
        dist_error = self.dist - self.target_dist  # 距离的误差
        delta_action = action1 - self.action
        delta_acc = acc_now - acc_prev  # 加速度变化
        self.action = action1

        reward = give_reward(self.dist, dist_error, delta_action, delta_acc)
        next_state = self.get_state()
        done_mask = 1
        self.N_Counter += 1
        if self.dist <= 0 or self.N_Counter == Env_Config['max_step']:
            done_mask = 0
        return next_state, reward, done_mask

    def get_state(self):  # 根据observe，转化为相应的neural
        dist_error = self.dist - self.target_dist  # 与目标距离的差值，正数表示远了
        self_velo = self.ego_car.v  # 己车的速度值，用于计算踩多少油门合适
        velo_rela = self.lead_car.v - self.ego_car.v  # 前车相对速度，正数表示前车在远离
        acc_rela = self.lead_car.a - self.ego_car.a  # 前车相对加速度，正数表示前车有加速趋势
        raw_state = [dist_error, self_velo, velo_rela, acc_rela]
        state_mean = [(self.state_high[i] + self.state_low[i]) / 2 for i in range(len(self.state_high))]
        state_bias = [(self.state_high[i] - self.state_low[i]) / 2 for i in range(len(self.state_high))]
        states = [(raw_state[i] - state_mean[i]) / state_bias[i] for i in range(len(self.state_high))]
        return np.array(states, dtype=np.float32)

    @staticmethod
    # 配置环境的初始状态
    def set_init_paras(user_config=None):
        if user_config is None:
            return [[0, 10, 10], [0, 0, 0], 10]  # 默认的参数
        if user_config is 'random':  # 代表训练时进行的随机性比较大的reset
            lead_acc = 0  # 给前车设定加速度为0也无所谓，反正前车可以自由改变加速度
            lead_velo = random.uniform(0, Env_Config['lead_car_velo'])
            lead_x = random.uniform(*Env_Config['lead_car_x'])
            ego_acc = 0
            ego_velo = random.uniform(0, Env_Config['ego_car_velo'])
            ego_x = 0
            target_dist = random.uniform(*Env_Config['target_range'])
            return [[lead_acc, lead_velo, lead_x], [ego_acc, ego_velo, ego_x], target_dist]
        if user_config is 'test':  # 代表进行测试时进行的随机性比较小的reset
            lead_acc = 0
            ego_acc = 0
            ego_x = 0
            lead_velo = random.uniform(3, 5)  # 前车的起始速度
            lead_x = random.uniform(6, 7)  # 前车的起始位置
            ego_velo = random.uniform(1, 2)  # 后车的起始速度
            target_dist = random.uniform(5, 6)  # 目标距离的范围
            return [[lead_acc, lead_velo, lead_x], [ego_acc, ego_velo, ego_x], target_dist]

        return user_config


class env_for_comp:  # 为RL补偿强化学习而设计的环境，并且设计一下根据不同的模式选择combine_policy
    def __init__(self, mode='complex_comp'):  # 默认使用AC矫正器
        self.lead_car, self.ego_car = Lead_Car(), Ego_Car()
        self.target_dist = 10
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length
        self.FLC = Fuzzy_Agent()  # 给自己加一个FLC控制器
        self.state_high = [*Env_Config['state_high'], 1]  # RL的基础上加上一个FLC的动作
        self.state_low = [*Env_Config['state_low'], -1]
        self.action_high = [1]  # action限制为-1~1
        self.combine_policy = mode  # 决定融合策略的效果
        self.action = 0  # 特指传给ego_car的action，为了给出reward做出准备
        self.FLC_Action = 0  # 特指FLC给出的Action，避免了二次计算
        self.N_Counter = 0  # 计算当前的步数
        self.reset()

    def reset(self, paras=None):
        _paras_ = self.set_init_paras(user_config=paras)
        self.lead_car.reset(_paras_[0])
        self.ego_car.reset(_paras_[1])
        self.target_dist = _paras_[2]
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length
        self.action = 0
        self.FLC_Action = 0
        self.N_Counter = 0
        return self.get_state()

    def step(self, action1, action2=None):
        acc_prev = self.ego_car.a  # 记录前一时刻的acc
        self.FLC_Action = self.FLC.get_action(self.get_state()[:-1])[0]
        Real_action = np.clip(self.action_correct(self.FLC_Action, action1), -1, 1)

        self.ego_car.step(Real_action)
        self.lead_car.step(action2)
        self.dist = self.lead_car.x - self.ego_car.x - self.lead_car.length

        acc_now = self.ego_car.a  # 记录此时的acc
        dist_error = self.dist - self.target_dist  # 距离的误差
        delta_action = action1 - self.action
        delta_acc = acc_now - acc_prev  # 加速度变化
        self.action = action1

        reward = give_reward(self.dist, dist_error, delta_action, delta_acc)
        next_state = self.get_state()
        done_mask = 1
        self.N_Counter += 1
        if self.dist <= 0 or self.N_Counter == Env_Config['max_step']:  # 撞车了或者步数到达最后一步了
            done_mask = 0
        return next_state, reward, done_mask

    def get_state(self):  # 根据observe，转化为相应的neural
        dist_error = self.dist - self.target_dist  # 与目标距离的差值，正数表示远了
        self_velo = self.ego_car.v  # 己车的速度值，用于计算踩多少油门合适
        velo_rela = self.lead_car.v - self.ego_car.v  # 前车相对速度，正数表示前车在远离
        acc_rela = self.lead_car.a - self.ego_car.a  # 前车相对加速度，正数表示前车有加速趋势
        raw_state = [dist_error, self_velo, velo_rela, acc_rela, self.FLC_Action]
        state_mean = [(self.state_high[i] + self.state_low[i]) / 2 for i in range(len(self.state_high))]
        state_bias = [(self.state_high[i] - self.state_low[i]) / 2 for i in range(len(self.state_high))]
        states = [(raw_state[i] - state_mean[i]) / state_bias[i] for i in range(len(self.state_high))]
        return np.array(states, dtype=np.float32)

    def action_correct(self, u1t, at):  # 根据st的情况，给出一个两种控制器的按比例输出结果
        if self.combine_policy == 'simple_comp':
            return u1t + at
        else:
            alpha = self.dist / 20  # 最大的两车距离可以认为差不多是20，即使多了也无所谓
            ut = u1t + alpha * at
            return ut

    @staticmethod
    def set_init_paras(user_config=None):  # 配置环境的初始状态
        if user_config is None:
            return [[0, 10, 10], [0, 0, 0], 10]  # 默认的参数
        if user_config is 'random':
            lead_acc = 0  # 给前车设定加速度为0也无所谓，反正前车可以自由改变加速度
            lead_velo = random.uniform(0, Env_Config['lead_car_velo'])
            lead_x = random.uniform(*Env_Config['lead_car_x'])
            ego_acc = 0
            ego_velo = random.uniform(0, Env_Config['ego_car_velo'])
            ego_x = 0
            target_dist = random.uniform(*Env_Config['target_range'])
            return [[lead_acc, lead_velo, lead_x], [ego_acc, ego_velo, ego_x], target_dist]
        if user_config is 'test':  # 代表进行测试时进行的随机性比较小的reset
            lead_acc = 0
            ego_acc = 0
            ego_x = 0
            lead_velo = random.uniform(3, 5)  # 前车的起始速度
            lead_x = random.uniform(6, 7)  # 前车的起始位置
            ego_velo = random.uniform(1, 2)  # 后车的起始速度
            target_dist = random.uniform(5, 6)  # 目标距离的范围
            return [[lead_acc, lead_velo, lead_x], [ego_acc, ego_velo, ego_x], target_dist]
        return user_config


# 根据step进行调整的lead car的运行策略，以总共400个step进行计算，并且在train和test时提供不同随机度的policy
def lead_policy(step, max_step, mode='train'):
    # 总共包含了四种变化，lead_car在一个episode中总共需要给出4种不同的策略
    if step == 0:
        return 0
    elif step % (max_step / 4) == 0:
        return random.uniform(-1, 1) if mode == 'train' else random.uniform(-0.5, 0.5)
    else:
        return None



# 一个给予reward的函数，其度量的方面包含了距离误差（控制的精准）、动作变化（对机械结构的损伤）以及加速度变化（乘客体验的变化），并且如果碰撞的话就给很大的负奖励
def give_reward(dist, dist_error, delta_action, delta_acc):
    reward_for_dist = -abs(dist_error)
    danger_penalty = np.clip(-1 / dist, -50, 0) if dist < 1 else 0
    collision_penalty = -500 if dist <= 0 else 0
    # reward_for_action_change = -10 * abs(delta_action)
    # reward_for_acc_change = -5 * abs(delta_acc)
    # reward_all = reward_for_dist + reward_for_action_change + reward_for_acc_change
    return reward_for_dist + danger_penalty + collision_penalty
