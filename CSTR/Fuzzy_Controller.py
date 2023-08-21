import numpy as np
import skfuzzy as fuzz

"""
针对CSTR问题设计的两输入一输出的TS-FLC模块
输入为Ca_rela[-0.5, 0.5], T_rela[-10~10]， 以及Tar_rela[-5~5]
输出为action[-1~1]
"""
# 因为在参数设置上有诸多不确定因素，所以先在此处设置一些超参数

max_Ca_rela = [-0.5, 0.5]  # 与工作点的相对浓度范围
max_T_rela = [-5, 5]  # 与工作点的相对温度范围
max_Tar_rela = [-5, 5]  # 目标值与工作点的相对温度范围
max_target_rela = [max_T_rela[0]+max_Tar_rela[0], max_T_rela[1]+max_Tar_rela[1]]

state_high = [max_Ca_rela[1], max_T_rela[1], max_Tar_rela[1]]
state_low = [max_Ca_rela[0], max_T_rela[0], max_Tar_rela[0]]

bestpara = [0, -1, 1, -0.4, 0.4]


class Fuzzy_Agent:
    def get_action(self, state):  # 输入和SAC相同的neural值，给出应当输出的动作
        state_mean = [(state_high[i] + state_low[i]) / 2 for i in range(len(state_high))]
        state_bias = [(state_high[i] - state_low[i]) / 2 for i in range(len(state_high))]
        states = [state[i] * state_bias[i] + state_mean[i] for i in range(len(state_high))]
        Ca_rela = states[0]
        T_rela = states[1]
        Tar_rela = states[2]

        # 将输入数据都限制在一定范围内，防止出现output出问题的情况
        Ca_within = np.clip(Ca_rela, 0.1 + max_Ca_rela[0], max_Ca_rela[1] - 0.1)
        target_rela = T_rela - Tar_rela  # 代表当前的温度与目标温度之间的差值
        target_rela_within = np.clip(target_rela, 0.1 + max_target_rela[0], max_target_rela[1] - 0.1)

        action = TS_compute(Ca_within, target_rela_within)
        return [action]

    def get_action_test(self, state):  # 主要为了和SAC智能体有相同的接口
        return self.get_action(state)


def TS_compute(ca, tar):  # 根据para的数值设置一种TS模糊推理的方法
    # <editor-fold desc="隶属度函数的设置">
    x_ca_range = np.arange(max_Ca_rela[0], max_Ca_rela[1], 0.1, np.float32)
    x_tar_range = np.arange(max_target_rela[0], max_target_rela[1], 0.1, np.float32)

    # ca_PB = fuzz.trapmf(x_ca_range, [0.2 * max_Ca_rela[1], 0.5 * max_Ca_rela[1], max_Ca_rela[1], max_Ca_rela[1]])  # 距离误差为正大，即靠的太远了
    # ca_PS = fuzz.trapmf(x_ca_range, [0, 0, 0.2 * max_Ca_rela[1], 0.5 * max_Ca_rela[1]])  # 距离误差为正小，即靠的有点远
    # ca_NS = fuzz.trapmf(x_ca_range, [0.5 * max_Ca_rela[0], 0.2 * max_Ca_rela[0], 0, 0])  # 距离误差为负小，即靠的有点近
    # ca_NB = fuzz.trapmf(x_ca_range, [max_Ca_rela[0], max_Ca_rela[0], 0.5 * max_Ca_rela[0], 0.2 * max_Ca_rela[0]])  # 距离误差为负大，即靠的太近了

    tar_PB = fuzz.trapmf(x_tar_range, [0.2 * max_target_rela[1], 0.5 * max_target_rela[1], max_target_rela[1], max_target_rela[1]])  # 相对加速度正大，即前车相对加速度太快
    tar_PS = fuzz.trapmf(x_tar_range, [0.05 * max_target_rela[0], 0, 0.2 * max_target_rela[1], 0.5 * max_target_rela[1]])  # 相对加速度正小，即前车相对加速度有点快
    tar_NS = fuzz.trapmf(x_tar_range, [0.5 * max_target_rela[0], 0.2 * max_target_rela[0], 0, 0.05 * max_target_rela[1]])  # 相对加速度负小，即前车相对加速度有点慢
    tar_NB = fuzz.trapmf(x_tar_range, [max_target_rela[0], max_target_rela[0], 0.5 * max_target_rela[0], 0.2 * max_target_rela[0]])  # 相对加速度负大，即前车相对加速度太慢
    # </editor-fold>

    # <editor-fold desc="计算各个隶属度">
    # c_pb = fuzz.interp_membership(x_ca_range, ca_PB, ca)
    # c_ps = fuzz.interp_membership(x_ca_range, ca_PS, ca)
    # c_ns = fuzz.interp_membership(x_ca_range, ca_NS, ca)
    # c_nb = fuzz.interp_membership(x_ca_range, ca_NB, ca)

    t_pb = fuzz.interp_membership(x_tar_range, tar_PB, tar)
    t_ps = fuzz.interp_membership(x_tar_range, tar_PS, tar)
    t_ns = fuzz.interp_membership(x_tar_range, tar_NS, tar)
    t_nb = fuzz.interp_membership(x_tar_range, tar_NB, tar)
    # </editor-fold>

    # <editor-fold desc="计算各个规则的激活函数">
    '第1种情况，当前温度比目标高很多，规则就是冷却剂降温'
    activation1 = t_pb
    output1 = bestpara[1]  # -1

    '第2种情况，当前温度比目标低很多，规则就是冷却剂升温'
    activation2 = t_nb
    output2 = bestpara[2]  # 1

    '第3种情况，当前温度比目标高一些，规则就是冷却剂降一点温'
    activation3 = t_ps
    output3 = bestpara[3]  # -0.4

    '第4种情况，当前温度比目标低一些，规则就是冷却剂升一点温'
    activation4 = t_ns
    output4 = bestpara[4]  # 0.4

    # </editor-fold>

    # <editor-fold desc="给出最终的结果">
    activation = [activation1, activation2, activation3, activation4]
    output = [output1, output2, output3, output4]
    num = sum([a * b for a, b in zip(activation, output)])  # 分子
    den = sum(activation) + 1e-12   # 分母，为了防止出现枫木等于0的情况，加上一个非常小的正数
    combined_output = num / den
    final_output = np.clip(combined_output, -1, 1)  # 最后使得输出在[-1~1]的区间内
    # </editor-fold>
    return final_output