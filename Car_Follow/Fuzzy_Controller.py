import numpy as np
import skfuzzy as fuzz

"""
针对跟车问题设计的两输入一输出的TS-FLC模块
输入为dist_error[-10, 10], velo_rela[-10~10]
输出为action[-1~1]
"""
# 因为在参数设置上有诸多不确定因素，所以先在此处设置一些超参数

max_dist_error = [-10, 20]  # 最大的距离误差
max_self_velo = [0, 20]  # 己车速度的最大值
max_rela_velo = [-10, 10]  # 相对速度的最大值
max_rela_acc = [-5, 5]  # 相对加速度的最大值
state_high = [max_dist_error[1], max_self_velo[1], max_rela_velo[1], max_rela_acc[1]]
state_low = [max_dist_error[0], max_self_velo[0], max_rela_velo[0], max_rela_acc[0]]

bestpara = [0, 1, -1, 0.8, 0.5, 0.5, 0.2, 0.2, 0.1, -0.5, -0.2]


class Fuzzy_Agent:
    def get_action(self, state):  # 输入和SAC相同的neural值，给出应当输出的动作
        state_mean = [(state_high[i] + state_low[i]) / 2 for i in range(len(state_high))]
        state_bias = [(state_high[i] - state_low[i]) / 2 for i in range(len(state_high))]
        states = [state[i] * state_bias[i] + state_mean[i] for i in range(len(state_high))]
        dist_error = states[0]
        self_velo = states[1]
        rela_velo = states[2]
        rela_acc = states[3]
        # 将输入数据都限制在一定范围内，防止出现output出问题的情况
        dist_error_within = np.clip(dist_error, 0.1 + max_dist_error[0], max_dist_error[1] - 0.1)
        self_dist_within = np.clip(self_velo, 0.1 + max_self_velo[0], max_self_velo[1] - 0.1)
        rela_velo_within = np.clip(rela_velo, 0.1 + max_rela_velo[0], max_rela_velo[1] - 0.1)
        acc_rela_within = np.clip(rela_acc, 0.1 + max_rela_acc[0], max_rela_acc[1] - 0.1)

        action = TS_compute(dist_error_within, self_dist_within, rela_velo_within, acc_rela_within)
        return [action]

    def get_action_test(self, state):  # 主要为了和SAC智能体有相同的接口
        return self.get_action(state)


def TS_compute(dist_error, self_v, r_velo, r_acc):  # 根据para的数值设置一种TS模糊推理的方法
    # <editor-fold desc="隶属度函数的设置">
    x_dist_range = np.arange(max_dist_error[0], max_dist_error[1], 0.1, np.float32)
    x_self_v_range = np.arange(max_self_velo[0], max_self_velo[1], 0.1, np.float32)
    x_velo_range = np.arange(max_rela_velo[0], max_rela_velo[1], 0.1, np.float32)
    x_acc_range = np.arange(max_rela_acc[0], max_rela_acc[1], 0.1, np.float32)

    dist_PB = fuzz.trapmf(x_dist_range, [0.2 * max_dist_error[1], 0.5 * max_dist_error[1], max_dist_error[1], max_dist_error[1]])  # 距离误差为正大，即靠的太远了
    dist_PS = fuzz.trapmf(x_dist_range, [0, 0, 0.2 * max_dist_error[1], 0.5 * max_dist_error[1]])  # 距离误差为正小，即靠的有点远
    dist_NS = fuzz.trapmf(x_dist_range, [0.5 * max_dist_error[0], 0.2 * max_dist_error[0], 0, 0])  # 距离误差为负小，即靠的有点近
    dist_NB = fuzz.trapmf(x_dist_range, [max_dist_error[0], max_dist_error[0], 0.5 * max_dist_error[0], 0.2 * max_dist_error[0]])  # 距离误差为负大，即靠的太近了

    self_v_B = fuzz.trapmf(x_self_v_range, [0.1 * max_self_velo[1], 0.75 * max_self_velo[1], max_self_velo[1], max_self_velo[1]])  # 己车速度为大
    self_v_S = fuzz.trapmf(x_self_v_range, [max_self_velo[0], max_self_velo[0], 0.1 * max_self_velo[1], 0.5 * max_self_velo[1]])  # 己车速度为小

    velo_PB = fuzz.trapmf(x_velo_range, [0.1 * max_rela_velo[1], 0.2 * max_rela_velo[1], max_rela_velo[1], max_rela_velo[1]])  # 相对速度正大，即前车速度太快
    velo_PS = fuzz.trapmf(x_velo_range, [0, 0, 0.1 * max_rela_velo[1], 0.2 * max_rela_velo[1]])  # 相对速度正小，即前车速度有点快
    velo_NS = fuzz.trapmf(x_velo_range, [0.2 * max_rela_velo[0], 0.1 * max_rela_velo[0], 0, 0])  # 相对速度负小，即前车速度有点慢
    velo_NB = fuzz.trapmf(x_velo_range, [max_rela_velo[0], max_rela_velo[0], 0.2 * max_rela_velo[0], 0.1 * max_rela_velo[0]])  # 相对速度负大，即前车速度太慢

    acc_PB = fuzz.trapmf(x_acc_range, [0.1 * max_rela_acc[1], 0.2 * max_rela_acc[1], max_rela_acc[1], max_rela_acc[1]])  # 相对加速度正大，即前车相对加速度太快
    acc_PS = fuzz.trapmf(x_acc_range, [0, 0, 0.1 * max_rela_acc[1], 0.2 * max_rela_acc[1]])  # 相对加速度正小，即前车相对加速度有点快
    acc_NS = fuzz.trapmf(x_acc_range, [0.2 * max_rela_acc[0], 0.1 * max_rela_acc[0], 0, 0])  # 相对加速度负小，即前车相对加速度有点慢
    acc_NB = fuzz.trapmf(x_acc_range, [max_rela_acc[0], max_rela_acc[0], 0.2 * max_rela_acc[0], 0.1 * max_rela_acc[0]])  # 相对加速度负大，即前车相对加速度太慢
    # </editor-fold>

    # <editor-fold desc="计算各个隶属度">
    d_pb = fuzz.interp_membership(x_dist_range, dist_PB, dist_error)
    d_ps = fuzz.interp_membership(x_dist_range, dist_PS, dist_error)
    d_ns = fuzz.interp_membership(x_dist_range, dist_NS, dist_error)
    d_nb = fuzz.interp_membership(x_dist_range, dist_NB, dist_error)

    sv_b = fuzz.interp_membership(x_self_v_range, self_v_B, self_v)
    sv_s = fuzz.interp_membership(x_self_v_range, self_v_S, self_v)

    v_pb = fuzz.interp_membership(x_velo_range, velo_PB, r_velo)
    v_ps = fuzz.interp_membership(x_velo_range, velo_PS, r_velo)
    v_ns = fuzz.interp_membership(x_velo_range, velo_NS, r_velo)
    v_nb = fuzz.interp_membership(x_velo_range, velo_NB, r_velo)

    a_pb = fuzz.interp_membership(x_acc_range, acc_PB, r_acc)
    a_ps = fuzz.interp_membership(x_acc_range, acc_PS, r_acc)
    a_ns = fuzz.interp_membership(x_acc_range, acc_NS, r_acc)
    a_nb = fuzz.interp_membership(x_acc_range, acc_NB, r_acc)
    # </editor-fold>

    # <editor-fold desc="计算各个规则的激活函数">
    '第1种情况，距离目标距离差很多，规则就是拉满'
    activation1 = d_pb
    output1 = bestpara[1]  # 1

    '第2种情况，超过目标距离很多，规则就是纯减速'
    activation2 = d_nb
    output2 = bestpara[2]  # -1

    '第3种情况，距离目标距离还差一点，前车速度相对快，己车速度快'
    activation3 = d_ps * (v_pb + v_ps) * sv_b
    output3 = bestpara[3]  # 0.8

    '第4种情况，距离目标距离还差一点，前车速度相对快，己车速度慢'
    activation4 = d_ps * (v_pb + v_ps) * sv_s
    output4 = bestpara[4]  # 0.5

    '第5种情况，距离目标距离还差一点，前车速度相对慢，己车速度快'
    activation5 = d_ps * (v_nb + v_ns) * sv_b
    output5 = bestpara[5]  # 0.5

    '第6种情况，距离目标距离还差一点，前车速度相对慢，己车速度慢'
    activation6 = d_ps * (v_nb + v_ns) * sv_s
    output6 = bestpara[6]  # 0.2

    '第7种情况，超过目标距离一点，前车速度相对快，己车速度快'
    activation7 = d_ns * (v_pb + v_ps) * sv_b
    output7 = bestpara[7]  # 0.2

    '第8种情况，超过目标距离一点，前车速度相对快，己车速度慢'
    activation8 = d_ns * (v_pb + v_ps) * sv_s
    output8 = bestpara[8]  # 0.1

    '第9种情况，超过目标距离一点，前车速度相对慢，己车速度快'
    activation9 = d_ns * (v_nb + v_ns) * sv_b
    output9 = bestpara[9]  # -0.5

    '第10种情况，超过目标距离一点，前车速度相对慢，己车速度慢'
    activation10 = d_ns * (v_nb + v_ns) * sv_s
    output10 = bestpara[10]  # -0.2
    # </editor-fold>

    # <editor-fold desc="给出最终的结果">
    activation = [activation1, activation2, activation3, activation4, activation5,
                  activation6, activation7, activation8, activation9, activation10]
    output = [output1, output2, output3, output4, output5, output6, output7, output8, output9, output10]
    num = sum([a * b for a, b in zip(activation, output)])  # 分子
    den = sum(activation) + 1e-12   # 分母，为了防止出现枫木等于0的情况，加上一个非常小的正数
    combined_output = num / den
    final_output = np.clip(combined_output, -1, 1)  # 最后使得输出在[-1~1]的区间内
    # </editor-fold>
    return final_output