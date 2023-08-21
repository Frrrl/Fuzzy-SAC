import os
import collections
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

config = {
    # For RL
    'device': 'cuda:0',  # ['cpu','cuda:0','cuda:1',....]
    'num_threads': 1,  # if it uses cpu
    'buffer_maxlen': 2 ** 19,  # buffer最大的长度为20W，按照1000个episode，每个episode200个数据进行计算
    'batch_size': 2 ** 10,  # 一次采样的数据量为1024
    'mid_dim': 256,  # 中间层设置为128
    'alpha': 1,  # alpha<=1 (only for sac temperature param)
    'tau': 0.02,  # target smoothing coefficient
    'q_lr': 0.001,  # q net learning rate
    'a_lr': 0.001,  # actor net learning rate
    'gamma': 0.99,  # discounted factor
    'max_episode': 100,  # max episode
    'max_step': 200,  # max step
    'random_seed': True,  # False means fix seed [False,True]
    'seed': 1,  # fix seed
    'beta': 0.2,  # [0,1)
    'v_lr': 0.001
}


class Buffer:
    def __init__(self, buffer_maxlen, device):
        self.device = torch.device(device)
        self.buffer = collections.deque(maxlen=buffer_maxlen)  # 创建一个双端队列，可以加入或弹出元素

    def push(self, data):  # 加入一个元素data进入buffer中
        self.buffer.append(data)

    def sample(self, batch_size):  # 按照batch_size从buffer中进行采样，每个样本含有s,a,r,s',done
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)  # 采样出batch_size个batch
        for experience in batch:  # 将batch中的数字全部送到相应的s,a,r,s',done中
            s, a, r, n_s, d = experience
            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        state_list = torch.as_tensor(np.array(state_list), dtype=torch.float32, device=self.device)  # 转变为tensor变量
        action_list = torch.as_tensor(np.array(action_list), dtype=torch.float32, device=self.device)
        reward_list = torch.as_tensor(np.array(reward_list), dtype=torch.float32, device=self.device)
        next_state_list = torch.as_tensor(np.array(next_state_list), dtype=torch.float32, device=self.device)
        done_list = torch.as_tensor(np.array(done_list), dtype=torch.float32, device=self.device)

        return state_list, action_list, reward_list.unsqueeze(-1), next_state_list, done_list.unsqueeze(-1)
        # 由于reward和done都只有一个数，所以需要进行一下unsqueeze(-1)

    def buffer_len(self):  # 计算buffer的长度
        return len(self.buffer)


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_combine = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU())
        self.net_mean = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))
        self.net_log_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, action_dim))
        self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))

    def forward(self, state):  # 不太明白和test的区别
        x = self.net_state(state)
        return self.net_mean(x).tanh()  # action

    def get_action(self, state):  # 给一个带有随机性的action
        x = self.net_combine(state)
        mean = self.net_mean(x)
        std = self.net_log_std(x).clamp(-16, 2).exp()
        return torch.normal(mean, std).tanh()

    def get_action_test(self, state):  # 给一个不带有随机性的action
        x = self.net_combine(state)
        mean = self.net_mean(x)
        return mean.tanh()

    def get_action_log_prob(self, state):  # 给出一个含有随机噪声的动作以及
        x = self.net_combine(state)
        mean = self.net_mean(x)
        log_std = self.net_log_std(x).clamp(-16, 2)  # 设定上下限，-16~2
        std = log_std.exp()  # 进行e的幂次方，消除log

        # re-parameterize
        noise = torch.randn_like(mean, requires_grad=True)  # 返回一个和mean相同大小的张量，由均值为0、方差为1的正态分布填充
        a_tan = (mean + std * noise).tanh()  # action.tanh()  一个含有噪声随机量的东西

        log_prob = log_std + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        return a_tan, log_prob.sum(1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(TwinCritic, self).__init__()
        # shared parameter
        self.net_combine = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        # only use q1
        x = self.net_combine(torch.cat((state, action), dim=1))
        return self.net_q1(x)

    def get_q1_q2(self, state, action):
        # return q1, q2 value
        x = self.net_combine(torch.cat((state, action), dim=1))
        return self.net_q1(x), self.net_q2(x)


def soft_target_update(target, current, tau=0.05):  # target_net每次更新的步骤为原来的95%基础上加上5%current的参数值
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class SAC:
    def __init__(self, env):
        self.env = env
        self.device = config['device']
        self.state_dim = len(env.state_high)
        self.action_dim = len(env.action_high)
        self.mid_dim = config['mid_dim']

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.q_lr = config['q_lr']
        self.a_lr = config['a_lr']
        self.v_lr = config['v_lr']

        # buffer
        self.batch_size = config['batch_size']
        self.buffer = Buffer(config['buffer_maxlen'], self.device)

        self.target_entropy = np.log(self.action_dim)
        self.alpha_log = torch.tensor((-np.log(self.action_dim),), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.actor = SACActor(self.state_dim, self.action_dim, self.mid_dim).to(self.device)

        # #加载训练好的网络的参数，在此基础上继续训练
        # self.actor.load_state_dict(torch.load('./Models/49999best_actor.pt'))

        self.actor_target = deepcopy(self.actor)  # 反正就是copy，至于怎么copy的无所谓
        self.critic = TwinCritic(self.state_dim, self.action_dim, int(self.mid_dim)).to(self.device)

        # #加载训练好的网络的参数，在此基础上继续训练
        # self.critic.load_state_dict(torch.load('./Models/49999best_critic.pt'))

        self.critic_target = deepcopy(self.critic)  # 给一个target网络，作用为设定loss，作为真实值和预测值的差别

        self.criterion = torch.nn.SmoothL1Loss()  # 采用L1正则化，Lasso回归，L2正则化则是岭回归，反正就是一种loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.q_lr)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.v_lr)  # 对alpha进行优化

    def get_action(self, states):  # 给出一个有偏差的action
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.actor.get_action(states)
        return actions.detach().cpu().numpy()

    def get_action_test(self, states):  # 给出一个无偏差的action
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.actor.get_action_test(states)
        return actions.detach().cpu().numpy()

    def push(self, data):  # 把一条轨迹塞进buffer当中
        self.buffer.push(data)

    def update(self):  # 核心的一步，将所有参数进行一次更新
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        alpha = self.alpha_log.exp().detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor_target.get_action_log_prob(next_state)
            next_q = torch.min(*self.critic_target.get_q1_q2(next_state, next_action))
            q_target = reward + done * (next_q + next_log_prob * alpha) * self.gamma
        q1, q2 = self.critic.get_q1_q2(state, action)
        critic_loss = self.criterion(q1, q_target) + self.criterion(q2, q_target)

        action_pg, log_prob = self.actor.get_action_log_prob(state)  # policy gradient
        alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
        actor_loss = -(torch.min(*self.critic_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        soft_target_update(self.critic_target, self.critic, self.tau)
        soft_target_update(self.actor_target, self.actor, self.tau)

    def save_nets(self, dir_name, episode):  # 保存模型
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        torch.save(self.critic.state_dict(), dir_name + '/Models/' + str(episode) + 'best_critic.pt')
        torch.save(self.actor.state_dict(), dir_name + '/Models/' + str(episode) + 'best_actor.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name, episode):  # 加载模型
        self.critic.load_state_dict(torch.load(dir_name + '/Models/' + str(episode) + 'best_critic.pt'))
        self.actor.load_state_dict(torch.load(dir_name + '/Models/' + str(episode) + 'best_actor.pt'))

        self.critic_target.load_state_dict(torch.load(dir_name + '/Models/' + str(episode) + 'best_critic.pt'))
        self.actor_target.load_state_dict(torch.load(dir_name + '/Models/' + str(episode) + 'best_actor.pt'))
        # print('RL load successfully')
