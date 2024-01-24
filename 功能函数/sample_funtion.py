import random
from collections import namedtuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
from collections import namedtuple
import numpy as np
from sklearn.mixture import GaussianMixture

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


def gmm_sample(p, memo, batch_size):
    l1 = []
    l0 = []
    for transition in memo:
        if transition.action == 1:
            l1.append(transition)
        else:
            l0.append(transition)
    datal0 = []
    if len(l0) > 0:
        sortL0 = sorted(l0, key=lambda x: x.reward.item(), reverse=True)
        l0_maxtrx = []  # 生成数据源矩阵
        l0_num = max(int(len(l0) * p), 1)  # 采样的个数，至少为1
        for i in range(l0_num):
            l = np.append(np.concatenate((sortL0[i].state, sortL0[i].next_state)), sortL0[i].reward)
            l0_maxtrx.append(l)
        if l0_num == 1:
            l = np.append(np.concatenate((sortL0[0].state, sortL0[0].next_state)), sortL0[0].reward)
            l = l +  1e-8
            l0_maxtrx.append(l)
        data0 =np.vstack(l0_maxtrx)
        gmm0 = GaussianMixture(n_components=2, random_state=42)
        gmm0.fit(data0)
        gen_num = len(l0) - l0_num
        if gen_num > 0:
            generated_samples0, _ = gmm0.sample(gen_num)
            datal0 = np.concatenate((data0[:l0_num], generated_samples0), axis=0)
        else:
            generated_samples0 = []
            datal0 = data0[:l0_num]
    datal1 = []
    if len(l1) > 0:
        sortL1 = sorted(l1, key=lambda x: x.reward.item(), reverse=True)
        l1_maxtrx = []  # 生成数据源矩阵
        l1_num = max(int(len(l1) * p), 1)  # 采样的个数，至少为1
        for i in range(l1_num):
            l = np.append(np.concatenate((sortL1[i].state, sortL1[i].next_state)), sortL1[i].reward)
            l1_maxtrx.append(l)
        if l1_num == 1:
            l =  np.append(np.concatenate((sortL1[0].state, sortL1[0].next_state)), sortL1[0].reward)
            l = l + 1e-8
            l1_maxtrx.append(l)
        data1 = np.vstack(l1_maxtrx)
        gmm1 = GaussianMixture(n_components=2, random_state=42)
        gmm1.fit(data1)
        gen_num = len(l1) - l1_num
        if gen_num > 0:
            generated_samples1, _ = gmm1.sample(gen_num)
            datal1 = np.concatenate((data1[:l1_num], generated_samples1), axis=0)
        else:
            datal1 = data1[:l1_num]
    new_meo = []
    l_s = memo[0].state.shape[0]
    if len(datal1) > 0:
        for i in range(len(datal1)):
            state = datal1[i][0:l_s]
            n_state = datal1[i][l_s:-1]
            reward = datal1[i][-1]
            act =1
            done = 0.
            transition = Transition(state=state, action=act, next_state=n_state, reward=reward, done=done)
            new_meo.append(transition)
    if len(datal0) > 0:
        for i in range(len(datal0)):
            state =datal0[i][0:l_s]
            n_state =datal0[i][l_s:-1]
            reward = datal0[i][-1]
            act = 0.
            done = 0.
            transition = Transition(state=state, action=act, next_state=n_state, reward=reward, done=done)
            new_meo.append(transition)
    return random.sample(new_meo, batch_size)


def create_list(len_value):
    return [1 / (i + 1) for i in range(len_value)]


# 使用 NumPy 进行 softmax 采样
def softmax_sampling_np(values, num_samples):
    probabilities = np.exp(values) / np.sum(np.exp(values), axis=0)
    sampled_indices = np.random.choice(len(values), size=num_samples, p=probabilities, replace=True)
    return sampled_indices


def softmax_sampling_torch(values, num_samples):
    values_tensor = torch.tensor(values, dtype=torch.float)
    probabilities = torch.nn.functional.softmax(values_tensor, dim=0)
    sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True).tolist()
    return sampled_indices


def softmax_sample(b_size, memory):
    sorted_memorylist_desc = sorted(memory, key=lambda x: x.reward.item(), reverse=True)
    len_value = len(memory)  # 替换为你想要的长度
    num_samples = b_size  # 替换为你想要的采样数量
    result_list = create_list(len_value)
    sampled_indices = softmax_sampling_np(result_list, num_samples)
    sampled_transitions = [sorted_memorylist_desc[i] for i in sampled_indices]
    return sampled_transitions
