import os
from collections import namedtuple
import random

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from torch import optim, nn

from 功能函数.sample_funtion import gmm_sample, softmax_sample
from 功能函数.state_pre import descriptive_statistics

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return softmax_sample(batch_size, self.memory)
    def __len__(self):
        return len(self.memory)


# 定义DQNNetwork
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, action_size)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2]:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.1)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


# 定义DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_num=3000, memory_size=200, batch_size=32, gamma=0.99, epsilon=1,
                 epsilon_decay=0.7,
                 epsilon_min=0.01,learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.now_step = 0
        self.epsilon_num = epsilon_num
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)

            q_values = self.model(state)
            return q_values.argmax().item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def replay(self):
        self.now_step += 1
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(
            self.device)
        non_final_next_states = torch.tensor(np.stack([s for s in batch.next_state if s is not None]),
                                             dtype=torch.float32).to(self.device)
        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1).long())

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        now_rate = self.now_step / self.epsilon_num
        if now_rate > self.epsilon_decay:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = (1 - now_rate) * (1 - self.epsilon_min)
        return loss

# 训练DQN代理

if __name__ == "__main__":
    dataset = pd.read_csv('../data_forest.csv')
    # rem = ['Id']
    # dataset.drop(rem, axis=1, inplace=True)
    w_file_name = 'softmax_2.0.txt'
    w_file_path = os.path.join('实验结果', w_file_name)
    directory = os.path.dirname(w_file_path)
    os.makedirs(directory, exist_ok=True)
    r, c = dataset.shape
    X = dataset.iloc[:, 0:(c - 1)]
    Y = dataset.iloc[:, (c - 1)]
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_val, Y_val)
    n_feature = X_train.shape[1]
    n_action = 2
    action_list_old = np.random.randint(2, size=n_feature)
    X_selected = X_train.iloc[:, action_list_old == 1]
    s = descriptive_statistics(X_selected)
    dqn_list = []
    n_state = s.shape[0]
    esposide_n = 3000
    for agent in range(n_feature):
        dqn_list.append(DQNAgent(state_size=n_state, action_size=n_action))
    action_list = np.zeros(n_feature)
    ave = 0
    best_set = []
    best_accracy = 0
    for es in range(esposide_n):
        for i in range(len(dqn_list)):
            action_list[i] = dqn_list[i].select_action(torch.tensor(s, dtype=torch.float32))
        X_selected = X_train.iloc[:, action_list == 1]
        s_ = descriptive_statistics(X_selected)
        model.fit(X_train.iloc[:, action_list == 1], Y_train)
        accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
        action_list_change = np.array([x or y for (x, y) in zip(action_list_old, action_list)])
        r_list = accuracy / sum(action_list_change) * action_list_change
        action_list_old = action_list
        s = s_
        for i in range(len(dqn_list)):
            dqn_list[i].remember(s, action_list[i], s_, r_list[i], 0)
            if i==0:
                loss0=dqn_list[i].replay()
            else:
                dqn_list[i].replay()
        print(f'{es}:    {accuracy}       loss:{loss0}')
        ave += accuracy
        num = 10
        if es % num == 0:
            tem = ave / num
            ave = 0

            with open(w_file_path, 'a') as file:
                file.write(f'ES: {es}    acc: {tem}, \n')

        if es / esposide_n > 0.95:
            if accuracy > best_accracy:
                best_accracy = accuracy
                best_set = action_list
    print("best_set:", best_set)
    print("best_acc:", best_accracy)
    with open(w_file_path, 'a') as file:
        file.write(f'best set: {best_set}    best_acc: {best_accracy},\n')

