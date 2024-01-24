import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.act_n = int(action_dim / 2)
        self.fc_actor = nn.Linear(state_dim, 128)
        self.fc_critic = nn.Linear(state_dim, 128)
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x_actor = F.relu(self.fc_actor(state))
        x_critic = F.relu(self.fc_critic(state))
        # Split the output of actor_head into two parts
        actor_output = self.actor_head(x_actor)
        state_mask_1 = state.bool()
        state_mask_2 = state.bool()
        # 将 mask 应用到相应的位置
        masked_probs = actor_output.clone()
        masked_probs[1:55] = masked_probs[1:55].masked_fill(state_mask_1, float('-inf'))
        masked_probs[56:110] = masked_probs[56:110].masked_fill(~state_mask_2, float('-inf'))
        action_probs_1 = F.softmax(masked_probs[:self.act_n], dim=-1)
        action_probs_2 = F.softmax(masked_probs[self.act_n:], dim=-1)
        action_probs = torch.cat([action_probs_1, action_probs_2], dim=-1)
        value = self.critic_head(x_critic)
        return action_probs, value


class A2C:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.model.actor_head.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.model.critic_head.parameters(), lr=lr_critic)
        self.gamma = gamma

    def update_policy(self, state, actions, rewards, next_state,obs_loss):
        state = torch.FloatTensor(state)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 2)  # Assuming two actions
        rewards = torch.tensor([rewards], dtype=torch.float32)
        next_state = torch.FloatTensor(next_state)

        # Compute TD error
        action_probs, value = self.model(state)
        next_value = self.model(next_state)[1]  # Value from the critic for the next state
        td_error = rewards + self.gamma * next_value - value

        # Compute actor loss
        selected_probs = torch.gather(action_probs, 0, actions.view(-1))
        log_probs = torch.log(selected_probs + 1e-8)
        # Backward pass
        actor_loss = -(log_probs * td_error.detach()).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # 计算 Critic Loss
        critic_loss = F.smooth_l1_loss(value, rewards)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        obs_loss=actor_loss+0.5*critic_loss
        return obs_loss
class Env:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.train_X, self.test_X, self.train_Y, self.test_Y = model_selection.train_test_split(X, Y, test_size=0.1,
                                                                                                random_state=0)
        self.model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
        self.model.fit(self.train_X, self.train_Y)
        self.acc = self.model.score(self.test_X, self.test_Y)

    def step(self, s, act):
        obs = np.copy(s)
        if act[0]:
            obs[act[0] - 1] = 1
        if act[1] - len(s) - 1:
            obs[act[1] - len(s) - 2] = 0
        self.model.fit(self.train_X.iloc[:, obs == 1], self.train_Y)
        acc = self.model.score(self.test_X.iloc[:, obs == 1], self.test_Y)
        rew = 0
        if acc<0.60:
            rew=0
        elif acc<0.75:
            rew=1
        else :
            rew=(acc-0.75)*100+1


        done1 = 0
        if np.sum(obs) / obs.shape[0] > 0.8:
            done1 = 1

        return obs, rew, done1,acc


if __name__ == "__main__":
    dataset = pd.read_csv('data_forest.csv')
    r, c = dataset.shape
    X = dataset.iloc[:, 0:(c - 1)]
    Y = dataset.iloc[:, (c - 1)]
    env = Env(X, Y)
    state_dim = X.shape[1]
    action_dim = X.shape[1] + 1
    lr_actor = 1e-3
    lr_critic = 1e-3
    gamma = 0.95
    a2c_agent = A2C(state_dim, action_dim * 2, lr_actor, lr_critic, gamma)
    num_episodes = 1000
    max_steps = 200
    obs_loss=0
    for episode in range(num_episodes):
        state = np.random.randint(2, size=state_dim)  # 随机初始化
        env.model.fit(env.train_X.iloc[:, state == 1], env.train_Y)
        acc = env.model.score(env.test_X.iloc[:, state == 1], env.test_Y)
        print(f"初始化acc:{acc},特征子集")
        total_reward = 0
        done = 0
        current_step = 0
        while not done and current_step < max_steps:
            current_step += 1
            # Collect data from the environment
            action_probs, _ = a2c_agent.model(torch.FloatTensor(state))

            # Assuming two dimensions, each with 54 actions
            action_indices = [
                torch.multinomial(action_probs[:action_dim], 1).item(),
                torch.multinomial(action_probs[action_dim:], 1).item() + action_dim
            ]

            actions = action_indices
            if actions[0]==0 :
                print("不减")
            if actions[1]==0:
                print("不加")
            if actions[0]==0 and actions[1]==0:
                print("不改变")
            next_state, reward, done,acc0 = env.step(state, actions)

            # Update policy

            obs_loss=a2c_agent.update_policy(state, actions, reward, next_state,obs_loss)
            state = next_state
            print(current_step,f"acc:{acc0}    reward:{reward}    loss:{obs_loss}   ")
            total_reward += reward

        # Print episode information
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        with open('mask动作.txt',
                  'a') as file:
            file.write(str(total_reward) + '\n')
    # After training, you can use the learned policy to act in your environment.
