"""
DQN智能体类 - 深度Q网络实现 (Double DQN)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """DQN网络: 11 -> 256 -> 128 -> 64 -> 4"""

    def __init__(self, state_size=11, action_size=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体 (Double DQN)"""

    def __init__(self, state_size=11, action_size=4, lr=0.0003, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 batch_size=64, memory_size=50000, target_update=200):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_step = 0

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)

    def select_action(self, state, epsilon=None) -> int:
        """epsilon-greedy策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return random.randrange(4)

        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax().item()

    def train(self, batch):
        """Double DQN训练,返回损失值"""
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: 使用 policy_net 选择动作，target_net 评估
        with torch.no_grad():
            # 用 policy_net 选择最佳动作
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # 用 target_net 评估该动作的Q值
            max_next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            # 计算 TD target
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.mse_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """每个 episode 结束后衰减 epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
        self.train_step = ckpt['train_step']
        self.target_net.eval()
