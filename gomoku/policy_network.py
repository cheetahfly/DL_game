"""
策略神经网络 - Policy Gradient
输入棋盘状态，输出每个位置的概率和评估值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    策略神经网络：输入棋盘状态，输出每个位置的概率和评估值
    输入: (batch, 1, 15, 15) - 0=空, 1=黑棋, -1=白棋
    输出: policy (batch, 225), value (batch, 1)
    """

    def __init__(self, board_size=15):
        super().__init__()
        self.board_size = board_size

        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # 全连接层
        self.fc_policy = nn.Linear(128 * board_size * board_size, 512)
        self.fc_value = nn.Linear(128 * board_size * board_size, 128)

        # 输出层
        self.fc_policy_out = nn.Linear(512, board_size * board_size)
        self.fc_value_out = nn.Linear(128, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播
        x: (batch, 1, 15, 15)
        返回: policy (batch, 225), value (batch, 1)
        """
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten

        # 策略头
        policy = self.fc_policy(x)
        policy = F.relu(policy)
        policy = self.fc_policy_out(policy)
        policy = F.softmax(policy, dim=-1)

        # 价值头
        value = self.fc_value(x)
        value = F.relu(value)
        value = self.fc_value_out(value)
        value = torch.tanh(value)

        return policy, value
