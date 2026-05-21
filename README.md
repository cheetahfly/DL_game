# AI 游戏项目

通过游戏学习深度学习和人工智能。

## 运行命令

### 贪吃蛇训练
```
cd F:\ai_data_ana\DL_game\snake_dqn
python train.py
```

### 五子棋 (4种模式)
```
cd F:\ai_data_ana\DL_game\gomoku
python main.py
```
选择: 1=AI对战, 2=自我训练, 3=观察对弈, 4=人类vs AI

两个项目均可正常运行！

---

## 项目结构

```
DL_game/
├── snake_dqn/          # 贪吃蛇 AI (DQN 深度强化学习)
├── gomoku/             # 五子棋 AI (Minimax + Alpha-Beta 搜索)
└── requirements.txt    # Python 依赖
```

---

## 贪吃蛇 DQN

使用深度强化学习让AI学会玩贪吃蛇。

### 安装依赖

```bash
pip install pygame torch numpy matplotlib
```

### 学习内容
- 强化学习核心概念 (状态、动作、奖励)
- DQN (Deep Q-Network) 原理
- 神经网络逼近价值函数
- Epsilon-Greedy 探索策略

---

## 五子棋

使用极大极小搜索和Alpha-Beta剪枝让AI学会下棋。

### 安装依赖

```bash
pip install pygame torch numpy
```

### 学习内容
- 极大极小 (Minimax) 搜索算法
- Alpha-Beta 剪枝优化
- 启发式评估函数设计
- 棋型识别 (活三、冲四等)

---

## 技术栈

| 项目 | 算法 | 框架 |
|------|------|------|
| 贪吃蛇 | DQN | PyTorch + Pygame |
| 五子棋 | Minimax + Alpha-Beta | Pygame |
