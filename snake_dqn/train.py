"""
训练入口 - DQN训练
"""

import sys
import os

# 将项目根目录添加到 sys.path，确保模块间导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from snake_game import SnakeGame
from dqn_agent import DQNAgent


def train():
    """训练DQN智能体"""
    BATCH_SIZE = 32  # 减小batch size，因为每局步数较少
    GRID_SIZE, CELL_SIZE, SPEED = 20, 20, 200
    MODEL_PATH = "F:/ai_data_ana/DL_game/snake_dqn/model.pth"
    TRAIN_FREQUENCY = 4  # 每4步训练一次

    # 询问训练局数
    try:
        EPISODES = int(input("请输入训练局数（直接回车默认500）: ").strip() or 500)
    except ValueError:
        EPISODES = 500
    print(f"训练 {EPISODES} 局\n")

    print("初始化...")
    game = SnakeGame(GRID_SIZE, CELL_SIZE, SPEED, gui=True)
    agent = DQNAgent(
        state_size=11, action_size=4, lr=0.0003, gamma=0.99,
        epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
        batch_size=BATCH_SIZE, memory_size=50000, target_update=200
    )

    # 尝试加载已有模型继续训练
    try:
        agent.load(MODEL_PATH)
        print(f"已加载模型: {MODEL_PATH}")
        print(f"继续训练 {EPISODES} 局...")
    except FileNotFoundError:
        print("未找到模型，从头开始训练...")

    scores, losses, avg_scores, recent = [], [], [], []
    step_count = 0

    print("\n开始训练...")
    print("-" * 50)

    try:
        for episode in range(EPISODES):
            state = game.reset()
            episode_experience = []  # 存储本局经验
            total_reward, total_loss, steps = 0, 0, 0

            while not game.done:
                action = agent.select_action(state)
                next_state, reward, done = game.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                episode_experience.append((state, action, reward, next_state, done))

                # 每隔 TRAIN_FREQUENCY 步训练一次，或内存足够时训练
                step_count += 1
                if step_count % TRAIN_FREQUENCY == 0 and len(agent.memory) >= BATCH_SIZE:
                    batch = agent.memory.sample(BATCH_SIZE)
                    loss = agent.train(batch)
                    total_loss += loss

                state = next_state
                total_reward += reward
                steps += 1
                game.render()

            # 每局结束后，用本局经验再训练几次
            if len(episode_experience) >= BATCH_SIZE:
                # 把本局经验加入memory（已经在上面加入了）
                pass
            else:
                # 经验不足时，重复利用已有经验训练
                for _ in range(4):  # 每局结束后训练4次
                    if len(agent.memory) >= BATCH_SIZE:
                        batch = agent.memory.sample(BATCH_SIZE)
                        loss = agent.train(batch)
                        total_loss += loss

            agent.decay_epsilon()
            scores.append(total_reward)
            recent.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg = np.mean(recent)
                avg_scores.append(avg)
                recent = []
                avg_loss = total_loss / max(steps, 1)
                print(f"局数: {episode + 1}/{EPISODES} | 分数: {game.score} | "
                      f"平均: {avg:.2f} | 损失: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")

            if (episode + 1) % 100 == 0:
                plot_scores(scores, avg_scores, episode + 1)

    except KeyboardInterrupt:
        print("\n训练中断")
    finally:
        save_path = "F:/ai_data_ana/DL_game/snake_dqn/model.pth"
        agent.save(save_path)
        print(f"\n模型已保存: {save_path}")
        show_final(scores)
        game.close()


def plot_scores(scores, avg_scores, episode):
    """显示训练曲线"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6)
    if avg_scores:
        plt.plot(range(9, len(scores) + 1, 10), avg_scores, 'r-', linewidth=2, label='10局平均')
    plt.title(f'分数 (第{episode}局)')
    plt.xlabel('局数')
    plt.ylabel('分数')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=False)


def show_final(scores):
    """显示最终曲线"""
    plt.figure(figsize=(12, 4))
    plt.plot(scores, alpha=0.6, label='分数')
    if len(scores) > 10:
        avg = [np.mean(scores[max(0, i - 10):i + 1]) for i in range(len(scores))]
        plt.plot(avg, 'r-', linewidth=2, label='10局移动平均')
    plt.title('训练分数')
    plt.xlabel('局数')
    plt.ylabel('分数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    train()
