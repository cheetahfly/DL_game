"""
测试/演示入口 - 加载模型或人类控制
"""

import sys
import os

# 将项目根目录添加到 sys.path，确保模块间导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from snake_game import SnakeGame
from dqn_agent import DQNAgent


def play(ai_mode=True, model_path="F:/ai_data_ana/DL_game/snake_dqn/model.pth"):
    """播放游戏"""
    GRID_SIZE, CELL_SIZE, SPEED = 20, 20, 10

    print("初始化游戏...")
    game = SnakeGame(GRID_SIZE, CELL_SIZE, SPEED, gui=True)

    agent = None
    if ai_mode:
        agent = DQNAgent(state_size=11, action_size=4)
        try:
            agent.load(model_path)
            agent.epsilon = 0
            print(f"加载模型: {model_path}")
        except FileNotFoundError:
            print(f"警告: 找不到模型 {model_path},使用随机动作")
            agent = None

    mode = "AI控制" if ai_mode else "人类控制"
    print(f"\n{'=' * 50}\n模式: {mode} (ESC退出)\n{'=' * 50}")

    state = game.reset()
    total_score, games = 0, 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    print(f"\n结束! {games}局, 总分: {total_score}")
                    return

                # 人类控制
                if not ai_mode:
                    key_map = {pygame.K_UP: 0, pygame.K_DOWN: 1,
                              pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
                    if e.key in key_map:
                        state, _, done = game.step(key_map[e.key])
                        if done:
                            games += 1
                            total_score += game.score
                            print(f"游戏{games}: {game.score}分")
                            state = game.reset()

        # AI控制
        if ai_mode and not game.done:
            action = agent.select_action(state) if agent else np.random.randint(0, 4)
            state, _, done = game.step(action)
            if done:
                games += 1
                total_score += game.score
                print(f"游戏{games}: {game.score}分, 平均{total_score / games:.2f}")
                state = game.reset()

        game.render()


def menu():
    """菜单"""
    print("\n" + "=" * 50)
    print("   贪吃蛇 DQN AI - 演示")
    print("=" * 50)
    print("1. AI控制 (加载模型)")
    print("2. 人类控制 (方向键)")
    print("3. AI控制 (随机动作)")
    print("0. 退出")
    print("=" * 50)

    return input("选择: ").strip()


if __name__ == "__main__":
    while True:
        ch = menu()
        if ch == "1":
            play(ai_mode=True)
        elif ch == "2":
            play(ai_mode=False)
        elif ch == "3":
            play(ai_mode=True, model_path=None)
        elif ch == "0":
            print("再见!")
            break
        else:
            print("无效选择")
