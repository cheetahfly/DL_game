"""
五子棋游戏主入口
支持三种模式:
1. AI vs 人类 (使用 Minimax+Alpha-Beta)
2. 自我对弈训练 (使用 Policy Network)
3. 观察 Policy Network 对弈
"""

import pygame
import sys
import os

# 将项目根目录添加到 sys.path，确保模块间导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GomokuGame
from ai import AIPlayer
from self_play import SelfPlay

sys.stdout.reconfigure(encoding='utf-8')


def mode1_ai_vs_human():
    """模式1: AI vs 人类 (使用 Minimax)"""
    CELL_SIZE = 40
    AI_DEPTH = 4

    game = GomokuGame(cell_size=CELL_SIZE)
    ai_player = AIPlayer(depth=AI_DEPTH)

    running = True
    clock = pygame.time.Clock()

    print("=" * 40)
    print("模式1: AI vs 人类")
    print("=" * 40)
    print("黑棋(AI) vs 白棋(玩家)")
    print("点击鼠标落子")
    print("按 ESC 退出, R 重新开始")
    print("=" * 40)

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    print("\n游戏已重新开始!")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not game.is_over() and game.get_current_player() == 2:
                    pos = game.get_board_pos(event.pos)
                    if pos and game.is_valid_move(pos[0], pos[1]):
                        row, col = pos
                        game.step(row, col, 2)
                        print(f"玩家落子: ({row}, {col})")

        if not game.is_over() and game.get_current_player() == 1:
            move = ai_player.get_move(game.board)
            if move:
                row, col = move
                game.step(row, col, 1)
                print(f"AI落子: ({row}, {col})")

        game.render()

        if game.is_over() and game.get_winner() != 0:
            winner = "黑棋(AI)" if game.get_winner() == 1 else "白棋(玩家)"
            print(f"\n游戏结束! {winner}获胜!")

    pygame.quit()


def mode2_self_play_training():
    """模式2: 自我对弈训练 Policy Network"""
    print("=" * 40)
    print("模式2: 自我对弈训练")
    print("=" * 40)

    model_path = "policy_net.pth"
    trainer = SelfPlay(model_path=None)  # 从头训练

    print(f"设备: {trainer.device}")
    print("开始训练...\n")

    # 训练100局
    trainer.train(num_games=100, save_path=model_path)

    print("\n训练完成!")


def mode3_watch_policy_net():
    """模式3: 观察训练好的 Policy Network 对弈"""
    import os
    model_path = "policy_net.pth"

    print("=" * 40)
    print("模式3: 观察 Policy Network 对弈")
    print("=" * 40)

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行模式2进行训练")
        return

    trainer = SelfPlay(model_path=model_path)
    print(f"已加载模型: {model_path}")

    # 先训练几局
    print("\n先训练20局...")
    trainer.train(num_games=20, save_path=model_path)

    # 观察对弈
    print("\n开始观察对弈...")
    trainer.watch_self_play(num_games=2)


def mode4_play_vs_policy_net():
    """模式4: 人类 vs 训练好的 Policy Network"""
    import os
    model_path = "policy_net.pth"

    print("=" * 40)
    print("模式4: 人类 vs Policy Network AI")
    print("=" * 40)

    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        print("请先运行模式2进行训练")
        return

    trainer = SelfPlay(model_path=model_path)
    print(f"已加载模型: {model_path}")

    trainer.play_vs_human(human_player=2, ai_player=1)


def main():
    """主函数 - 显示菜单"""
    pygame.init()

    print("\n" + "=" * 50)
    print("           五子棋 - 策略网络训练系统")
    print("=" * 50)
    print()
    print("请选择模式:")
    print()
    print("  1 - AI vs 人类 (Minimax+Alpha-Beta)")
    print("  2 - 自我对弈训练 (Policy Gradient)")
    print("  3 - 观察 Policy Network 对弈")
    print("  4 - 人类 vs Policy Network AI")
    print()
    print("=" * 50)

    choice = input("请输入选项 (1-4): ").strip()

    pygame.quit()

    if choice == "1":
        mode1_ai_vs_human()
    elif choice == "2":
        mode2_self_play_training()
    elif choice == "3":
        mode3_watch_policy_net()
    elif choice == "4":
        mode4_play_vs_policy_net()
    else:
        print("无效选项!")


if __name__ == '__main__':
    main()
