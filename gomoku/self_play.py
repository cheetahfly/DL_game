"""
自我对弈训练器 - Policy Gradient
使用策略网络进行自我对弈并更新网络
"""

import sys
import torch
import numpy as np
from policy_network import PolicyNetwork

sys.stdout.reconfigure(encoding='utf-8')


class SelfPlay:
    """自我对弈训练器"""

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork().to(self.device)
        if model_path:
            self.load(model_path)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.games_played = 0
        self.black_wins = 0
        self.white_wins = 0
        torch.set_printoptions(precision=4, linewidth=120)

    def get_move(self, board):
        """用神经网络获取落子位置"""
        state = self.board_to_tensor(board)
        with torch.no_grad():
            policy, _ = self.policy_net(state)
        probs = policy.cpu().numpy()[0]
        valid_moves = board.get_valid_moves()
        valid_probs = np.zeros(225)
        for row, col in valid_moves:
            valid_probs[row * 15 + col] = probs[row * 15 + col]
        # epsilon 探索
        if np.random.random() < 0.1:
            return valid_moves[np.random.randint(len(valid_moves))]
        valid_probs /= max(valid_probs.sum(), 1e-8)
        if valid_probs.sum() == 0:
            return valid_moves[np.random.randint(len(valid_moves))]
        return np.unravel_index(np.random.choice(225, p=valid_probs), (15, 15))

    def board_to_tensor(self, board):
        """将棋盘转换为神经网络输入"""
        state = np.zeros((1, 15, 15), dtype=np.float32)
        for r in range(15):
            for c in range(15):
                val = board.get(r, c)
                state[0, r, c] = 1 if val == 1 else (-1 if val == 2 else 0)
        return torch.FloatTensor(state).unsqueeze(1).to(self.device)

    def play_one_game(self):
        """下一局自我对弈"""
        from game import GomokuGame
        game = GomokuGame()
        game.reset()
        states, players = [], []

        while not game.is_over():
            board = game.board
            row, col = self.get_move(board)
            states.append(self.board_to_tensor(board))
            players.append(1 if game.current_player == 1 else -1)
            if not game.step(row, col, game.current_player):
                continue

        winner = game.get_winner()
        reward = 1 if winner == 1 else (-1 if winner == 0 else 0)
        return states, players, reward

    def train(self, num_games=100, save_path="policy_net.pth"):
        """训练 num_games 局"""
        for i in range(num_games):
            states, players, final_reward = self.play_one_game()
            if final_reward == 1:
                self.black_wins += 1
            elif final_reward == -1:
                self.white_wins += 1

            self.optimizer.zero_grad()
            total_loss = 0
            for state, player in zip(states, players):
                policy, value = self.policy_net(state)
                value_loss = (value - player * final_reward) ** 2
                log_prob = torch.log(policy + 1e-8)
                advantage = player * final_reward
                policy_loss = -(log_prob * advantage).mean()
                total_loss += value_loss.mean() + policy_loss

            total_loss.backward()
            self.optimizer.step()
            self.games_played += 1

            if (i + 1) % 10 == 0:
                total = max(1, self.games_played)
                print(f"已训练 {self.games_played} 局, 黑胜 {self.black_wins}, "
                      f"白胜 {self.white_wins}, 黑胜率 {self.black_wins/total:.1%}")

        self.save(save_path)
        print(f"\n模型已保存到: {save_path}")

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))

    def watch_self_play(self, num_games=1):
        """观察自我对弈"""
        from game import GomokuGame
        for i in range(num_games):
            print(f"\n=== 第 {i+1} 局 ===")
            game = GomokuGame()
            game.reset()
            move_num = 0

            while not game.is_over():
                row, col = self.get_move(game.board)
                player = game.current_player
                color = "黑" if player == 1 else "白"
                if game.step(row, col, player):
                    move_num += 1
                    print(f"{move_num}. {color}棋落子: ({row}, {col})")
                game.render()

            winner = game.get_winner()
            if winner == 0:
                print("结果: 平局")
            else:
                print(f"结果: {'黑' if winner == 1 else '白'}棋获胜")

    def play_vs_human(self, human_player=2, ai_player=1):
        """人类 vs 训练好的网络"""
        from game import GomokuGame
        game = GomokuGame()
        game.reset()
        print(f"\n人类执{'白' if human_player==2 else '黑'}棋")
        print("点击落子, ESC退出, R重开\n")

        import pygame
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = game.get_board_pos(event.pos)
                    if pos and game.is_valid_move(pos[0], pos[1]):
                        game.step(pos[0], pos[1], human_player)

            if not game.is_over() and game.get_winner() == 0 and game.current_player != human_player:
                row, col = self.get_move(game.board)
                game.step(row, col, ai_player)

            game.render()
            if game.is_over():
                winner = game.get_winner()
                if winner == 0:
                    print("\n结果: 平局")
                elif winner == human_player:
                    print("\n恭喜! 你赢了!")
                else:
                    print("\nAI获胜!")
                pygame.time.wait(2000)
                running = False
        pygame.quit()
