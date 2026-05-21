"""
AI玩家 - 使用Minimax和Alpha-Beta剪枝
"""

import random
from board import Board


class AIPlayer:
    """AI玩家类"""

    # 棋型评估权重
    SCORES = {
        'five': 100000,      # 五连
        'live_four': 10000,  # 活四
        'rush_four': 1000,   # 冲四
        'live_three': 1000,  # 活三
        'sleep_three': 100,  # 眠三
        'live_two': 100,     # 活二
        'sleep_two': 10,     # 眠二
        'live_one': 1,       # 活一
    }

    def __init__(self, depth=4):
        """
        初始化AI
        depth: 搜索深度
        """
        self.depth = depth

    def get_move(self, board):
        """
        获取最佳落子位置
        board: Board对象
        返回: (row, col) 最佳落子位置
        """
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        best_score = float('-inf')
        best_move = valid_moves[0]

        # 按距离中心的位置排序，优先搜索中心附近的点
        center = board.size // 2
        valid_moves.sort(key=lambda pos: abs(pos[0] - center) + abs(pos[1] - center))

        alpha = float('-inf')
        beta = float('inf')

        for row, col in valid_moves:
            # 尝试落子
            board.grid[row][col] = 1  # AI用黑棋
            score = self.minimax(board, self.depth - 1, alpha, beta, False)
            board.grid[row][col] = 0  # 撤销

            if score > best_score:
                best_score = score
                best_move = (row, col)

            alpha = max(alpha, score)

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing):
        """
        Minimax算法带Alpha-Beta剪枝
        board: 棋盘
        depth: 搜索深度
        alpha: Alpha值
        beta: Beta值
        maximizing: 是否是最大化节点
        返回: 评估分数
        """
        # 检查游戏是否结束
        winner = self._check_winner(board)
        if winner == 1:
            return 1000000 + depth  # AI赢
        if winner == 2:
            return -1000000 - depth  # 人类赢

        # 达到搜索深度
        if depth == 0:
            return self.evaluate(board)

        valid_moves = board.get_valid_moves()

        if not valid_moves:
            return self.evaluate(board)

        # 限制搜索节点数量，防止太慢
        if len(valid_moves) > 15:
            # 只保留评估分数最高的前15个位置
            moves_with_scores = []
            for row, col in valid_moves:
                board.grid[row][col] = 1
                score = self.evaluate(board)
                board.grid[row][col] = 0
                moves_with_scores.append((row, col, score))
            moves_with_scores.sort(key=lambda x: x[2], reverse=True)
            valid_moves = [(row, col) for row, col, _ in moves_with_scores[:15]]

        if maximizing:
            max_score = float('-inf')
            for row, col in valid_moves:
                board.grid[row][col] = 1
                score = self.minimax(board, depth - 1, alpha, beta, False)
                board.grid[row][col] = 0

                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float('inf')
            for row, col in valid_moves:
                board.grid[row][col] = 2  # 人类用白棋
                score = self.minimax(board, depth - 1, alpha, beta, True)
                board.grid[row][col] = 0

                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score

    def evaluate(self, board):
        """
        评估函数 - 统计棋型并计算分数
        返回: 分数(AI为正，人类为负)
        """
        ai_score = self._evaluate_player(board, 1)
        human_score = self._evaluate_player(board, 2)
        return ai_score - human_score

    def _evaluate_player(self, board, player):
        """
        评估指定玩家的分数
        player: 1(AI/黑棋) 或 2(人类/白棋)
        """
        score = 0
        patterns = self._find_patterns(board, player)

        for pattern_type, count in patterns.items():
            score += self.SCORES.get(pattern_type, 0) * count

        return score

    def _find_patterns(self, board, player):
        """
        找出所有指定玩家的棋型
        返回: 各种棋型的数量字典
        """
        patterns = {
            'five': 0,
            'live_four': 0,
            'rush_four': 0,
            'live_three': 0,
            'sleep_three': 0,
            'live_two': 0,
            'sleep_two': 0,
            'live_one': 0,
        }

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 用于去重
        found_patterns = set()

        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row][col] != player:
                    continue

                for dr, dc in directions:
                    pattern = self._analyze_direction(board, row, col, dr, dc, player)
                    if pattern and pattern not in found_patterns:
                        found_patterns.add(pattern)
                        if pattern in patterns:
                            patterns[pattern] += 1

        return patterns

    def _analyze_direction(self, board, row, col, dr, dc, player):
        """
        分析指定方向上的棋型
        返回: 棋型名称
        """
        empty = 0
        opponent = 3 - player

        # 统计连续棋子数
        count = 1
        spaces = []

        # 正方向
        r, c = row + dr, col + dc
        for _ in range(4):
            if not board.is_valid_position(r, c):
                break
            val = board.grid[r][c]
            if val == player:
                count += 1
                r += dr
                c += dc
            elif val == empty:
                spaces.append((r, c))
                break
            else:
                break

        # 反方向
        r, c = row - dr, col - dc
        for _ in range(4):
            if not board.is_valid_position(r, c):
                break
            val = board.grid[r][c]
            if val == player:
                count += 1
                r -= dr
                c -= dc
            elif val == empty:
                spaces.append((r, c))
                break
            else:
                break

        if count >= 5:
            return 'five'
        elif count == 4:
            if len(spaces) == 2:
                return 'live_four'
            elif len(spaces) == 1:
                return 'rush_four'
        elif count == 3:
            if len(spaces) == 2:
                # 检查是否是活三(中间有空位)
                return 'live_three'
            elif len(spaces) == 1:
                return 'sleep_three'
        elif count == 2:
            if len(spaces) == 2:
                return 'live_two'
            elif len(spaces) == 1:
                return 'sleep_two'
        elif count == 1 and len(spaces) == 2:
            return 'live_one'

        return None

    def _check_winner(self, board):
        """
        检查获胜者
        返回: 0(无), 1(AI赢), 2(人类赢)
        """
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row][col] != 0:
                    if board.check_win(row, col):
                        return board.grid[row][col]
        return 0
