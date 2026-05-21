"""
棋盘数据结构
15x15 五子棋棋盘
"""

class Board:
    """棋盘类，管理棋盘状态"""

    def __init__(self, size=15):
        """
        初始化棋盘
        size: 棋盘大小，默认15x15
        """
        self.size = size
        self.grid = [[0] * size for _ in range(size)]  # 0=空, 1=黑棋, 2=白棋

    def put(self, row, col, player):
        """
        在指定位置落子
        row: 行索引
        col: 列索引
        player: 1(黑棋) 或 2(白棋)
        返回: 是否成功落子
        """
        if not self.is_valid_position(row, col):
            return False
        if self.grid[row][col] != 0:
            return False
        self.grid[row][col] = player
        return True

    def get(self, row, col):
        """
        获取指定位置的棋子
        row: 行索引
        col: 列索引
        返回: 0(空), 1(黑棋), 2(白棋)
        """
        if not self.is_valid_position(row, col):
            return -1
        return self.grid[row][col]

    def is_valid_position(self, row, col):
        """检查位置是否在棋盘范围内"""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row, col):
        """检查位置是否为空"""
        return self.is_valid_position(row, col) and self.grid[row][col] == 0

    def is_full(self):
        """检查棋盘是否已满"""
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == 0:
                    return False
        return True

    def get_valid_moves(self):
        """
        获取所有可落子位置
        为了提高效率，只返回有棋子周围的位置
        """
        moves = set()

        # 先检查是否有任意棋子，有则只在周围搜索
        has_any_piece = False
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] != 0:
                    has_any_piece = True
                    break
            if has_any_piece:
                break

        if not has_any_piece:
            # 棋盘为空，返回中心点
            center = self.size // 2
            return [(center, center)]

        # 搜索有棋子周围的所有空位
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] != 0:
                    # 搜索周围3x3范围
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = row + dr, col + dc
                            if self.is_empty(nr, nc):
                                moves.add((nr, nc))

        return list(moves)

    def check_win(self, row, col):
        """
        检查指定位置落子后是否形成五子连珠
        row: 行索引
        col: 列索引
        返回: 是否获胜
        """
        player = self.grid[row][col]
        if player == 0:
            return False

        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]

        for dr, dc in directions:
            count = 1
            # 正方向
            r, c = row + dr, col + dc
            while self.is_valid_position(r, c) and self.grid[r][c] == player:
                count += 1
                r += dr
                c += dc
            # 反方向
            r, c = row - dr, col - dc
            while self.is_valid_position(r, c) and self.grid[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

    def clone(self):
        """复制棋盘"""
        new_board = Board(self.size)
        new_board.grid = [row[:] for row in self.grid]
        return new_board
