"""
游戏逻辑和规则
"""

import pygame
import sys

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (210, 180, 140)
LINE_COLOR = (0, 0, 0)
BG_COLOR = (230, 200, 160)
TEXT_COLOR = (50, 50, 50)


class GomokuGame:
    """五子棋游戏类"""

    def __init__(self, cell_size=40):
        """
        初始化游戏
        cell_size: 每个格子的大小(像素)
        """
        self.cell_size = cell_size
        self.board_size = 15
        self.margin = cell_size // 2

        # 初始化Pygame
        pygame.init()

        # 创建窗口
        screen_size = self.cell_size * (self.board_size - 1) + self.margin * 2
        self.screen = pygame.display.set_mode((screen_size, screen_size + 60))
        pygame.display.set_caption('五子棋')

        # 设置字体
        self.font = pygame.font.Font(pygame.font.match_font('microsoftyahei'), 28)
        self.small_font = pygame.font.Font(pygame.font.match_font('microsoftyahei'), 20)

        # 重置游戏
        self.reset()

    def reset(self):
        """重置游戏"""
        from board import Board
        self.board = Board(self.board_size)
        self.current_player = 1  # 黑棋先手
        self.game_over = False
        self.winner = 0
        self.last_pos = None  # 最后一个落子位置

    def step(self, row, col, player):
        """
        落子
        row: 行索引
        col: 列索引
        player: 1(黑棋) 或 2(白棋)
        返回: 是否成功落子
        """
        if self.game_over:
            return False

        if player != self.current_player:
            return False

        if self.board.put(row, col, player):
            self.last_pos = (row, col)

            # 检查是否获胜
            if self.board.check_win(row, col):
                self.game_over = True
                self.winner = player

            # 切换玩家
            self.current_player = 3 - self.current_player
            return True

        return False

    def is_over(self):
        """游戏是否结束"""
        return self.game_over or self.board.is_full()

    def get_winner(self):
        """获取获胜方"""
        return self.winner

    def get_current_player(self):
        """获取当前玩家"""
        return self.current_player

    def render(self, screen=None):
        """渲染游戏画面"""
        if screen is None:
            screen = self.screen

        # 清屏
        screen.fill(BG_COLOR)

        # 绘制棋盘背景
        board_rect = pygame.Rect(0, 0,
                                  self.cell_size * (self.board_size - 1) + self.margin * 2,
                                  self.cell_size * (self.board_size - 1) + self.margin * 2)
        pygame.draw.rect(screen, BROWN, board_rect)

        # 绘制网格线
        for i in range(self.board_size):
            pos = self.margin + i * self.cell_size
            # 横线
            pygame.draw.line(screen, LINE_COLOR,
                           (self.margin, pos),
                           (self.margin + (self.board_size - 1) * self.cell_size, pos), 1)
            # 竖线
            pygame.draw.line(screen, LINE_COLOR,
                           (pos, self.margin),
                           (pos, self.margin + (self.board_size - 1) * self.cell_size), 1)

        # 绘制星位(天元和四个角星)
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            pygame.draw.circle(screen, LINE_COLOR, (x, y), 4)

        # 绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.get(row, col)
                if piece != 0:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = BLACK if piece == 1 else WHITE
                    pygame.draw.circle(screen, color, (x, y), self.cell_size // 2 - 2)
                    if piece == 2:  # 白棋加边框
                        pygame.draw.circle(screen, BLACK, (x, y), self.cell_size // 2 - 2, 1)

        # 绘制最后一个落子位置标记
        if self.last_pos:
            row, col = self.last_pos
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            piece = self.board.get(row, col)
            color = WHITE if piece == 1 else BLACK
            pygame.draw.circle(screen, color, (x, y), 5, 2)

        # 绘制底部信息栏
        info_rect = pygame.Rect(0, board_rect.height, screen.get_width(), 60)
        pygame.draw.rect(screen, BG_COLOR, info_rect)

        # 显示当前回合
        player_text = "黑棋" if self.current_player == 1 else "白棋"
        if self.game_over:
            if self.winner == 0:
                text = "平局!"
            else:
                winner_text = "黑棋" if self.winner == 1 else "白棋"
                text = f"游戏结束! {winner_text}获胜!"
        else:
            text = f"当前回合: {player_text}"

        text_surface = self.font.render(text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, board_rect.height + 30))
        screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def get_board_pos(self, mouse_pos):
        """
        将鼠标坐标转换为棋盘坐标
        mouse_pos: (x, y) 鼠标像素坐标
        返回: (row, col) 棋盘坐标，或 None
        """
        x, y = mouse_pos
        # 转换为棋盘坐标
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)

        # 检查是否在有效范围内
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None

    def is_valid_move(self, row, col):
        """检查是否是有效的落子位置"""
        if not self.board.is_valid_position(row, col):
            return False
        return self.board.is_empty(row, col)
