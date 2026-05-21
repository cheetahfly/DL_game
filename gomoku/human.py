"""
人类玩家接口
"""


class HumanPlayer:
    """人类玩家类"""

    def __init__(self, player=2):
        """
        初始化人类玩家
        player: 1(黑棋) 或 2(白棋)
        """
        self.player = player
        self.last_move = None

    def get_move(self, game):
        """
        获取人类落子位置
        game: GomokuGame对象
        返回: (row, col) 落子位置，或 None
        """
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    pos = game.get_board_pos(event.pos)
                    if pos:
                        row, col = pos
                        if game.is_valid_move(row, col):
                            self.last_move = pos
                            return pos

        return self.last_move

    def reset(self):
        """重置状态"""
        self.last_move = None
