"""
贪吃蛇游戏环境类 - 状态空间11维, 动作空间4个方向
"""

import numpy as np
import pygame
import sys


class SnakeGame:
    """贪吃蛇游戏环境"""
    ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上右下左

    def __init__(self, grid_size=20, cell_size=20, speed=50, gui=True):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.speed = speed
        self.gui = gui
        self.screen = None

        if gui:
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * cell_size,) * 2)
            pygame.display.set_caption('贪吃蛇 DQN AI')
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self) -> np.ndarray:
        """重置游戏,返回初始状态"""
        c = self.grid_size // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = 3  # 初始向右
        self.food = self._spawn_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def _spawn_food(self) -> tuple:
        """生成食物"""
        while True:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos

    def _get_state(self) -> np.ndarray:
        """获取11维状态向量"""
        head = self.snake[0]
        state = np.zeros(11, dtype=np.float32)

        # [0-3] 四个方向到障碍物的距离
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            dist, x, y = 1, head[0], head[1]
            while True:
                x, y = x + dx, y + dy
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or (x, y) in self.snake:
                    break
                dist += 1
            state[i] = dist / self.grid_size

        # [4-7] 四个方向是否有食物
        fx, fy = self.food
        hx, hy = head
        if hy > fy and hx == fx: state[4] = 1
        if hy < fy and hx == fx: state[5] = 1
        if hx > fx and hy == fy: state[6] = 1
        if hx < fx and hy == fy: state[7] = 1

        # [8-9] 食物相对蛇头方向
        state[8] = (fx - hx) / self.grid_size
        state[9] = (fy - hy) / self.grid_size

        # [10] 蛇长归一化
        state[10] = len(self.snake) / (self.grid_size * self.grid_size)
        return state

    def step(self, action: int) -> tuple:
        """执行动作,返回(next_state, reward, done)"""
        # 禁止180度转向
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites.get(self.direction):
            action = self.direction

        self.direction = action
        dx, dy = self.ACTIONS[action]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # 撞墙或自撞
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            return self._get_state(), -10, True

        self.snake.insert(0, new_head)

        # 吃食物
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            return self._get_state(), 10, False

        self.snake.pop()
        return self._get_state(), 0, False

    def render(self):
        """Pygame渲染"""
        if not self.gui:
            return

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((50, 50, 50))

        # 网格
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (80, 80, 80),
                           (0, i * self.cell_size), (self.grid_size * self.cell_size, i * self.cell_size))
            pygame.draw.line(self.screen, (80, 80, 80),
                           (i * self.cell_size, 0), (i * self.cell_size, self.grid_size * self.cell_size))

        # 食物
        f_rect = pygame.Rect(self.food[0] * self.cell_size + 2,
                            self.food[1] * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(self.screen, (200, 0, 0), f_rect)

        # 蛇
        for i, (x, y) in enumerate(self.snake):
            color = (0, 200, 0) if i == 0 else (0, 150, 0)
            rect = pygame.Rect(x * self.cell_size + 2, y * self.cell_size + 2,
                             self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, color, rect)

        # 分数
        font = pygame.font.Font(pygame.font.match_font('microsoftyahei'), 24)
        self.screen.blit(font.render(f'分数: {self.score}', True, (255, 255, 255)), (5, 5))
        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        if self.gui:
            pygame.quit()
