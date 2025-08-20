from enum import Enum
import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from collections import deque
from io import StringIO

class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LEFT = np.array([-1, 0])
    RIGHT = np.array([1, 0])
    UP = np.array([0, -1])
    DOWN = np.array([0, 1])

    EMPTY_CELL = 0
    FOOD_CELL = 2
    SNAKE_CELL = 1

    INFO = {}

    def __init__(self, render_mode=None, size_x=5, size_y=5):
        self._size_x = size_x
        self._size_y = size_y
        self.window_size_x = 512
        self.window_size_y = self.window_size_x * size_y // size_x

        self._grid = np.zeros((size_x, size_y), dtype=np.int_)
        self._snake_queue = deque()
        self._direction = np.array([0, 0], dtype=np.int_)

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            Actions.LEFT.value: self.LEFT,
            Actions.RIGHT.value: self.RIGHT,
            Actions.UP.value: self.UP,
            Actions.DOWN.value: self.DOWN
        }

        self.observation_space = gym.spaces.Dict(
            {
                "head_x": gym.spaces.Discrete(size_x),
                "head_y": gym.spaces.Discrete(size_y),
                "grid": gym.spaces.Box(0, 2, shape=(size_x, size_y), dtype=np.int_),
                "is_full": gym.spaces.Discrete(2),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_observation(self):
        head = self._get_head()
        return {
            "head_x": head[0],
            "head_y": head[1],
            "grid": self._grid.copy(),
            "is_full": self._is_full(),
        }

    def _is_full(self):
        return self._get_snake_length() == self._size_x * self._size_y

    def _get_head(self):
        return self._snake_queue[-1]

    def _get_snake_length(self):
        return len(self._snake_queue)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._direction = self._action_to_direction[int(self.np_random.integers(0, 4))]
        snake_head = np.array([self._size_x // 2, self._size_y // 2])
        self._snake_queue = deque()
        self._snake_queue.append(snake_head)
        self._grid.fill(self.EMPTY_CELL)
        self._grid[tuple(snake_head)] = self.SNAKE_CELL

        self._position_food()

        self._render_human_if_needed()

        return self._get_observation(), self.INFO

    def step(self, action):
        next_direction = self._action_to_direction[action]
        if (self._direction[0] != 0 and next_direction[0] != 0) or (self._direction[1] != 0 and next_direction[1] != 0):
            next_direction = self._direction
        self._direction = next_direction

        terminated = False
        truncated = False
        reward = 0
        next_head = self._get_head() + self._direction

        if next_head[0] < 0 or next_head[0] >= self._size_x or next_head[1] < 0 or next_head[1] >= self._size_y:
            terminated = True
            reward -= 100
        elif self._grid[tuple(next_head)] == self.SNAKE_CELL:
            terminated = True
            reward -= 100
        elif self._grid[tuple(next_head)] == self.FOOD_CELL:
            self._snake_queue.append(next_head)
            self._grid[tuple(next_head)] = self.SNAKE_CELL
            self._position_food()
            reward += 100
            if self._is_full():
                terminated = True
        else:
            tail = self._snake_queue.popleft()
            self._grid[tuple(tail)] = self.EMPTY_CELL
            self._snake_queue.append(next_head)
            self._grid[tuple(next_head)] = self.SNAKE_CELL

        self._render_human_if_needed()

        return self._get_observation(), reward, terminated, truncated, self.INFO

    def _position_food(self):
        empty_cells_count = self._size_x * self._size_y - self._get_snake_length()

        if empty_cells_count <= 0:
            return

        food_remaining_index = self.np_random.integers(0, empty_cells_count)

        for x in range(self._size_x):
            for y in range(self._size_y):
                if self._grid[x, y] == self.EMPTY_CELL:
                    if food_remaining_index == 0:
                        self._grid[x, y] = self.FOOD_CELL
                        return
                    food_remaining_index -= 1

    def render(self):
        if self.render_mode == "ansi":
            return self._get_ascii_render()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._get_canvas_render())), axes=(1, 0, 2)
            )
        return None

    def _get_ascii_render(self):
        render_buffer = StringIO()
        for x in range(self._size_x):
            for y in range(self._size_y):
                cell = self._grid[x, y]
                if cell == self.EMPTY_CELL:
                    render_buffer.write("#")
                elif cell == self.FOOD_CELL:
                    render_buffer.write("o")
                elif cell == self.SNAKE_CELL:
                    render_buffer.write("x")
            render_buffer.write("\n")
        return render_buffer.getvalue()

    def _get_canvas_render(self):
        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill((0, 0, 0))

        pixel_size = self.window_size_x // self._size_x

        for x in range(self._size_x):
            for y in range(self._size_y):
                cell = self._grid[x, y]
                if cell == self.FOOD_CELL:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            (x * pixel_size, y * pixel_size),
                            (pixel_size, pixel_size),
                        ),
                    )
                elif cell == self.SNAKE_CELL:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            (x * pixel_size, y * pixel_size),
                            (pixel_size, pixel_size),
                        ),
                    )
        return canvas

    def _render_human_if_needed(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = self._get_canvas_render()
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None