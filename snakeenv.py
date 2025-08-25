from enum import Enum
import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from collections import deque
from io import StringIO

class Actions(Enum):
    NONE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 1

class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LEFT = np.array([-1, 0])
    RIGHT = np.array([1, 0])
    UP = np.array([0, -1])
    DOWN = np.array([0, 1])

    EMPTY_CELL = 0
    OCCUPIED_CELL = 1

    INFO = {}

    def __init__(self, render_mode=None, size_x=5, size_y=5):
        self._size_x = size_x
        self._size_y = size_y
        self.window_size_x = 512
        self.window_size_y = self.window_size_x * size_y // size_x

        self._collisions = np.zeros((size_x, size_y), dtype=np.bool_)
        self._snake_queue = deque()
        self._direction = np.array([0, 0], dtype=np.int_)
        self._head_position = None
        self._food_position = None
        self._steps_since_last_food = 0

        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Dict(
            {
                "collisions": gym.spaces.Box(low=0, high=1, shape=(size_x, size_y), dtype=np.int_),
                "head_position_x": gym.spaces.Box(low=0, high=size_x - 1, shape=(2,), dtype=np.int_),
                "head_position_y": gym.spaces.Box(low=0, high=size_y - 1, shape=(2,), dtype=np.int_),
                "food_position_x": gym.spaces.Box(low=0, high=size_x - 1, shape=(2,), dtype=np.int_),
                "food_position_y": gym.spaces.Box(low=0, high=size_y - 1, shape=(2,), dtype=np.int_),
                "direction_x": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int_),
                "direction_y": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int_),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._direction = self.RIGHT
        self._head_position = np.array([self._size_x // 2, self._size_y // 2])
        self._snake_queue = deque()
        self._snake_queue.append(self._head_position)
        self._collisions.fill(self.EMPTY_CELL)
        self._collisions[tuple(self._head_position)] = self.OCCUPIED_CELL
        self._position_food()

        self._render_human_if_needed()

        return self._get_observation(), self.INFO

    def step(self, action):
        if action != Actions.NONE.value:
            if action == Actions.TURN_LEFT.value:
                self._turn(-1)
            elif action == Actions.TURN_RIGHT.value:
                self._turn(1)

        terminated = False
        truncated = False
        reward = 0
        next_head = self._head_position + self._direction

        if self._steps_since_last_food >= 5 * self._size_x * self._size_y:
            truncated = True
            reward -= 1

        if next_head[0] < 0 or next_head[0] >= self._size_x or next_head[1] < 0 or next_head[1] >= self._size_y:
            terminated = True
            reward -= 1
        elif self._collisions[tuple(next_head)] == self.OCCUPIED_CELL:
            terminated = True
            reward -= 1
        elif np.array_equal(next_head, self._food_position):
            self._head_position = next_head
            self._snake_queue.append(next_head)
            self._collisions[tuple(next_head)] = self.OCCUPIED_CELL
            self._position_food()
            reward += 1
            if self._is_full():
                reward += 1
                terminated = True
        else:
            tail = self._snake_queue.popleft()
            self._collisions[tuple(tail)] = self.EMPTY_CELL
            self._head_position = next_head
            self._snake_queue.append(next_head)
            self._collisions[tuple(next_head)] = self.OCCUPIED_CELL
            self._steps_since_last_food += 1

        self._render_human_if_needed()

        return self._get_observation(), reward, terminated, truncated, self.INFO

    def render(self):
        if self.render_mode == "ansi":
            return self._get_ascii_render()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._get_canvas_render())), axes=(1, 0, 2)
            )
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_observation(self):
        return {
            "collisions": self._collisions.copy(),
            "head_position_x": self._head_position[0],
            "head_position_y": self._head_position[1],
            "food_position_x": self._food_position[0],
            "food_position_y": self._food_position[1],
            "direction_x": self._direction[0],
            "direction_y": self._direction[1],
        }

    def _is_full(self):
        return self._get_snake_length() == self._size_x * self._size_y

    def _get_snake_length(self):
        return len(self._snake_queue)

    def _position_food(self):
        self._steps_since_last_food = 0

        empty_cells_count = self._size_x * self._size_y - self._get_snake_length()
        if empty_cells_count <= 0:
            return

        food_remaining_index = self.np_random.integers(0, empty_cells_count)
        for x in range(self._size_x):
            for y in range(self._size_y):
                if self._collisions[x, y] == self.EMPTY_CELL:
                    if food_remaining_index == 0:
                        self._food_position = np.array([x, y])
                        return
                    food_remaining_index -= 1

    def _turn(self, direction):
        if self._direction[0] == 0:
            self._direction[0] = direction * self._direction[1]
            self._direction[1] = 0
        else:
            self._direction[1] = direction * -1 * self._direction[0]
            self._direction[0] = 0

    def _get_ascii_render(self):
        render_buffer = StringIO()
        for x in range(self._size_x):
            for y in range(self._size_y):
                if self._collisions[x, y] == self.EMPTY_CELL:
                    render_buffer.write("#")
                else:
                    if self._food_position[0] == x and self._food_position[1] == y:
                        render_buffer.write("o")
                    else:
                        render_buffer.write("x")
            render_buffer.write("\n")
        return render_buffer.getvalue()

    def _get_canvas_render(self):
        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill((0, 0, 0))

        pixel_size = self.window_size_x // self._size_x

        for x in range(self._size_x):
            for y in range(self._size_y):
                if self._collisions[x, y] == self.OCCUPIED_CELL:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            (x * pixel_size, y * pixel_size),
                            (pixel_size, pixel_size),
                        ),
                    )

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                (self._food_position[0] * pixel_size, self._food_position[1] * pixel_size),
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
