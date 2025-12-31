import copy
from dataclasses import dataclass, field
from functools import cached_property
from random import Random

import numpy as np

from kata12.types import Env2D, Step

EMPTY = 0
WALL = 1
AGENT = 2
GOAL = 3


@dataclass
class MazeState:
    grid: list[list[int]]
    """The grid, 0 for empty, 1 for wall, 2 for agent, 3 for end"""

    x: int
    y: int
    goal_x: int
    goal_y: int
    max_steps: int
    timestep: int = 0

    def __post_init__(self):
        self.grid[self.x][self.y] = AGENT
        self.grid[self.goal_x][self.goal_y] = GOAL

    @cached_property
    def n(self) -> int:
        return len(self.grid)

    @property
    def obs(self) -> list[int]:
        return [col for row in self.grid for col in row]

    def __str__(self) -> str:
        result = "".join(["".join(str(i) for i in row) + "\n" for row in self.grid])
        return result

    def to_img(self):
        color_map = {
            EMPTY: [255, 255, 255],
            WALL: [0, 0, 0],
            AGENT: [255, 0, 0],
            GOAL: [0, 255, 0],
        }

        img = np.zeros((self.n, self.n, 3), dtype=np.uint8)
        for i in range(self.n):
            for j in range(self.n):
                img[i, j] = color_map[self.grid[i][j]]

        return img

    @property
    def dist_to_goal(self) -> int:
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def step(self, action: int) -> Step:
        self.timestep += 1

        actions = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        dx, dy = actions[action]

        old_dist = self.dist_to_goal

        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < self.n and 0 <= ny < self.n and self.grid[nx][ny] in [EMPTY, GOAL]:
            self.grid[self.x][self.y] = EMPTY
            self.x, self.y = nx, ny
            self.grid[self.x][self.y] = AGENT

        new_dist = self.dist_to_goal

        terminated = (self.goal_x, self.goal_y) == (self.x, self.y)
        truncated = (self.timestep >= self.max_steps) and not terminated

        if terminated:
            reward = 0.0
        elif new_dist == old_dist:
            reward = -0.011
        elif new_dist > old_dist:
            reward = -0.01
        else:
            reward = 0.0

        return Step(
            observation=self.obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )


@dataclass
class MazeEnv(Env2D):
    n: int = 7
    fill_prob: float | None = 0.5
    max_steps_scale: int = 2
    randomize_agent: bool = False
    max_repeats: int = 1
    state: MazeState | None = None
    rng: Random = field(default_factory=lambda: Random(0))
    initial_state: MazeState | None = None
    repeats: int | None = None

    @property
    def max_steps(self):
        return self.n * 2 * self.max_steps_scale

    @property
    def input_dim(self):
        return self.n, self.n

    @property
    def max_input(self):
        return 4

    @property
    def max_action(self):
        return 4

    def __str__(self) -> str:
        return str(self.state)

    def _random_monotone_path(self) -> list[tuple[int, int]]:
        moves = [(0, 1)] * (self.n - 1) + [(1, 0)] * (self.n - 1)
        self.rng.shuffle(moves)

        x, y = 0, 0
        path = [(x, y)]
        for a, b in moves:
            x, y = x + a, y + b
            path.append((x, y))
        return path

    def make_state(self) -> MazeState:
        path = self._random_monotone_path()
        path_set = set(path)

        grid = [[0] * self.n for _ in range(self.n)]

        fill_prob = self.fill_prob if self.fill_prob is not None else self.rng.random()

        for x in range(self.n):
            for y in range(self.n):
                if (x, y) not in path_set:
                    grid[x][y] = 1 if self.rng.random() < fill_prob else 0

        if self.randomize_agent:
            x, y = self.rng.choice(path[:-1])
        else:
            x, y = path[0]
        return MazeState(
            grid,
            x=x,
            y=y,
            goal_x=path[-1][0],
            goal_y=path[-1][1],
            max_steps=self.max_steps,
        )

    def reset(self) -> Step:
        if self.repeats == None or self.repeats == self.max_repeats:
            self.repeats = 0
            self.initial_state = self.make_state()

        assert self.initial_state is not None
        self.state = copy.deepcopy(self.initial_state)
        self.repeats += 1
        return Step(observation=self.state.obs)

    def step(self, action) -> Step:
        if self.state is None:
            raise ValueError("Not reset!")

        return self.state.step(action)

    def to_img(self) -> np.ndarray:
        """Convert the current maze state to an RGB image array."""
        if self.state is None:
            raise ValueError("Environment not reset!")
        return self.state.to_img()
