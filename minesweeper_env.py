"""Custom Gymnasium Minesweeper environment (see architecture_overview.md).
 
Game rules ported from dastratakos/Minesweeper (board.py):
  * first-click safety by relocating the mine under the first reveal,
  * zero-cascade of empty tiles (here an iterative BFS instead of recursion),
  * win when every non-mine tile has been revealed.
 
RL additions from the architecture overview:
  * 11 x H x W binary observation (one-hot numbers 0-8, covered, flagged),
  * overloaded action space: target covered -> Reveal, target number -> Chord,
  * `action_masks()` marking legal actions,
  * dense reward: +0.1 per safe tile revealed, -1 on a mine, +1 (+0.1*m) on a win.
"""
from __future__ import annotations
 
from collections import deque
 
import gymnasium as gym
import numpy as np
 
COVERED = -1  # sentinel stored in `visible` for unrevealed tiles
 
 
class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}
 
    SAFE_REWARD = 0.1
    MINE_REWARD = -1.0
    WIN_BONUS = 1.0
 
    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        num_mines: int = 10,
        render_mode: str | None = None,
    ):
        super().__init__()
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if not 1 <= num_mines <= width * height - 1:
            raise ValueError("num_mines must be in [1, width*height - 1]")
 
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.total_safe = width * height - num_mines
        self.render_mode = render_mode
 
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(11, height, width), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(width * height)
 
        # Precomputed in-bounds neighbour lists.
        self._nbrs = [
            [self._compute_neighbors(r, c) for c in range(width)] for r in range(height)
        ]
 
        # Internal state (populated by reset()).
        self.board = np.zeros((height, width), dtype=bool)  # True = mine
        self.adjacent = np.zeros((height, width), dtype=np.int8)  # mines around tile
        self.visible = np.full((height, width), COVERED, dtype=np.int8)
        self.flags = np.zeros((height, width), dtype=bool)
        self.first_click = True
        self.revealed_safe = 0
        self.terminated = False
        self.won = False
        self.lost = False
        self.detonated: list[tuple[int, int]] = []
 
    # ------------------------------------------------------------------ helpers
    def _compute_neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < self.height and 0 <= c + dc < self.width
        ]
 
    def _neighbor_sum(self, arr: np.ndarray) -> np.ndarray:
        """Per-tile sum of the 8 surrounding values (zero padded)."""
        p = np.pad(arr.astype(np.int8), 1)
        h, w = self.height, self.width
        out = np.zeros((h, w), dtype=np.int8)
        for dr in range(3):
            for dc in range(3):
                if dr == 1 and dc == 1:
                    continue
                out += p[dr : dr + h, dc : dc + w]
        return out
 
    def _update_adjacent(self) -> None:
        self.adjacent = self._neighbor_sum(self.board)
 
    def _relocate_mine(self, r: int, c: int) -> None:
        """First-click safety: move the mine at (r, c) to a random empty tile."""
        empty = np.flatnonzero(~self.board.ravel())  # (r, c) is a mine, so excluded
        target = int(self.np_random.choice(empty))
        self.board.flat[target] = True
        self.board[r, c] = False
        self._update_adjacent()
 
    def _flood_reveal(self, r: int, c: int) -> int:
        """Reveal (r, c); BFS through zero tiles. Returns #newly revealed safe tiles."""
        count = 0
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            if self.visible[cr, cc] != COVERED or self.flags[cr, cc]:
                continue
            self.visible[cr, cc] = self.adjacent[cr, cc]
            count += 1
            if self.adjacent[cr, cc] == 0:
                queue.extend(self._nbrs[cr][cc])
        self.revealed_safe += count
        return count
 
    # ----------------------------------------------------------- public helpers
    def toggle_flag(self, r: int, c: int) -> bool:
        """Flag / unflag a covered tile. Not an agent action (the action space is
        reveal/chord only); provided for humans, solvers and tests. Returns True
        if the flag state changed."""
        if self.terminated or self.visible[r, c] != COVERED:
            return False
        self.flags[r, c] = not self.flags[r, c]
        return True
 
    def action_masks(self) -> np.ndarray:
        """Flat boolean mask, length H*W, index = r*W + c. True = legal action."""
        if self.terminated:
            return np.zeros(self.height * self.width, dtype=bool)
 
        covered = self.visible == COVERED
        reveal = covered & ~self.flags  # flagged tiles are protected
 
        flag_count = self._neighbor_sum(self.flags)
        open_count = self._neighbor_sum(reveal)
        chord = (
            (self.visible >= 1)
            & (flag_count == self.visible)
            & (open_count > 0)  # something would actually be opened
        )
        return (reveal | chord).ravel()
 
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for k in range(9):
            obs[k] = self.visible == k
        obs[9] = self.visible == COVERED  # flagged tiles are still unrevealed
        obs[10] = self.flags
        return obs
 
    def _info(self, **extra) -> dict:
        return {
            "action_mask": self.action_masks(),
            "safe_revealed": self.revealed_safe,
            **extra,
        }
 
    # ---------------------------------------------------------------- gym API
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros((self.height, self.width), dtype=bool)
        mine_idx = self.np_random.choice(
            self.height * self.width, size=self.num_mines, replace=False
        )
        self.board.flat[mine_idx] = True
        self._update_adjacent()
 
        self.visible = np.full((self.height, self.width), COVERED, dtype=np.int8)
        self.flags = np.zeros((self.height, self.width), dtype=bool)
        self.first_click = True
        self.revealed_safe = 0
        self.terminated = False
        self.won = False
        self.lost = False
        self.detonated = []
        return self._get_obs(), self._info()
 
    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("step() called on a finished episode; call reset().")
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"action {action} out of range")
        if not self.action_masks()[action]:
            raise ValueError(f"action {action} is not legal in the current state")
 
        r, c = divmod(action, self.width)
        hit_mine = False
        m = 0
 
        if self.visible[r, c] == COVERED:
            action_type = "reveal"
            if self.first_click:
                self.first_click = False
                if self.board[r, c]:
                    self._relocate_mine(r, c)
            if self.board[r, c]:
                hit_mine = True
                self.detonated = [(r, c)]
            else:
                m = self._flood_reveal(r, c)
        else:
            action_type = "chord"
            targets = [
                (nr, nc)
                for nr, nc in self._nbrs[r][c]
                if self.visible[nr, nc] == COVERED and not self.flags[nr, nc]
            ]
            mines = [t for t in targets if self.board[t]]  # wrong flags => mines here
            if mines:
                hit_mine = True
                self.detonated = mines
            else:
                m = sum(self._flood_reveal(tr, tc) for tr, tc in targets)
 
        if hit_mine:
            reward = self.MINE_REWARD
            self.terminated = True
            self.lost = True
        elif self.revealed_safe == self.total_safe:
            reward = self.WIN_BONUS + self.SAFE_REWARD * m
            self.terminated = True
            self.won = True
        else:
            reward = self.SAFE_REWARD * m
 
        info = self._info(action_type=action_type, won=self.won, lost=self.lost)
        # Only reachable if the player flagged safe tiles wrongly and boxed itself in.
        truncated = (not self.terminated) and not info["action_mask"].any()
        return self._get_obs(), float(reward), self.terminated, truncated, info
 
    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        rows = []
        for r in range(self.height):
            line = []
            for c in range(self.width):
                v = self.visible[r, c]
                if (r, c) in self.detonated:
                    ch = "*"
                elif v == COVERED:
                    if self.flags[r, c]:
                        ch = "F"
                    elif self.terminated and self.board[r, c]:
                        ch = "x"
                    else:
                        ch = "."
                else:
                    ch = " " if v == 0 else str(v)
                line.append(ch)
            rows.append(" ".join(line))
        return "\n".join(rows)