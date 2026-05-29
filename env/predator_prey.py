"""
File: env/predator_prey.py

Grid: H x W cells (default 10x10), toroidal (wrap-around edges).
Agents: 2. Agent 0 = predator, Agent 1 = prey.
Observation: Each agent sees a local 5x5 window centered on itself,
             flattened to 25 cells, plus its own (x,y) and opponent (x,y)
             normalized to [0,1]. Total obs dim = 25 + 4 = 29.
Actions: Discrete(5) — stay, up, down, left, right.
Reward:
  - Predator: +1.0 on capture, -0.01 per step (time penalty)
  - Prey: +0.01 per step survived, -1.0 on capture
Episode ends: capture OR max_steps (default 200).
Reset: agents placed at random non-overlapping positions.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PredatorPreyEnv(gym.Env):
    """1v1 toroidal grid-world predator-prey environment."""

    metadata = {"render_modes": ["human"]}

    # Action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right
    ACTION_DELTAS = {
        0: np.array([0, 0]),   # stay
        1: np.array([-1, 0]),  # up
        2: np.array([1, 0]),   # down
        3: np.array([0, -1]),  # left
        4: np.array([0, 1]),   # right
    }

    def __init__(self, grid_size: int = 10, max_steps: int = 200) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Observation: 5x5 local window (25) + own (x,y) + opp (x,y) = 29
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(29,), dtype=np.float32
        )
        # Each agent has Discrete(5) actions
        self.action_space = spaces.Discrete(5)

        # State
        self.predator_pos: np.ndarray | None = None
        self.prey_pos: np.ndarray | None = None
        self.current_step: int = 0
        self._rng: np.random.Generator | None = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Place agents at random non-overlapping positions
        while True:
            self.predator_pos = self._rng.integers(0, self.grid_size, size=2)
            self.prey_pos = self._rng.integers(0, self.grid_size, size=2)
            if not np.array_equal(self.predator_pos, self.prey_pos):
                break

        self.current_step = 0
        obs_pred = self._get_obs(self.predator_pos, self.prey_pos)
        obs_prey = self._get_obs(self.prey_pos, self.predator_pos)
        info = self._make_info(captured=False)
        return (obs_pred, obs_prey), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, actions: tuple[int, int]
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[float, float],
        bool,
        bool,
        dict,
    ]:
        action_pred, action_prey = actions

        # Move agents (toroidal wrap)
        self.predator_pos = (
            self.predator_pos + self.ACTION_DELTAS[action_pred]
        ) % self.grid_size
        self.prey_pos = (
            self.prey_pos + self.ACTION_DELTAS[action_prey]
        ) % self.grid_size

        self.current_step += 1

        # Check capture
        captured = bool(np.array_equal(self.predator_pos, self.prey_pos))

        # Compute rewards
        if captured:
            reward_pred = 1.0
            reward_prey = -1.0
        else:
            reward_pred = -0.01
            reward_prey = 0.01

        # Termination / truncation
        terminated = captured
        truncated = (not terminated) and (self.current_step >= self.max_steps)

        # Observations
        obs_pred = self._get_obs(self.predator_pos, self.prey_pos)
        obs_prey = self._get_obs(self.prey_pos, self.predator_pos)

        info = self._make_info(captured=captured)

        return (obs_pred, obs_prey), (reward_pred, reward_prey), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(
        self, own_pos: np.ndarray, opp_pos: np.ndarray
    ) -> np.ndarray:
        """Build a 29-dim observation for one agent.

        Components:
          - 25-dim: 5x5 local grid window (flattened), values in {0, 0.5, 1}
            mapped from grid cell codes: 0=empty → 0.0, 1=predator → 0.5, 2=prey → 1.0
          - 4-dim: (own_x, own_y, opp_x, opp_y) each normalised to [0,1]
        """
        window = self._local_window(own_pos)
        coords = np.array(
            [
                own_pos[0] / max(self.grid_size - 1, 1),
                own_pos[1] / max(self.grid_size - 1, 1),
                opp_pos[0] / max(self.grid_size - 1, 1),
                opp_pos[1] / max(self.grid_size - 1, 1),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([window, coords]).astype(np.float32)
        assert obs.shape == (29,), f"Observation shape mismatch: {obs.shape}"
        return obs

    def _local_window(self, center: np.ndarray) -> np.ndarray:
        """Extract a 5x5 window centered on *center*, toroidal, flattened.

        Cell encoding (normalised to [0,1]):
          empty    → 0.0
          predator → 0.5
          prey     → 1.0
        """
        window = np.zeros((5, 5), dtype=np.float32)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r = (center[0] + dr) % self.grid_size
                c = (center[1] + dc) % self.grid_size
                if np.array_equal(np.array([r, c]), self.predator_pos):
                    window[dr + 2, dc + 2] = 0.5
                elif np.array_equal(np.array([r, c]), self.prey_pos):
                    window[dr + 2, dc + 2] = 1.0
                # else stays 0.0
        return window.flatten()

    # ------------------------------------------------------------------
    # Render (ASCII)
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:  # noqa: ARG002
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        pr, pc = self.predator_pos
        yr, yc = self.prey_pos
        if np.array_equal(self.predator_pos, self.prey_pos):
            grid[pr][pc] = "X"
        else:
            grid[pr][pc] = "P"
            grid[yr][yc] = "Y"
        header = "  " + " ".join(str(c) for c in range(self.grid_size))
        print(header)
        for r in range(self.grid_size):
            print(f"{r} " + " ".join(grid[r]))
        print(f"Step: {self.current_step}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_info(self, captured: bool) -> dict:
        return {
            "captured": captured,
            "steps": self.current_step,
            "predator_pos": tuple(self.predator_pos),
            "prey_pos": tuple(self.prey_pos),
        }
