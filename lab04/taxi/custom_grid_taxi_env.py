"""
Własne środowisko Gymnasium: taxi na siatce 5×5 z przeszkodami (jak problem Diettericha),
zaimplementowane od zera — nie jest to podklasa gymnasium.envs.toy_text.taxi.TaxiEnv.
"""

from __future__ import annotations

from contextlib import closing
from io import StringIO
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces, utils

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

LOCS: list[tuple[int, int]] = [(0, 0), (0, 4), (4, 0), (4, 3)]

ENV_ID = "CustomGridTaxi-v0"


class CustomGridTaxiEnv(gym.Env):
    """
    Taksówka zbiera pasażera w jednym z czterech punktów (R,G,Y,B) i dowozi do wylosowanego celu.
    Obserwacja (domyślnie): indeks stanu 0..499 (jak w klasycznym Taxi-v3), wygodny dla PPO/DQN.
    Opcjonalnie: wektor znormalizowany (``observation_mode="vector"``) do eksperymentów.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        observation_mode: Literal["discrete", "vector"] = "discrete",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.desc = np.asarray(MAP, dtype="c")
        self.max_row = 4
        self.max_col = 4
        self.last_action: int | None = None

        self.action_space = spaces.Discrete(6)
        if observation_mode == "discrete":
            self.observation_space = spaces.Discrete(500)
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            )

        self.taxi_row = 0
        self.taxi_col = 0
        self.passenger_idx = 0
        self.destination_idx = 0

    # --- logika nagród / przejść (sucha wersja bez tabeli P) ---

    def _pickup(
        self, taxi_loc: tuple[int, int], pass_idx: int, reward: float
    ) -> tuple[int, float]:
        if pass_idx < 4 and taxi_loc == LOCS[pass_idx]:
            return 4, reward
        return pass_idx, -10.0

    def _dropoff(
        self,
        taxi_loc: tuple[int, int],
        pass_idx: int,
        dest_idx: int,
        default_reward: float,
    ) -> tuple[int, float, bool]:
        if taxi_loc == LOCS[dest_idx] and pass_idx == 4:
            return dest_idx, 20.0, True
        if taxi_loc in LOCS and pass_idx == 4:
            return LOCS.index(taxi_loc), default_reward, False
        return pass_idx, -10.0, False

    def _state_index(self) -> int:
        return int(
            ((self.taxi_row * 5 + self.taxi_col) * 5 + self.passenger_idx) * 4
            + self.destination_idx
        )

    def _get_obs(self) -> int | np.ndarray:
        if self.observation_mode == "discrete":
            return self._state_index()
        return np.array(
            [
                self.taxi_row / self.max_row,
                self.taxi_col / self.max_col,
                self.passenger_idx / 4.0,
                self.destination_idx / 3.0,
            ],
            dtype=np.float32,
        )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(6, dtype=np.int8)
        r, c = self.taxi_row, self.taxi_col
        if r < self.max_row:
            mask[0] = 1
        if r > 0:
            mask[1] = 1
        if c < self.max_col and self.desc[1 + r, 2 * c + 2] == b":":
            mask[2] = 1
        if c > 0 and self.desc[1 + r, 2 * c] == b":":
            mask[3] = 1
        if self.passenger_idx < 4 and (r, c) == LOCS[self.passenger_idx]:
            mask[4] = 1
        if self.passenger_idx == 4 and (
            (r, c) == LOCS[self.destination_idx] or (r, c) in LOCS
        ):
            mask[5] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        """Maska boolowska wymagana przez sb3_contrib ActionMasker."""
        return self._action_mask().astype(bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.passenger_idx = int(self.np_random.integers(0, 4))
        dest_choices = [d for d in range(4) if d != self.passenger_idx]
        self.destination_idx = int(self.np_random.choice(dest_choices))
        self.taxi_row = int(self.np_random.integers(0, 5))
        self.taxi_col = int(self.np_random.integers(0, 5))
        self.last_action = None
        obs = self._get_obs()
        return obs, {"prob": 1.0, "action_mask": self._action_mask()}

    def step(self, action: int):
        row, col = self.taxi_row, self.taxi_col
        p_idx, d_idx = self.passenger_idx, self.destination_idx
        new_row, new_col = row, col
        new_pass = p_idx
        reward = -1.0
        terminated = False

        if action == 0:
            new_row = min(row + 1, self.max_row)
        elif action == 1:
            new_row = max(row - 1, 0)
        elif action == 2:
            if self.desc[1 + row, 2 * col + 2] == b":":
                new_col = min(col + 1, self.max_col)
        elif action == 3:
            if self.desc[1 + row, 2 * col] == b":":
                new_col = max(col - 1, 0)
        elif action == 4:
            new_pass, reward = self._pickup((row, col), p_idx, reward)
        elif action == 5:
            new_pass, reward, terminated = self._dropoff(
                (row, col), p_idx, d_idx, reward
            )

        self.taxi_row, self.taxi_col = new_row, new_col
        self.passenger_idx = new_pass
        self.last_action = action

        obs = self._get_obs()
        info = {"prob": 1.0, "action_mask": self._action_mask()}
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        return self._render_text()

    def _render_text(self) -> str:
        desc = self.desc.copy().tolist()
        outfile = StringIO()
        out = [[c.decode("utf-8") for c in line] for line in desc]

        def ul(x: str) -> str:
            return "_" if x == " " else x

        r, c, pass_idx, dest_idx = (
            self.taxi_row,
            self.taxi_col,
            self.passenger_idx,
            self.destination_idx,
        )

        if pass_idx < 4:
            out[1 + r][2 * c + 1] = utils.colorize(
                out[1 + r][2 * c + 1], "yellow", highlight=True
            )
            pi, pj = LOCS[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:
            out[1 + r][2 * c + 1] = utils.colorize(
                ul(out[1 + r][2 * c + 1]), "green", highlight=True
            )

        di, dj = LOCS[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.last_action is not None:
            names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            outfile.write(f"  ({names[self.last_action]})\n")
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()


def register_custom_grid_taxi() -> None:
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="custom_grid_taxi_env:CustomGridTaxiEnv",
        max_episode_steps=200,
    )


register_custom_grid_taxi()
