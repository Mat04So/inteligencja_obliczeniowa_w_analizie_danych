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

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

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
        self.steps = 0
        self.episode_reward = 0.0

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

        self._cell = 96
        self._hud_h = 96
        self._win_w = self._cell * 5
        self._win_h = self._cell * 5 + self._hud_h
        self._pygame = None
        self._window = None
        self._clock = None
        self._fonts: dict[str, Any] = {}

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
        self.steps = 0
        self.episode_reward = 0.0
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
        self.steps += 1
        self.episode_reward += reward

        obs = self._get_obs()
        info = {"prob": 1.0, "action_mask": self._action_mask()}
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_text()
        return self._render_pygame()

    def close(self):
        if self._window is not None and self._pygame is not None:
            try:
                if self.render_mode == "human":
                    self._pygame.display.quit()
                self._pygame.quit()
            except Exception:
                pass
        self._window = None
        self._clock = None
        self._fonts = {}
        self._pygame = None

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

    def _render_pygame(self):
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "Tryb 'human'/'rgb_array' wymaga pakietu pygame. "
                "Zainstaluj: pip install pygame"
            ) from exc

        if self._window is None:
            self._pygame = pygame
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CustomGridTaxi-v0")
                self._window = pygame.display.set_mode((self._win_w, self._win_h))
            else:
                self._window = pygame.Surface((self._win_w, self._win_h))
            self._clock = pygame.time.Clock()
            self._fonts["small"] = pygame.font.SysFont("Menlo,Monaco,monospace", 16)
            self._fonts["mid"] = pygame.font.SysFont("Menlo,Monaco,monospace", 20, bold=True)
            self._fonts["big"] = pygame.font.SysFont("Menlo,Monaco,monospace", 26, bold=True)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        canvas = self._window
        cs = self._cell
        loc_colors = {
            0: (220, 60, 60),
            1: (60, 180, 75),
            2: (240, 200, 40),
            3: (60, 120, 220),
        }
        loc_names = ["R", "G", "Y", "B"]

        canvas.fill((250, 250, 252))

        grid_bg = (245, 245, 248)
        grid_line = (215, 215, 220)
        for r in range(5):
            for c in range(5):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                pygame.draw.rect(canvas, grid_bg, rect)
                pygame.draw.rect(canvas, grid_line, rect, 1)

        wall_color = (40, 40, 50)
        wall_w = 5
        pygame.draw.rect(canvas, wall_color, (0, 0, self._win_w, cs * 5), wall_w)
        for r in range(5):
            for c in range(4):
                if self.desc[1 + r, 2 * c + 2] == b"|":
                    x = (c + 1) * cs
                    pygame.draw.line(
                        canvas, wall_color, (x, r * cs), (x, (r + 1) * cs), wall_w
                    )

        for i, (lr, lc) in enumerate(LOCS):
            color = loc_colors[i]
            cx = lc * cs + cs // 2
            cy = lr * cs + cs // 2
            pygame.draw.circle(canvas, color, (cx, cy), cs // 3, 4)
            label = self._fonts["big"].render(loc_names[i], True, color)
            canvas.blit(label, (lc * cs + 8, lr * cs + 4))

        dr, dc = LOCS[self.destination_idx]
        goal_rect = pygame.Rect(dc * cs + 6, dr * cs + 6, cs - 12, cs - 12)
        pygame.draw.rect(canvas, (170, 60, 200), goal_rect, 4, border_radius=10)
        flag = self._fonts["small"].render("CEL", True, (170, 60, 200))
        canvas.blit(flag, (dc * cs + cs - flag.get_width() - 8, dr * cs + cs - flag.get_height() - 6))

        if self.passenger_idx < 4:
            pr, pc = LOCS[self.passenger_idx]
            cx = pc * cs + cs // 2
            cy = pr * cs + cs // 2 + 8
            pygame.draw.circle(canvas, (35, 35, 45), (cx, cy + 6), cs // 8)
            pygame.draw.circle(canvas, (245, 220, 200), (cx, cy - 6), cs // 12)
            pygame.draw.circle(canvas, (35, 35, 45), (cx, cy - 6), cs // 12, 2)

        tr, tc = self.taxi_row, self.taxi_col
        pad = cs // 6
        taxi_rect = pygame.Rect(tc * cs + pad, tr * cs + pad, cs - 2 * pad, cs - 2 * pad)
        on_board = self.passenger_idx == 4
        body = (60, 200, 110) if on_board else (250, 200, 50)
        pygame.draw.rect(canvas, body, taxi_rect, border_radius=12)
        pygame.draw.rect(canvas, (30, 30, 35), taxi_rect, 2, border_radius=12)
        win_rect = pygame.Rect(
            taxi_rect.x + 6,
            taxi_rect.y + 6,
            taxi_rect.w - 12,
            taxi_rect.h // 2 - 4,
        )
        pygame.draw.rect(canvas, (200, 230, 250), win_rect, border_radius=6)
        pygame.draw.rect(canvas, (30, 30, 35), win_rect, 2, border_radius=6)
        t_label = self._fonts["mid"].render("TAXI", True, (30, 30, 35))
        canvas.blit(
            t_label,
            t_label.get_rect(
                center=(taxi_rect.centerx, taxi_rect.centery + taxi_rect.h // 4)
            ),
        )

        hud = pygame.Rect(0, cs * 5, self._win_w, self._hud_h)
        pygame.draw.rect(canvas, (28, 28, 36), hud)
        pygame.draw.line(canvas, wall_color, (0, cs * 5), (self._win_w, cs * 5), wall_w)

        pass_state = (
            "pasażer: w taksówce"
            if on_board
            else f"pasażer: {loc_names[self.passenger_idx]}"
        )
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        last = action_names[self.last_action] if self.last_action is not None else "—"
        lines = [
            f"{pass_state}    cel: {loc_names[self.destination_idx]}",
            f"krok: {self.steps}    suma nagród: {self.episode_reward:.1f}",
            f"ostatnia akcja: {last}",
        ]
        for i, text in enumerate(lines):
            surf = self._fonts["small"].render(text, True, (235, 235, 240))
            canvas.blit(surf, (12, cs * 5 + 8 + i * 22))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        rgb = pygame.surfarray.pixels3d(canvas)
        return np.transpose(np.array(rgb), axes=(1, 0, 2))


def register_custom_grid_taxi() -> None:
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point="custom_grid_taxi_env:CustomGridTaxiEnv",
        max_episode_steps=200,
    )


register_custom_grid_taxi()
