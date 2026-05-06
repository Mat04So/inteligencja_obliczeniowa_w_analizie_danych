"""
Własne środowisko Gymnasium: taxi na siatce 5×5 z DWOMA pasażerami.

Agent musi odebrać każdego pasażera z jego lokacji i dowieźć do celu.
Taxi przewozi jednego pasażera naraz, kolejność dowolna.

Obserwacja: Box(12,) — wektor znormalizowany do [0, 1].
Nagrody: +20 za pierwszą dostawę, +30 za drugą (bonus za ukończenie).
"""

from __future__ import annotations

from contextlib import closing
from io import StringIO
from typing import Any

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
LOC_NAMES = ["R", "G", "Y", "B"]

ENV_ID_2P = "CustomGridTaxi2P-v0"

# Pasażer 1 → niebieski, Pasażer 2 → pomarańczowy (pygame RGB)
P_COLORS = [(60, 120, 220), (240, 140, 40)]
P_DST_COLORS = [(100, 160, 255), (255, 180, 80)]


class CustomGridTaxi2PEnv(gym.Env):
    """
    Rozszerzona wersja CustomGridTaxi z dwoma pasażerami.

    Stan: pozycja taxi (5×5), status P1 i P2 (waiting/in_taxi/delivered),
          lokacje startowe i docelowe obu pasażerów (przypisane przez permutację 4 LOCS).

    Obserwacja: np.ndarray shape=(12,), dtype=float32, wszystko w [0, 1]:
        [taxi_r, taxi_c, p1_status, p1_src_r, p1_src_c, p1_dst_r, p1_dst_c,
                          p2_status, p2_src_r, p2_src_c, p2_dst_r, p2_dst_c]
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.desc = np.asarray(MAP, dtype="c")
        self.max_row = 4
        self.max_col = 4
        self.last_action: int | None = None
        self.steps = 0
        self.episode_reward = 0.0
        self._deliveries = 0

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.taxi_row = 0
        self.taxi_col = 0
        self.p1_src = 0
        self.p1_dst = 1
        self.p2_src = 2
        self.p2_dst = 3
        # 0=waiting at src, 1=in_taxi, 2=delivered
        self.p1_status = 0
        self.p2_status = 0

        self._cell = 96
        self._hud_h = 112
        self._win_w = self._cell * 5
        self._win_h = self._cell * 5 + self._hud_h
        self._pygame = None
        self._window = None
        self._clock = None
        self._fonts: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _carrying(self) -> int:
        """0=nobody, 1=P1, 2=P2."""
        if self.p1_status == 1:
            return 1
        if self.p2_status == 1:
            return 2
        return 0

    def _get_obs(self) -> np.ndarray:
        mr, mc = float(self.max_row), float(self.max_col)

        def _passenger_obs(src: int, dst: int, status: int):
            # status 0 (waiting): pokaż src + dst
            # status 1 (in taxi): src = 0 (nieistotne), pokaż dst
            # status 2 (delivered): wszystko = 0
            if status == 2:
                return 1.0, 0.0, 0.0, 0.0, 0.0
            sr, sc = LOCS[src]
            dr, dc = LOCS[dst]
            src_r = sr / mr if status == 0 else 0.0
            src_c = sc / mc if status == 0 else 0.0
            return status / 2.0, src_r, src_c, dr / mr, dc / mc

        p1 = _passenger_obs(self.p1_src, self.p1_dst, self.p1_status)
        p2 = _passenger_obs(self.p2_src, self.p2_dst, self.p2_status)
        return np.array(
            [self.taxi_row / mr, self.taxi_col / mc, *p1, *p2],
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

        taxi_loc = (r, c)
        carrying = self._carrying()

        # pickup: tylko gdy puste taxi i przy czekającym pasażerze
        if carrying == 0:
            if self.p1_status == 0 and taxi_loc == LOCS[self.p1_src]:
                mask[4] = 1
            if self.p2_status == 0 and taxi_loc == LOCS[self.p2_src]:
                mask[4] = 1

        # dropoff: tylko gdy wieziony pasażer jest przy swoim celu
        if carrying == 1 and taxi_loc == LOCS[self.p1_dst]:
            mask[5] = 1
        if carrying == 2 and taxi_loc == LOCS[self.p2_dst]:
            mask[5] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        return self._action_mask().astype(bool)

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        perm = self.np_random.permutation(4).tolist()
        self.p1_src, self.p1_dst, self.p2_src, self.p2_dst = perm

        self.taxi_row = int(self.np_random.integers(0, 5))
        self.taxi_col = int(self.np_random.integers(0, 5))
        self.p1_status = 0
        self.p2_status = 0
        self.last_action = None
        self.steps = 0
        self.episode_reward = 0.0
        self._deliveries = 0

        return self._get_obs(), {"prob": 1.0, "action_mask": self._action_mask()}

    def step(self, action: int):
        r, c = self.taxi_row, self.taxi_col
        new_row, new_col = r, c
        reward = -1.0
        terminated = False

        if action == 0:
            new_row = min(r + 1, self.max_row)
        elif action == 1:
            new_row = max(r - 1, 0)
        elif action == 2:
            if self.desc[1 + r, 2 * c + 2] == b":":
                new_col = min(c + 1, self.max_col)
        elif action == 3:
            if self.desc[1 + r, 2 * c] == b":":
                new_col = max(c - 1, 0)
        elif action == 4:  # pickup
            taxi_loc = (r, c)
            carrying = self._carrying()
            if carrying == 0:
                if self.p1_status == 0 and taxi_loc == LOCS[self.p1_src]:
                    self.p1_status = 1
                    reward = 5.0
                elif self.p2_status == 0 and taxi_loc == LOCS[self.p2_src]:
                    self.p2_status = 1
                    reward = 5.0
                else:
                    reward = -10.0
            else:
                reward = -10.0
        elif action == 5:  # dropoff
            taxi_loc = (r, c)
            carrying = self._carrying()
            if carrying == 1 and taxi_loc == LOCS[self.p1_dst]:
                self.p1_status = 2
                self._deliveries += 1
                reward = 30.0 if self._deliveries == 2 else 20.0
            elif carrying == 2 and taxi_loc == LOCS[self.p2_dst]:
                self.p2_status = 2
                self._deliveries += 1
                reward = 30.0 if self._deliveries == 2 else 20.0
            else:
                reward = -10.0

        self.taxi_row, self.taxi_col = new_row, new_col
        self.last_action = action
        self.steps += 1
        self.episode_reward += reward
        terminated = self.p1_status == 2 and self.p2_status == 2

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {"prob": 1.0, "action_mask": self._action_mask()},
        )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

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
        out = [[ch.decode("utf-8") for ch in line] for line in desc]
        outfile = StringIO()
        r, c = self.taxi_row, self.taxi_col
        carrying = self._carrying()

        color_map = {0: "yellow", 1: "blue", 2: "cyan"}
        out[1 + r][2 * c + 1] = utils.colorize(
            out[1 + r][2 * c + 1], color_map[carrying], highlight=True
        )

        if self.p1_status == 0:
            pr, pc = LOCS[self.p1_src]
            out[1 + pr][2 * pc + 1] = utils.colorize(
                out[1 + pr][2 * pc + 1], "blue", bold=True
            )
        if self.p2_status == 0:
            pr, pc = LOCS[self.p2_src]
            out[1 + pr][2 * pc + 1] = utils.colorize(
                out[1 + pr][2 * pc + 1], "cyan", bold=True
            )

        dr, dc = LOCS[self.p1_dst]
        out[1 + dr][2 * dc + 1] = utils.colorize(out[1 + dr][2 * dc + 1], "magenta")
        dr, dc = LOCS[self.p2_dst]
        out[1 + dr][2 * dc + 1] = utils.colorize(out[1 + dr][2 * dc + 1], "magenta")

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        status_names = ["czeka", "w taxi", "dostarczony"]
        outfile.write(
            f"P1({LOC_NAMES[self.p1_src]}→{LOC_NAMES[self.p1_dst]}): {status_names[self.p1_status]} | "
            f"P2({LOC_NAMES[self.p2_src]}→{LOC_NAMES[self.p2_dst]}): {status_names[self.p2_status]}\n"
        )
        if self.last_action is not None:
            names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            outfile.write(f"  ({names[self.last_action]})\n")

        with closing(outfile):
            return outfile.getvalue()

    def _render_pygame(self):
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("pip install pygame") from exc

        if self._window is None:
            self._pygame = pygame
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CustomGridTaxi2P-v0")
                self._window = pygame.display.set_mode((self._win_w, self._win_h))
            else:
                self._window = pygame.Surface((self._win_w, self._win_h))
            self._clock = pygame.time.Clock()
            self._fonts["s"] = pygame.font.SysFont("Menlo,Monaco,monospace", 15)
            self._fonts["m"] = pygame.font.SysFont("Menlo,Monaco,monospace", 19, bold=True)
            self._fonts["b"] = pygame.font.SysFont("Menlo,Monaco,monospace", 25, bold=True)

        if self.render_mode == "human":
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self.close()
                    return None

        pygame = self._pygame
        canvas = self._window
        cs = self._cell

        canvas.fill((250, 250, 252))

        grid_bg = (245, 245, 248)
        grid_line = (210, 210, 218)
        for row in range(5):
            for col in range(5):
                pygame.draw.rect(canvas, grid_bg, (col * cs, row * cs, cs, cs))
                pygame.draw.rect(canvas, grid_line, (col * cs, row * cs, cs, cs), 1)

        wall_col = (35, 35, 45)
        wall_w = 5
        pygame.draw.rect(canvas, wall_col, (0, 0, self._win_w, cs * 5), wall_w)
        for row in range(5):
            for col in range(4):
                if self.desc[1 + row, 2 * col + 2] == b"|":
                    x = (col + 1) * cs
                    pygame.draw.line(canvas, wall_col, (x, row * cs), (x, (row + 1) * cs), wall_w)

        # Destination highlights (P1 = blue frame, P2 = orange frame)
        for p_i, (p_dst, p_status, dst_col) in enumerate(
            [
                (self.p1_dst, self.p1_status, P_DST_COLORS[0]),
                (self.p2_dst, self.p2_status, P_DST_COLORS[1]),
            ]
        ):
            if p_status < 2:
                dr, dc = LOCS[p_dst]
                frame = pygame.Rect(dc * cs + 5, dr * cs + 5, cs - 10, cs - 10)
                pygame.draw.rect(canvas, dst_col, frame, 4, border_radius=8)
                lbl = self._fonts["s"].render(f"P{p_i + 1}", True, dst_col)
                canvas.blit(lbl, (dc * cs + cs - lbl.get_width() - 6, dr * cs + cs - lbl.get_height() - 4))

        # Loc name labels (always visible)
        for i, (lr, lc) in enumerate(LOCS):
            lbl = self._fonts["b"].render(LOC_NAMES[i], True, (160, 160, 170))
            canvas.blit(lbl, (lc * cs + 6, lr * cs + 4))

        # Waiting passengers (circles)
        for p_i, (p_status, p_src, p_color) in enumerate(
            [
                (self.p1_status, self.p1_src, P_COLORS[0]),
                (self.p2_status, self.p2_src, P_COLORS[1]),
            ]
        ):
            if p_status == 0:
                pr, pc = LOCS[p_src]
                cx = pc * cs + cs // 2
                cy = pr * cs + cs // 2 + cs // 8
                pygame.draw.circle(canvas, p_color, (cx, cy + 6), cs // 8)
                pygame.draw.circle(canvas, (245, 220, 195), (cx, cy - 6), cs // 14)
                pygame.draw.circle(canvas, p_color, (cx, cy - 6), cs // 14, 2)
                num = self._fonts["s"].render(str(p_i + 1), True, (255, 255, 255))
                canvas.blit(num, num.get_rect(center=(cx + cs // 5, cy + 6)))

        # Taxi
        tr, tc = self.taxi_row, self.taxi_col
        carrying = self._carrying()
        pad = cs // 6
        taxi_rect = pygame.Rect(tc * cs + pad, tr * cs + pad, cs - 2 * pad, cs - 2 * pad)
        body_colors = {0: (250, 210, 50), 1: P_COLORS[0], 2: P_COLORS[1]}
        pygame.draw.rect(canvas, body_colors[carrying], taxi_rect, border_radius=12)
        pygame.draw.rect(canvas, (30, 30, 35), taxi_rect, 2, border_radius=12)
        win_r = pygame.Rect(taxi_rect.x + 6, taxi_rect.y + 6, taxi_rect.w - 12, taxi_rect.h // 2 - 4)
        pygame.draw.rect(canvas, (200, 230, 250), win_r, border_radius=6)
        pygame.draw.rect(canvas, (30, 30, 35), win_r, 2, border_radius=6)
        label_str = f"P{carrying}" if carrying > 0 else "TAXI"
        t_lbl = self._fonts["m"].render(label_str, True, (255, 255, 255) if carrying else (30, 30, 35))
        canvas.blit(t_lbl, t_lbl.get_rect(center=(taxi_rect.centerx, taxi_rect.centery + taxi_rect.h // 4)))

        # HUD
        hud = pygame.Rect(0, cs * 5, self._win_w, self._hud_h)
        pygame.draw.rect(canvas, (25, 25, 33), hud)
        pygame.draw.line(canvas, wall_col, (0, cs * 5), (self._win_w, cs * 5), wall_w)

        status_names = ["czeka", "w taxi", "✓"]
        p1_str = f"P1 {LOC_NAMES[self.p1_src]}→{LOC_NAMES[self.p1_dst]}: {status_names[self.p1_status]}"
        p2_str = f"P2 {LOC_NAMES[self.p2_src]}→{LOC_NAMES[self.p2_dst]}: {status_names[self.p2_status]}"
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        last = action_names[self.last_action] if self.last_action is not None else "—"
        hud_lines = [
            (p1_str, P_DST_COLORS[0]),
            (p2_str, P_DST_COLORS[1]),
            (f"krok: {self.steps}    nagroda: {self.episode_reward:.0f}    akcja: {last}", (210, 210, 220)),
        ]
        for i, (text, color) in enumerate(hud_lines):
            surf = self._fonts["s"].render(text, True, color)
            canvas.blit(surf, (12, cs * 5 + 10 + i * 24))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None

        rgb = pygame.surfarray.pixels3d(canvas)
        return np.transpose(np.array(rgb), axes=(1, 0, 2))


def register_custom_grid_taxi2p() -> None:
    if ENV_ID_2P in gym.registry:
        return
    gym.register(
        id=ENV_ID_2P,
        entry_point="custom_grid_taxi2p_env:CustomGridTaxi2PEnv",
        max_episode_steps=400,
    )


register_custom_grid_taxi2p()
