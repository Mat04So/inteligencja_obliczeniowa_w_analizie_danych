"""
Własne środowisko Gymnasium: taxi na siatce 10×10 z przeszkodami.
Cztery przystanki R, G, Y, B są losowane na wolnych polach mapy przy resecie.

Zaimplementowane od zera - nie jest to podklasa gymnasium.envs.toy_text.taxi.TaxiEnv.

Opcjonalny **bonus za zbliżanie się** (``distance_bonus_scale > 0``): przy ruchu N/S/E/W
dodawana jest ``scale * (d_stare - d_nowe)`` względem najkrótszej ścieżki BFS
do aktualnego celu (pozycja odbioru albo cel dostawy), co przyspiesza uczenie.
Poprawny pickup/dropoff daje dodatnią nagrodę, a dropoff jest maskowany tylko
przy właściwym celu pasażera.
"""

from __future__ import annotations

from collections import deque
from contextlib import closing
from io import StringIO
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces, utils

ENV_ID = "CustomGridTaxi-v0"

DEFAULT_GRID_SIZE = 10
DEFAULT_MAX_EPISODE_STEPS = 250
DEFAULT_BLOCKED_CELLS_10X10 = frozenset(
    {
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (1, 6),
        (2, 6),
        (4, 6),
        (5, 6),
        (6, 6),
        (8, 6),
        (5, 1),
        (5, 2),
        (5, 7),
        (5, 8),
    }
)


def _default_blocked_cells(n: int) -> frozenset[tuple[int, int]]:
    """Stała mapa przeszkód dla zadania treningowego 10×10."""
    if n == 10:
        return DEFAULT_BLOCKED_CELLS_10X10
    return frozenset()


def _build_open_grid_map(
    n: int,
    blocked_cells: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    draw_corner_locs: bool = True,
) -> tuple[list[bytes], list[tuple[int, int]]]:
    """Mapa ASCII jak w toy_text taxi: otwarte przejścia, przeszkody jako zablokowane pola."""
    if n < 2:
        raise ValueError("grid_size musi być >= 2 (potrzebne są 4 rozróżnialne rogi).")
    blocked = blocked_cells or frozenset()
    locs = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    letters = ["R", "G", "Y", "B"]
    cells = [[" " for _ in range(n)] for _ in range(n)]
    for r, c in blocked:
        if (r, c) in locs:
            raise ValueError("Przeszkoda nie może blokować przystanku w rogu mapy.")
        cells[r][c] = "#"
    if draw_corner_locs:
        for (r, c), ch in zip(locs, letters, strict=True):
            cells[r][c] = ch

    topbot = ("+" + "-" * (2 * n - 1) + "+").encode("ascii")
    lines: list[bytes] = [topbot]
    for r in range(n):
        inner = "".join(cells[r][c] + (":" if c < n - 1 else "") for c in range(n))
        lines.append(("|" + inner + "|").encode("ascii"))
    lines.append(topbot)
    return lines, locs


class CustomGridTaxiEnv(gym.Env):
    """
    Taksówka zbiera pasażera w jednym z czterech punktów (R,G,Y,B) i dowozi do wylosowanego celu.

    Domyślnie siatka ``grid_size``×``grid_size`` (10×10), przystanki losowane na wolnych polach.
    Obserwacja domyślna: bogatszy wektor znormalizowany (``observation_mode="vector"``).
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        observation_mode: Literal["discrete", "vector"] = "vector",
        grid_size: int = DEFAULT_GRID_SIZE,
        distance_bonus_scale: float = 2.0,
        randomize_locations: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.grid_size = int(grid_size)
        self.distance_bonus_scale = float(distance_bonus_scale)
        self.randomize_locations = bool(randomize_locations)
        self.blocked_cells = set(_default_blocked_cells(self.grid_size))
        map_lines, self.locs = _build_open_grid_map(
            self.grid_size,
            self.blocked_cells,
            draw_corner_locs=not self.randomize_locations,
        )
        self.desc = np.asarray(map_lines, dtype="c")
        self.max_row = self.grid_size - 1
        self.max_col = self.grid_size - 1
        self._n_states_taxi = self.grid_size * self.grid_size
        self._discrete_n = self._n_states_taxi * 5 * 4
        self._open_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.blocked_cells
        ]
        self._distance_maps = {
            cell: self._bfs_distances(cell)
            for cell in self._open_cells
        }
        self._max_bfs_distance = max(
            dist
            for dist_map in self._distance_maps.values()
            for dist in dist_map.values()
        )
        self.last_action: int | None = None
        self.steps = 0
        self.episode_reward = 0.0

        self.action_space = spaces.Discrete(6)
        if observation_mode == "discrete":
            self.observation_space = spaces.Discrete(self._discrete_n)
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(10,), dtype=np.float32
            )

        self.taxi_row = 0
        self.taxi_col = 0
        self.passenger_idx = 0
        self.passenger_start_idx = 0
        self.destination_idx = 0

        self._cell = max(20, min(48, 700 // self.grid_size))
        self._hud_h = 96
        gs = self.grid_size
        self._win_w = self._cell * gs
        self._win_h = self._cell * gs + self._hud_h
        self._pygame = None
        self._window = None
        self._clock = None
        self._fonts: dict[str, Any] = {}

    # --- logika nagród / przejść (sucha wersja bez tabeli P) ---

    def _is_open_cell(self, row: int, col: int) -> bool:
        return (
            0 <= row <= self.max_row
            and 0 <= col <= self.max_col
            and (row, col) not in self.blocked_cells
        )

    def _neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        candidates = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1),
        ]
        return [(r, c) for r, c in candidates if self._is_open_cell(r, c)]

    def _bfs_distances(self, target: tuple[int, int]) -> dict[tuple[int, int], int]:
        distances = {target: 0}
        queue: deque[tuple[int, int]] = deque([target])
        while queue:
            row, col = queue.popleft()
            for nr, nc in self._neighbors(row, col):
                if (nr, nc) not in distances:
                    distances[(nr, nc)] = distances[(row, col)] + 1
                    queue.append((nr, nc))
        return distances

    def _shortest_distance(self, start: tuple[int, int], target: tuple[int, int]) -> int:
        return self._distance_maps[target][start]

    def _sample_locations(self) -> list[tuple[int, int]]:
        """Losuje cztery różne przystanki z wolnych pól mapy."""
        if not self.randomize_locations:
            return list(self.locs)
        indices = self.np_random.choice(len(self._open_cells), size=4, replace=False)
        return [self._open_cells[int(i)] for i in indices]

    def _pickup(
        self, taxi_loc: tuple[int, int], pass_idx: int, reward: float
    ) -> tuple[int, float]:
        if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
            return 4, 10.0
        return pass_idx, -10.0

    def _dropoff(
        self,
        taxi_loc: tuple[int, int],
        pass_idx: int,
        dest_idx: int,
        default_reward: float,
    ) -> tuple[int, float, bool]:
        if taxi_loc == self.locs[dest_idx] and pass_idx == 4:
            return dest_idx, 100.0, True
        return pass_idx, -10.0, False

    def _subgoal_rc(self, pass_idx: int, dest_idx: int) -> tuple[int, int]:
        """Cel nawigacji: pasażer na przystanku (odbiór) albo miejsce dostawy."""
        if pass_idx < 4:
            return self.locs[pass_idx]
        return self.locs[dest_idx]

    def _state_index(self) -> int:
        return int(
            (
                (self.taxi_row * self.grid_size + self.taxi_col) * 5
                + self.passenger_idx
            )
            * 4
            + self.destination_idx
        )

    def _get_obs(self) -> int | np.ndarray:
        if self.observation_mode == "discrete":
            return self._state_index()
        target_r, target_c = self._subgoal_rc(self.passenger_idx, self.destination_idx)
        src_r, src_c = self.locs[self.passenger_start_idx]
        dst_r, dst_c = self.locs[self.destination_idx]
        passenger_state = 0.5 if self.passenger_idx == 4 else 0.0
        bfs_distance = self._shortest_distance(
            (self.taxi_row, self.taxi_col),
            (target_r, target_c),
        )
        return np.array(
            [
                self.taxi_row / self.max_row,
                self.taxi_col / self.max_col,
                passenger_state,
                src_r / self.max_row,
                src_c / self.max_col,
                dst_r / self.max_row,
                dst_c / self.max_col,
                target_r / self.max_row,
                target_c / self.max_col,
                bfs_distance / self._max_bfs_distance,
            ],
            dtype=np.float32,
        )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(6, dtype=np.int8)
        r, c = self.taxi_row, self.taxi_col
        if self._is_open_cell(r + 1, c):
            mask[0] = 1
        if self._is_open_cell(r - 1, c):
            mask[1] = 1
        if self._is_open_cell(r, c + 1):
            mask[2] = 1
        if self._is_open_cell(r, c - 1):
            mask[3] = 1
        if self.passenger_idx < 4 and (r, c) == self.locs[self.passenger_idx]:
            mask[4] = 1
        if self.passenger_idx == 4 and (r, c) == self.locs[self.destination_idx]:
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
        self.locs = self._sample_locations()
        self.passenger_idx = int(self.np_random.integers(0, 4))
        self.passenger_start_idx = self.passenger_idx
        dest_choices = [d for d in range(4) if d != self.passenger_idx]
        self.destination_idx = int(self.np_random.choice(dest_choices))
        taxi_cell = self._open_cells[int(self.np_random.integers(0, len(self._open_cells)))]
        self.taxi_row, self.taxi_col = taxi_cell
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
        reward = -0.2
        terminated = False
        old_target = self._subgoal_rc(p_idx, d_idx)
        d_old = self._shortest_distance((row, col), old_target)

        if action == 0:
            if self._is_open_cell(row + 1, col):
                new_row = row + 1
        elif action == 1:
            if self._is_open_cell(row - 1, col):
                new_row = row - 1
        elif action == 2:
            if self._is_open_cell(row, col + 1):
                new_col = col + 1
        elif action == 3:
            if self._is_open_cell(row, col - 1):
                new_col = col - 1
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

        # Bonus za skrócenie najkrótszej ścieżki BFS do aktualnego celu.
        if self.distance_bonus_scale > 0.0 and action in (0, 1, 2, 3):
            d_new = self._shortest_distance((new_row, new_col), old_target)
            reward += self.distance_bonus_scale * float(d_old - d_new)

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
        loc_names = ["R", "G", "Y", "B"]
        for loc_i, (lr, lc) in enumerate(self.locs):
            out[1 + lr][2 * lc + 1] = utils.colorize(
                loc_names[loc_i], "cyan", bold=True
            )

        if pass_idx < 4:
            out[1 + r][2 * c + 1] = utils.colorize(
                out[1 + r][2 * c + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:
            out[1 + r][2 * c + 1] = utils.colorize(
                ul(out[1 + r][2 * c + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
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

        canvas.fill((232, 238, 247))

        grid_bg = (247, 249, 252)
        grid_alt = (240, 244, 249)
        grid_line = (211, 219, 231)
        wall_color = (45, 53, 67)
        wall_edge = (25, 30, 40)
        wall_w = 5
        gs = self.grid_size
        for r in range(gs):
            for c in range(gs):
                rect = pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4)
                if (r, c) in self.blocked_cells:
                    pygame.draw.rect(canvas, wall_color, rect, border_radius=8)
                    pygame.draw.rect(canvas, wall_edge, rect, 2, border_radius=8)
                    brick_y = rect.y + rect.h // 2
                    pygame.draw.line(
                        canvas,
                        (66, 76, 93),
                        (rect.x + 6, brick_y),
                        (rect.right - 6, brick_y),
                        2,
                    )
                else:
                    bg = grid_alt if (r + c) % 2 else grid_bg
                    pygame.draw.rect(canvas, bg, rect, border_radius=7)
                    pygame.draw.rect(canvas, grid_line, rect, 1, border_radius=7)

        pygame.draw.rect(canvas, wall_color, (0, 0, self._win_w, cs * gs), wall_w)
        for r in range(gs):
            for c in range(gs - 1):
                if self.desc[1 + r, 2 * c + 2] == b"|":
                    x = (c + 1) * cs
                    pygame.draw.line(
                        canvas, wall_color, (x, r * cs), (x, (r + 1) * cs), wall_w
                    )

        for i, (lr, lc) in enumerate(self.locs):
            color = loc_colors[i]
            cx = lc * cs + cs // 2
            cy = lr * cs + cs // 2
            pad = max(5, cs // 7)
            badge = pygame.Rect(lc * cs + pad, lr * cs + pad, cs - 2 * pad, cs - 2 * pad)
            pygame.draw.rect(canvas, (255, 255, 255), badge, border_radius=12)
            pygame.draw.rect(canvas, color, badge, 4, border_radius=12)
            label = self._fonts["big"].render(loc_names[i], True, color)
            canvas.blit(label, label.get_rect(center=badge.center))

        dr, dc = self.locs[self.destination_idx]
        goal_rect = pygame.Rect(dc * cs + 5, dr * cs + 5, cs - 10, cs - 10)
        pygame.draw.rect(canvas, (232, 209, 255), goal_rect, border_radius=14)
        pygame.draw.rect(canvas, (152, 70, 215), goal_rect, 4, border_radius=14)
        flag = self._fonts["small"].render("CEL", True, (92, 36, 150))
        canvas.blit(flag, flag.get_rect(center=(goal_rect.centerx, goal_rect.bottom - 12)))

        if self.passenger_idx < 4:
            pr, pc = self.locs[self.passenger_idx]
            cx = pc * cs + cs // 2
            cy = pr * cs + cs // 2 + 8
            shadow = pygame.Rect(cx - cs // 5, cy + cs // 7, (2 * cs) // 5, cs // 8)
            pygame.draw.ellipse(canvas, (170, 178, 188), shadow)
            pygame.draw.circle(canvas, (36, 48, 64), (cx, cy + 5), cs // 8)
            pygame.draw.circle(canvas, (247, 217, 190), (cx, cy - 7), cs // 11)
            pygame.draw.circle(canvas, (36, 48, 64), (cx, cy - 7), cs // 11, 2)

        tr, tc = self.taxi_row, self.taxi_col
        pad = cs // 6
        taxi_rect = pygame.Rect(tc * cs + pad, tr * cs + pad, cs - 2 * pad, cs - 2 * pad)
        on_board = self.passenger_idx == 4
        body = (60, 200, 110) if on_board else (250, 200, 50)
        shadow = taxi_rect.move(3, 4)
        pygame.draw.rect(canvas, (150, 158, 170), shadow, border_radius=14)
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
        wheel_r = max(3, cs // 12)
        for wx in (taxi_rect.left + taxi_rect.w // 4, taxi_rect.right - taxi_rect.w // 4):
            pygame.draw.circle(canvas, (25, 28, 35), (wx, taxi_rect.bottom - 3), wheel_r)
        pygame.draw.circle(canvas, (255, 246, 160), (taxi_rect.left + 6, taxi_rect.centery), 3)
        pygame.draw.circle(canvas, (255, 246, 160), (taxi_rect.right - 6, taxi_rect.centery), 3)
        t_label = self._fonts["mid"].render("TAXI", True, (30, 30, 35))
        canvas.blit(
            t_label,
            t_label.get_rect(
                center=(taxi_rect.centerx, taxi_rect.centery + taxi_rect.h // 4)
            ),
        )

        hud = pygame.Rect(0, cs * gs, self._win_w, self._hud_h)
        pygame.draw.rect(canvas, (28, 34, 48), hud)
        pygame.draw.line(canvas, wall_color, (0, cs * gs), (self._win_w, cs * gs), wall_w)

        pass_state = (
            "pasażer: w taksówce"
            if on_board
            else f"pasażer: {loc_names[self.passenger_idx]}"
        )
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        last = action_names[self.last_action] if self.last_action is not None else "-"
        lines = [
            f"{pass_state}    cel: {loc_names[self.destination_idx]}    mapa: {gs}x{gs}",
            f"krok: {self.steps}    nagroda: {self.episode_reward:.1f}",
            f"ostatnia akcja: {last}",
        ]
        for i, text in enumerate(lines):
            surf = self._fonts["small"].render(text, True, (238, 242, 248))
            canvas.blit(surf, (12, cs * gs + 8 + i * 22))

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
        kwargs={"grid_size": DEFAULT_GRID_SIZE},
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
    )


register_custom_grid_taxi()
