"""
Lab 04 - własne środowisko Gymnasium + demonstracja / ewaluacja agenta.

Uruchomienie:
  python main.py random                        # 10×10 ze ścianami i losowymi R/G/Y/B
  python main.py eval [--model PATH]           # agent MaskablePPO, random locs, obs vector
  python main.py random --render ansi          # tryb tekstowy
  python main.py eval --grid-size 10           # jawny rozmiar siatki
Trening:
  python train.py

Uwaga: aplikacja oczekuje modelu wytrenowanego z losowymi R/G/Y/B i ``observation_mode="vector"``.
Po zmianie wariantu uruchom trening od nowa.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ACTIONS = {0: "south", 1: "north", 2: "east", 3: "west", 4: "pickup", 5: "dropoff"}
DEFAULT_APP_GRID_SIZE = 10
DEFAULT_MAX_EPISODE_STEPS = 250
APP_OBSERVATION_MODE = "vector"

_DEFAULT_MODEL = "models/v1_10x10_walls_random_locs_vector/best_model.zip"


def make_env(render_mode: str | None, grid_size: int = DEFAULT_APP_GRID_SIZE) -> gym.Env:
    from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi

    gs = int(grid_size)
    register_custom_grid_taxi()
    return gym.make(
        ENV_ID,
        render_mode=render_mode,
        grid_size=gs,
        observation_mode=APP_OBSERVATION_MODE,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS if gs == 10 else max(250, gs * gs * 2),
    )


def _space_signature(space: gym.Space) -> tuple[str, object]:
    if hasattr(space, "n"):
        return ("discrete", int(space.n))
    return ("box", tuple(getattr(space, "shape", ())))


def _show_frame(env: gym.Env, render: str, header: str | None = None) -> None:
    if render == "ansi":
        if header:
            print(header)
        out = env.render()
        if out:
            print(out)
    else:
        env.render()


def _hold_window(env: gym.Env, render: str, seconds: float) -> None:
    if render != "human":
        return
    end_t = time.time() + seconds
    while time.time() < end_t:
        env.render()
        time.sleep(0.05)


def run_random_episode(
    seed: int = 42,
    max_steps: int = 400,
    render: str = "human",
    fps: int = 4,
    grid_size: int = DEFAULT_APP_GRID_SIZE,
) -> None:
    env = make_env(render, grid_size)
    if render == "human":
        env.unwrapped.metadata["render_fps"] = fps
    obs, info = env.reset(seed=seed)

    print(f"== Losowa polityka ({env.unwrapped.spec.id}) ==")
    print(f"seed={seed}")
    _show_frame(env, render)

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        mask = info.get("action_mask")
        if mask is not None:
            action = int(env.action_space.sample(mask=np.asarray(mask, dtype=np.int8)))
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render == "ansi":
            print(f"\nStep {step} | action={action} ({ACTIONS[action]}) | r={reward}")
        _show_frame(env, render)

        if terminated or truncated:
            print(f"Step {step}: koniec epizodu (terminated={terminated}, truncated={truncated})")
            break

    print(f"Suma nagród: {total_reward:.1f}")
    _hold_window(env, render, seconds=2.0)
    env.close()


def run_trained_episode(
    model_path: Path | None = None,
    seed: int = 42,
    max_steps: int = 400,
    render: str = "human",
    fps: int = 4,
    grid_size: int = DEFAULT_APP_GRID_SIZE,
) -> None:
    if model_path is None:
        model_path = Path(__file__).resolve().parent / _DEFAULT_MODEL

    if not model_path.is_file():
        raise SystemExit(
            f"Brak modelu: {model_path}\n"
            "Najpierw uruchom: python train.py"
        )

    env = ActionMasker(make_env(render, grid_size), lambda e: e.unwrapped.action_masks())
    if render == "human":
        env.unwrapped.metadata["render_fps"] = fps
    model = MaskablePPO.load(model_path)
    if _space_signature(model.observation_space) != _space_signature(env.observation_space):
        env.close()
        raise SystemExit(
            "Model ma inną przestrzeń obserwacji niż aplikacja.\n"
            f"Model: {model.observation_space}, env: {env.observation_space}\n"
            "Wytrenuj model od nowa: python train.py"
        )

    obs, _ = env.reset(seed=seed)
    print(f"== Agent MaskablePPO ({env.unwrapped.spec.id}) ==")
    print(
        f"Model: {model_path}    seed={seed}    "
        f"siatka={grid_size}×{grid_size} walls random-locs    obs={APP_OBSERVATION_MODE}"
    )
    _show_frame(env, render)

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True
        )
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        if render == "ansi":
            print(
                f"\nStep {step} | action={int(action)} "
                f"({ACTIONS[int(action)]}) | r={reward}"
            )
        _show_frame(env, render)
        if terminated or truncated:
            print(f"Step {step}: koniec epizodu (terminated={terminated}, truncated={truncated})")
            break

    print(f"Suma nagród: {total_reward:.1f}")
    _hold_window(env, render, seconds=2.5)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 04 - CustomGridTaxi-v0, jeden pasażer"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in [
        ("random", "Jeden epizod z losowymi akcjami (maskowane)"),
        ("eval", "Epizod z wytrenowanym MaskablePPO"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--render", choices=["human", "ansi"], default="human",
                       help="human=okno pygame (domyślne), ansi=ASCII")
        p.add_argument("--fps", type=int, default=4)
        p.add_argument(
            "--grid-size",
            type=int,
            default=DEFAULT_APP_GRID_SIZE,
            help="Bok siatki kwadratowej (domyślnie 10; ściany aktywne dla 10×10)",
        )
        p.add_argument("--max-steps", type=int, default=400)
        if name == "eval":
            p.add_argument(
                "--model",
                type=Path,
                default=None,
                help=(
                    "Ścieżka do modelu "
                    "(domyślnie models/v1_10x10_walls_random_locs_vector/best_model.zip)"
                ),
            )

    args = parser.parse_args()
    if args.cmd == "random":
        run_random_episode(
            seed=args.seed,
            max_steps=args.max_steps,
            render=args.render,
            fps=args.fps,
            grid_size=args.grid_size,
        )
    elif args.cmd == "eval":
        run_trained_episode(
            model_path=getattr(args, "model", None),
            seed=args.seed,
            max_steps=args.max_steps,
            render=args.render,
            fps=args.fps,
            grid_size=args.grid_size,
        )


if __name__ == "__main__":
    main()
