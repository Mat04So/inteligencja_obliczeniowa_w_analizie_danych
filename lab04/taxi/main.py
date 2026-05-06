"""
Lab 04 — własne środowisko Gymnasium + demonstracja / ewaluacja agenta.

Uruchomienie:
  python main.py random                        # v1, okno pygame
  python main.py random --env v2               # 2 pasażerów
  python main.py eval [--model PATH]           # agent MaskablePPO (v1)
  python main.py eval --env v2                 # agent MaskablePPO (v2)
  python main.py random --render ansi          # tryb tekstowy
Trening:
  python train.py
  python train.py --env v2 --timesteps 600000
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

_DEFAULT_MODELS = {
    "v1": "models/v1/best_model.zip",
    "v2": "models/v2/best_model.zip",
}


def make_env(env_ver: str, render_mode: str | None) -> gym.Env:
    if env_ver == "v1":
        from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi
        register_custom_grid_taxi()
        return gym.make(ENV_ID, render_mode=render_mode)
    else:
        from custom_grid_taxi2p_env import ENV_ID_2P, register_custom_grid_taxi2p
        register_custom_grid_taxi2p()
        return gym.make(ENV_ID_2P, render_mode=render_mode)


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
    env_ver: str = "v1",
    seed: int = 42,
    max_steps: int = 400,
    render: str = "human",
    fps: int = 4,
) -> None:
    env = make_env(env_ver, render)
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
    env_ver: str = "v1",
    model_path: Path | None = None,
    seed: int = 42,
    max_steps: int = 400,
    render: str = "human",
    fps: int = 4,
) -> None:
    if model_path is None:
        model_path = Path(__file__).resolve().parent / _DEFAULT_MODELS[env_ver]

    if not model_path.is_file():
        raise SystemExit(
            f"Brak modelu: {model_path}\n"
            f"Najpierw uruchom: python train.py{'  --env ' + env_ver if env_ver != 'v1' else ''}"
        )

    env = ActionMasker(make_env(env_ver, render), lambda e: e.unwrapped.action_masks())
    if render == "human":
        env.unwrapped.metadata["render_fps"] = fps
    model = MaskablePPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    print(f"== Agent MaskablePPO ({env.unwrapped.spec.id}) ==")
    print(f"Model: {model_path}    seed={seed}")
    _show_frame(env, render)

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True
        )
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        if render == "ansi":
            print(f"\nStep {step} | action={int(action)} ({ACTIONS[int(action)]}) | r={reward}")
        _show_frame(env, render)
        if terminated or truncated:
            print(f"Step {step}: koniec epizodu (terminated={terminated}, truncated={truncated})")
            break

    print(f"Suma nagród: {total_reward:.1f}")
    _hold_window(env, render, seconds=2.5)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 04 — CustomGridTaxi")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in [
        ("random", "Jeden epizod z losowymi akcjami (maskowane)"),
        ("eval", "Epizod z wytrenowanym MaskablePPO"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--env", choices=["v1", "v2"], default="v1",
                       help="v1=1 pasażer (domyślnie), v2=2 pasażerów")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--render", choices=["human", "ansi"], default="human",
                       help="human=okno pygame (domyślne), ansi=ASCII")
        p.add_argument("--fps", type=int, default=4)
        if name == "eval":
            p.add_argument("--model", type=Path, default=None,
                           help="Ścieżka do modelu (domyślnie wg --env)")

    args = parser.parse_args()
    if args.cmd == "random":
        run_random_episode(
            env_ver=args.env, seed=args.seed, render=args.render, fps=args.fps
        )
    elif args.cmd == "eval":
        run_trained_episode(
            env_ver=args.env,
            model_path=getattr(args, "model", None),
            seed=args.seed,
            render=args.render,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
