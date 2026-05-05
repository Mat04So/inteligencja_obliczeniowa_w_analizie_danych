"""
Lab 04 — własne środowisko Gymnasium + demonstracja / ewaluacja agenta.

Uruchomienie:
  python main.py random              # losowa polityka na CustomGridTaxi-v0
  python main.py eval [--model PATH] # agent MaskablePPO (wymaga wytrenowanego modelu)
Trening (ok. 99% skuteczności przy ~300k kroków na CPU):
  python train.py [--timesteps 300000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ACTIONS = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}


def make_env(render_mode: str | None) -> gym.Env:
    register_custom_grid_taxi()
    return gym.make(ENV_ID, render_mode=render_mode)


def run_random_episode(seed: int = 42, max_steps: int = 200) -> None:
    env = make_env("ansi")
    obs, info = env.reset(seed=seed)

    print("== Losowa polityka (własne środowisko CustomGridTaxi-v0) ==")
    print(f"Obserwacja: {obs}")
    print(f"action_mask: {info.get('action_mask')}")
    out0 = env.render()
    if out0:
        print(out0)

    total_reward = 0.0

    for step in range(1, max_steps + 1):
        mask = info.get("action_mask")
        if mask is not None:
            action = int(env.action_space.sample(mask=np.asarray(mask, dtype=np.int8)))
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {step}")
        print(f"Action: {action} ({ACTIONS[action]})")
        print(f"Reward: {reward}")
        print(f"obserwacja: {obs}")
        out = env.render()
        if out:
            print(out)

        if terminated or truncated:
            print("Koniec epizodu.")
            break

    print(f"Suma nagród: {total_reward}")
    env.close()


def run_trained_episode(
    model_path: Path,
    seed: int = 42,
    max_steps: int = 200,
    deterministic: bool = True,
) -> None:
    if not model_path.is_file():
        raise SystemExit(
            f"Brak modelu: {model_path}\n"
            "Najpierw uruchom: python train.py"
        )

    env = ActionMasker(
        make_env("ansi"), lambda e: e.unwrapped.action_masks()
    )
    model = MaskablePPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    print("== Agent MaskablePPO ==")
    print(f"Model: {model_path}")
    print(f"seed={seed}")
    out = env.render()
    if out:
        print(out)

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=deterministic
        )
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        print(f"\nStep {step} | action={int(action)} ({ACTIONS[int(action)]}) | r={reward}")
        out = env.render()
        if out:
            print(out)
        if terminated or truncated:
            print("Koniec epizodu.")
            break

    print(f"Suma nagród: {total_reward}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 04 — CustomGridTaxi")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rand = sub.add_parser("random", help="Jeden epizod z losowymi akcjami (maskowane)")
    p_rand.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Epizod z wytrenowanym MaskablePPO")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "best_model.zip",
    )

    args = parser.parse_args()
    if args.cmd == "random":
        run_random_episode(seed=args.seed)
    elif args.cmd == "eval":
        run_trained_episode(model_path=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
