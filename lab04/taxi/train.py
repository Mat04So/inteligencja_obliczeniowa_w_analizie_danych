"""
Trenowanie agenta MaskablePPO na CustomGridTaxi-v0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


def make_masked_env(render_mode: str | None = None) -> gym.Env:
    register_custom_grid_taxi()
    env = gym.make(ENV_ID, render_mode=render_mode)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


def evaluate(model: MaskablePPO, n_episodes: int = 100, seed: int = 0) -> float:
    """Odsetek epizodów zakończonych dostawą (+20 przy terminated)."""
    env = make_masked_env()
    successes = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        success = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            if terminated and reward >= 20:
                success = True
            done = terminated or truncated
        if success:
            successes += 1
    env.close()
    return successes / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening MaskablePPO na CustomGridTaxi")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    args = parser.parse_args()

    register_custom_grid_taxi()
    base_env = gym.make(ENV_ID)
    check_env(base_env, warn=True)
    base_env.close()

    train_env = Monitor(make_masked_env())
    eval_env = Monitor(make_masked_env())

    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.model_dir / "best_model.zip"
    final_path = args.model_dir / "custom_grid_taxi_ppo.zip"

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(args.model_dir),
        log_path=str(args.model_dir / "logs"),
        eval_freq=max(5000, args.timesteps // 20),
        n_eval_episodes=15,
        deterministic=True,
        render=False,
    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(final_path)
    print(f"Zapisano model: {final_path}")
    if best_path.exists():
        print(f"Najlepszy checkpoint (EvalCallback): {best_path}")

    model_to_test = MaskablePPO.load(best_path) if best_path.exists() else model
    rate = evaluate(model_to_test, n_episodes=200, seed=args.seed + 1)
    print(f"Oszacowanie skuteczności (200 ep., greedy): {100 * rate:.1f}%")


if __name__ == "__main__":
    main()
