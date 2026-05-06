"""
Trenowanie agenta MaskablePPO na CustomGridTaxi-v0 lub CustomGridTaxi2P-v0.

  python train.py                              # v1, 300k kroków
  python train.py --env v2                     # v2 (2 pasażerów), 600k kroków
  python train.py --env v2 --timesteps 800000  # dłuższy trening v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


def _make_base_env(env_ver: str, render_mode: str | None = None) -> gym.Env:
    if env_ver == "v1":
        from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi
        register_custom_grid_taxi()
        return gym.make(ENV_ID, render_mode=render_mode)
    else:
        from custom_grid_taxi2p_env import ENV_ID_2P, register_custom_grid_taxi2p
        register_custom_grid_taxi2p()
        return gym.make(ENV_ID_2P, render_mode=render_mode)


def make_masked_env(env_ver: str, render_mode: str | None = None) -> gym.Env:
    env = _make_base_env(env_ver, render_mode)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


def evaluate(model: MaskablePPO, env_ver: str, n_episodes: int = 100, seed: int = 0) -> float:
    env = make_masked_env(env_ver)
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
            if terminated:
                success = True
            done = terminated or truncated
        if success:
            successes += 1
    env.close()
    return successes / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening MaskablePPO na CustomGridTaxi")
    parser.add_argument("--env", choices=["v1", "v2"], default="v1",
                        help="v1=1 pasażer (domyślnie), v2=2 pasażerów")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Domyślnie: 300k (v1) lub 600k (v2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    args = parser.parse_args()

    default_steps = {"v1": 300_000, "v2": 1_000_000}
    timesteps = args.timesteps or default_steps[args.env]
    suffix = "" if args.env == "v1" else "_v2"

    print(f"Środowisko: {args.env}  |  timesteps: {timesteps:,}  |  seed: {args.seed}")

    base_env = _make_base_env(args.env)
    check_env(base_env, warn=True)
    base_env.close()

    train_env = Monitor(make_masked_env(args.env))
    eval_env = Monitor(make_masked_env(args.env))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    # każda wersja ma swój podkatalog → callback nigdy nie nadpisuje drugiej wersji
    best_dir = args.model_dir / args.env
    best_dir.mkdir(exist_ok=True)
    best_path = best_dir / "best_model.zip"
    final_path = args.model_dir / f"custom_grid_taxi_ppo{suffix}.zip"

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(args.model_dir / f"logs{suffix}"),
        eval_freq=max(5000, timesteps // 20),
        n_eval_episodes=15,
        deterministic=True,
        render=False,
    )
    _default_best = best_dir / "best_model.zip"

    ent_coef = 0.02 if args.env == "v2" else 0.0
    policy_kwargs = dict(net_arch=[256, 256]) if args.env == "v2" else {}
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs or None,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(final_path)
    print(f"Zapisano model: {final_path}")

    if _default_best.exists():
        print(f"Najlepszy checkpoint (EvalCallback): {_default_best}")

    load_path = _default_best if _default_best.exists() else final_path
    model_to_test = MaskablePPO.load(load_path)
    rate = evaluate(model_to_test, args.env, n_episodes=200, seed=args.seed + 1)
    print(f"Oszacowanie skuteczności (200 ep., greedy): {100 * rate:.1f}%")


if __name__ == "__main__":
    main()
