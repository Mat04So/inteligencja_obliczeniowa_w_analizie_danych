"""
Trenowanie agenta MaskablePPO na CustomGridTaxi-v0: taxi 10×10 ze ścianami i losowymi punktami.

  python train.py                               # 10×10, ściany, losowe R/G/Y/B, obs vector
  ./run_train_macos.sh                          # macOS: bezpieczniejszy start (OpenMP przed importem)
  python train.py --timesteps 300000            # krótszy bieg testowy
  python train.py --n-envs 8                    # równoległe środowiska

Na macOS ustawiane są zmienne OPENMP / KMP przed importem PyTorch (Abort „mutex lock failed”).
Przy wielu środowiskach na Darwin używany jest DummyVecEnv (wolniejszy, bez subprocessów).

Jeśli nadal Abort zaraz po pip lub przy starcie: użyj osobnego venv (patrz niżej) oraz
``./run_train_macos.sh ...`` - zmienne są ustawiane przed procesem Pythona, nie tylko w train.py.

Instalacja w izolacji (zalecane; log pip pokazywał konflikt z manim przy wspólnym Python 3.9):
  cd lab04/taxi && python3 -m venv .venv && source .venv/bin/activate
  pip install -U pip && pip install -r requirements.txt
  ./run_train_macos.sh

Z notebooka (po załadowaniu komórki startowej): ``train.run_training_session(...)``.

Uwaga: model używa ``observation_mode="vector"``, mapy 10×10 ze ścianami i losowych punktów.
Po zmianie wariantu wytrenuj model od nowa; stare checkpointy nie są zgodne.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


TRAIN_ENV_VERSION = "v1"
TRAIN_VARIANT = "10x10_walls_random_locs_vector"
DEFAULT_TRAIN_GRID_SIZE = 10
DEFAULT_MAX_EPISODE_STEPS = 250
TRAIN_OBSERVATION_MODE = "vector"


def linear_schedule(initial_value: float):
    """Liniowo zmniejsza learning rate w trakcie treningu PPO."""
    def _schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return _schedule


def _bootstrap_training_runtime() -> None:
    """Przed importem torch/stable-baselines: stabilizacja na macOS / stacku conda."""
    if sys.platform == "darwin":
        # Konflikt runtime OpenMP (numpy/torch) → libc++ mutex abort.
        # Nadpisujemy (nie setdefault): conda często ustawia OMP_* na wyższe wartości.
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


_bootstrap_training_runtime()

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def _make_base_env(
    env_ver: str,
    render_mode: str | None = None,
    grid_size: int | None = None,
    observation_mode: str = TRAIN_OBSERVATION_MODE,
) -> gym.Env:
    if env_ver != TRAIN_ENV_VERSION:
        raise ValueError(
            "train.py obsługuje teraz tylko env_ver='v1' "
            "(CustomGridTaxi-v0, jeden pasażer)."
        )

    from custom_grid_taxi_env import ENV_ID, register_custom_grid_taxi

    gs = int(grid_size) if grid_size is not None else DEFAULT_TRAIN_GRID_SIZE
    register_custom_grid_taxi()
    max_episode_steps = DEFAULT_MAX_EPISODE_STEPS if gs == 10 else max(250, gs * gs * 2)
    return gym.make(
        ENV_ID,
        render_mode=render_mode,
        grid_size=gs,
        observation_mode=observation_mode,
        max_episode_steps=max_episode_steps,
    )


def make_masked_env(
    env_ver: str,
    render_mode: str | None = None,
    grid_size: int | None = None,
    observation_mode: str = TRAIN_OBSERVATION_MODE,
) -> gym.Env:
    env = _make_base_env(
        env_ver,
        render_mode,
        grid_size=grid_size,
        observation_mode=observation_mode,
    )
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


def evaluate(
    model: MaskablePPO,
    env_ver: str = TRAIN_ENV_VERSION,
    n_episodes: int = 100,
    seed: int = 0,
    grid_size: int | None = None,
    observation_mode: str = TRAIN_OBSERVATION_MODE,
) -> float:
    env = make_masked_env(
        env_ver,
        grid_size=grid_size,
        observation_mode=observation_mode,
    )
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


def run_training_session(
    *,
    env_ver: str = TRAIN_ENV_VERSION,
    grid_size: int | None = DEFAULT_TRAIN_GRID_SIZE,
    timesteps: int | None = None,
    n_envs: int | None = None,
    seed: int = 42,
    model_dir: Path | None = None,
    final_eval_episodes: int = 200,
    verbose_final_eval: bool = True,
) -> dict[str, Any]:
    """
    Pełny pipeline treningu - do wywołania z notebooka lub innego skryptu.
    Zwraca m.in. załadowany model i ścieżki zapisu (jak ``python train.py``).
    """
    if env_ver != TRAIN_ENV_VERSION:
        raise ValueError(
            "run_training_session obsługuje teraz tylko env_ver='v1' "
            "(jeden pasażer)."
        )

    default_steps = 1_500_000
    default_n_envs = 8
    ts = timesteps if timesteps is not None else default_steps
    ne = n_envs if n_envs is not None else default_n_envs
    md = model_dir or Path(__file__).resolve().parent / "models"
    suffix = f"_{TRAIN_VARIANT}"

    gs_disp = int(grid_size) if grid_size is not None else DEFAULT_TRAIN_GRID_SIZE
    use_dummy_vec = sys.platform == "darwin" and ne > 1
    vec_env_cls = DummyVecEnv if use_dummy_vec else None
    print(
        f"Środowisko: {env_ver}  |  siatka: {gs_disp}×{gs_disp}  |  "
        f"obs: {TRAIN_OBSERVATION_MODE}  |  "
        f"timesteps: {ts:,}  |  n_envs: {ne}  |  seed: {seed}"
    )
    if use_dummy_vec:
        print(
            "(macOS) VecEnv: DummyVecEnv - brak subprocessów; przy dużym n_envs trening "
            "jest wolniejszy, ale stabilniejszy niż SubprocVecEnv."
        )

    grid_kw: dict[str, int | str | None] = {
        "grid_size": grid_size,
        "observation_mode": TRAIN_OBSERVATION_MODE,
    }

    base_env = _make_base_env(env_ver, **grid_kw)
    check_env(base_env, warn=True)
    base_env.close()

    train_env = make_vec_env(
        lambda: make_masked_env(env_ver, **grid_kw),
        n_envs=ne,
        seed=seed,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecMonitor(train_env)
    eval_env = Monitor(make_masked_env(env_ver, **grid_kw))

    md.mkdir(parents=True, exist_ok=True)
    best_dir = md / f"{env_ver}_{TRAIN_VARIANT}"
    best_dir.mkdir(exist_ok=True)
    final_path = md / f"custom_grid_taxi_ppo{suffix}.zip"

    n_eval_episodes = 50
    eval_freq = max(2000, ts // (25 * ne))
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(md / f"logs{suffix}"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    _default_best = best_dir / "best_model.zip"

    # v1: jeden pasażer, mapa 10×10 ze ścianami, obs wektorowa + BFS shaping.
    # Kluczowe decyzje:
    #   n_steps=512  - częstsze aktualizacje, dużo epizodów na małej mapie
    #   gamma=0.99   - krótszy horyzont przy limicie 250 kroków
    #   ent_coef=0.01 - eksploracja pickup/dropoff bez nadmiernego losowania
    #   target_kl    - wczesne zatrzymanie, gdy update robi się zbyt agresywny
    ppo_kwargs = dict(
        learning_rate=linear_schedule(2.5e-4),
        n_steps=512,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    model = MaskablePPO("MlpPolicy", train_env, verbose=1, seed=seed, **ppo_kwargs)
    model.learn(total_timesteps=ts, callback=eval_callback)
    model.save(final_path)
    print(f"Zapisano model: {final_path}")

    if _default_best.exists():
        print(f"Najlepszy checkpoint (EvalCallback): {_default_best}")

    load_path = _default_best if _default_best.exists() else final_path
    model_to_test = MaskablePPO.load(load_path)
    rate = evaluate(
        model_to_test,
        env_ver,
        n_episodes=final_eval_episodes,
        seed=seed + 1,
        **grid_kw,
    )
    if verbose_final_eval:
        print(
            f"Oszacowanie skuteczności ({final_eval_episodes} ep., greedy): {100 * rate:.1f}%"
        )
        if rate < 0.9:
            print(
                "Uwaga: skuteczność < 90%. Uruchom dłuższy trening albo kontynuuj strojenie."
            )

    train_env.close()
    eval_env.close()

    return {
        "grid_size": gs_disp,
        "env_ver": env_ver,
        "final_path": final_path,
        "best_path": load_path,
        "success_rate": rate,
        "model": model_to_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trening MaskablePPO na CustomGridTaxi-v0 (jeden pasażer)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_TRAIN_GRID_SIZE,
        help="Bok siatki kwadratowej (domyślnie 10; ściany aktywne dla 10×10)",
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Domyślnie: 1.5M")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Liczba równoległych envów (domyślnie: 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    args = parser.parse_args()

    run_training_session(
        env_ver=TRAIN_ENV_VERSION,
        grid_size=args.grid_size,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
