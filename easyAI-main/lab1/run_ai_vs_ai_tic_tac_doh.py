import argparse
import os
import random
import sys
from dataclasses import dataclass

# Ensure local easyAI package is importable when this script is run from the lab1 folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from easyAI import AI_Player, Negamax
from lab1.probabilistic_tic_tac_doh import TicTacToe


@dataclass
class MatchResult:
    games: int = 0
    ai1_wins: int = 0
    ai2_wins: int = 0
    draws: int = 0
    ai1_starts: int = 0
    ai2_starts: int = 0


def winner_index(game: TicTacToe) -> int:
    """Return winner index (1 or 2), or 0 for draw."""
    if game.lose():
        return game.opponent_index
    return 0


def play_series(
    games: int,
    depth_ai1: int,
    depth_ai2: int,
    miss_chance: float,
    seed: int | None,
) -> MatchResult:
    if seed is not None:
        random.seed(seed)

    result = MatchResult()

    for i in range(games):
        starter = 1 if i % 2 == 0 else 2

        ai1 = AI_Player(Negamax(depth_ai1))
        ai2 = AI_Player(Negamax(depth_ai2))
        game = TicTacToe([ai1, ai2], miss_chance=miss_chance)
        game.current_player = starter

        if starter == 1:
            result.ai1_starts += 1
        else:
            result.ai2_starts += 1

        game.play(verbose=False)
        winner = winner_index(game)

        result.games += 1
        if winner == 1:
            result.ai1_wins += 1
        elif winner == 2:
            result.ai2_wins += 1
        else:
            result.draws += 1

    return result


def print_summary(title: str, depth_ai1: int, depth_ai2: int, result: MatchResult) -> None:
    ai1_rate = 100.0 * result.ai1_wins / result.games if result.games else 0.0
    ai2_rate = 100.0 * result.ai2_wins / result.games if result.games else 0.0
    draw_rate = 100.0 * result.draws / result.games if result.games else 0.0

    print(f"\n=== {title} ===")
    print(f"AI1 depth={depth_ai1}, AI2 depth={depth_ai2}")
    print(f"Games: {result.games}")
    print(f"Starts -> AI1: {result.ai1_starts}, AI2: {result.ai2_starts}")
    print(f"Wins   -> AI1: {result.ai1_wins} ({ai1_rate:.1f}%)")
    print(f"Wins   -> AI2: {result.ai2_wins} ({ai2_rate:.1f}%)")
    print(f"Draws  -> {result.draws} ({draw_rate:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated AI vs AI matches for TicTacDoh (probabilistic TicTacToe) "
            "with alternating starting player and result counting."
        )
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games per variant.")
    parser.add_argument("--depth-ai1", type=int, default=3, help="Negamax depth for AI1.")
    parser.add_argument("--depth-ai2", type=int, default=5, help="Negamax depth for AI2.")
    parser.add_argument(
        "--prob-miss-chance",
        type=float,
        default=0.2,
        help="Miss chance for probabilistic variant (default 0.2).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Run only deterministic variant (miss chance = 0.0).",
    )
    parser.add_argument(
        "--probabilistic-only",
        action="store_true",
        help="Run only probabilistic variant (miss chance = --prob-miss-chance).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.depth_ai1 <= 0 or args.depth_ai2 <= 0:
        raise ValueError("Both depths must be > 0")
    if not 0.0 <= args.prob_miss_chance <= 1.0:
        raise ValueError("--prob-miss-chance must be in [0.0, 1.0]")
    if args.deterministic_only and args.probabilistic_only:
        raise ValueError("Use only one of --deterministic-only or --probabilistic-only")

    run_deterministic = not args.probabilistic_only
    run_probabilistic = not args.deterministic_only

    if run_deterministic:
        deterministic = play_series(
            games=args.games,
            depth_ai1=args.depth_ai1,
            depth_ai2=args.depth_ai2,
            miss_chance=0.0,
            seed=args.seed,
        )
        print_summary("Deterministic TicTacToe", args.depth_ai1, args.depth_ai2, deterministic)

    if run_probabilistic:
        probabilistic = play_series(
            games=args.games,
            depth_ai1=args.depth_ai1,
            depth_ai2=args.depth_ai2,
            miss_chance=args.prob_miss_chance,
            seed=args.seed,
        )
        print_summary(
            f"Probabilistic TicTacDoh (miss chance={args.prob_miss_chance})",
            args.depth_ai1,
            args.depth_ai2,
            probabilistic,
        )


if __name__ == "__main__":
    main()
