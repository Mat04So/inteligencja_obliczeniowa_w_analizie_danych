import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, field

# Ensure local easyAI package is importable when this script is run from the lab1 folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from easyAI import Negamax
from lab1.probabilistic_tic_tac_doh import TicTacToe


INF = float("infinity")


@dataclass
class PlayerTimeStats:
    total_time_s: float = 0.0
    moves: int = 0

    @property
    def avg_time_ms(self) -> float:
        if self.moves == 0:
            return 0.0
        return 1000.0 * self.total_time_s / self.moves


@dataclass
class SeriesResult:
    variant_name: str
    algorithm_name: str
    depth_ai1: int
    depth_ai2: int
    games: int = 0
    ai1_wins: int = 0
    ai2_wins: int = 0
    draws: int = 0
    ai1_starts: int = 0
    ai2_starts: int = 0
    ai1_time: PlayerTimeStats = field(default_factory=PlayerTimeStats)
    ai2_time: PlayerTimeStats = field(default_factory=PlayerTimeStats)


class TimedAIPlayer:
    def __init__(self, ai_algo):
        self.ai_algo = ai_algo
        self.total_time_s = 0.0
        self.moves = 0

    def ask_move(self, game):
        start = time.perf_counter()
        move = self.ai_algo(game)
        elapsed = time.perf_counter() - start
        self.total_time_s += elapsed
        self.moves += 1
        return move


class NegamaxNoAlphaBeta:
    """Negamax without alpha-beta pruning for mandatory comparison."""

    def __init__(self, depth, scoring=None):
        self.depth = depth
        self.scoring = scoring

    def __call__(self, game):
        scoring = self.scoring if self.scoring else (lambda g: g.scoring())
        _negamax_no_ab(game, self.depth, self.depth, scoring)
        return game.ai_move


def _negamax_no_ab(game, depth, orig_depth, scoring):
    if depth == 0 or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)

    possible_moves = game.possible_moves()
    state = game
    best_move = possible_moves[0]
    best_value = -INF
    unmake_move = hasattr(state, "unmake_move")

    if depth == orig_depth:
        state.ai_move = best_move

    for move in possible_moves:
        if not unmake_move:
            game = state.copy()

        game.make_move(move)
        game.switch_player()

        value = -_negamax_no_ab(game, depth - 1, orig_depth, scoring)

        if unmake_move:
            game.switch_player()
            game.unmake_move(move)

        if value > best_value:
            best_value = value
            best_move = move
            if depth == orig_depth:
                state.ai_move = move

    return best_value


def winner_index(game: TicTacToe) -> int:
    if game.lose():
        return game.opponent_index
    return 0


def make_algo(algorithm_name: str, depth: int):
    if algorithm_name == "negamax_ab":
        return Negamax(depth)
    if algorithm_name == "negamax_no_ab":
        return NegamaxNoAlphaBeta(depth)
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_series(
    games: int,
    algorithm_name: str,
    depth_ai1: int,
    depth_ai2: int,
    miss_chance: float,
    seed: int | None,
) -> SeriesResult:
    variant_name = "deterministic" if miss_chance == 0.0 else f"probabilistic_{miss_chance}"
    result = SeriesResult(
        variant_name=variant_name,
        algorithm_name=algorithm_name,
        depth_ai1=depth_ai1,
        depth_ai2=depth_ai2,
    )

    if seed is not None:
        random.seed(seed)

    for i in range(games):
        starter = 1 if i % 2 == 0 else 2

        ai1 = TimedAIPlayer(make_algo(algorithm_name, depth_ai1))
        ai2 = TimedAIPlayer(make_algo(algorithm_name, depth_ai2))
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

        result.ai1_time.total_time_s += ai1.total_time_s
        result.ai1_time.moves += ai1.moves
        result.ai2_time.total_time_s += ai2.total_time_s
        result.ai2_time.moves += ai2.moves

    return result


def print_result(result: SeriesResult) -> None:
    ai1_rate = 100.0 * result.ai1_wins / result.games if result.games else 0.0
    ai2_rate = 100.0 * result.ai2_wins / result.games if result.games else 0.0
    draw_rate = 100.0 * result.draws / result.games if result.games else 0.0

    print("\n=== Comparison ===")
    print(f"Variant:   {result.variant_name}")
    print(f"Algorithm: {result.algorithm_name}")
    print(f"Depths:    AI1={result.depth_ai1}, AI2={result.depth_ai2}")
    print(f"Games:     {result.games}")
    print(f"Starts:    AI1={result.ai1_starts}, AI2={result.ai2_starts}")
    print(f"Wins:      AI1={result.ai1_wins} ({ai1_rate:.1f}%), AI2={result.ai2_wins} ({ai2_rate:.1f}%)")
    print(f"Draws:     {result.draws} ({draw_rate:.1f}%)")
    print(
        "Avg move time (ms): "
        f"AI1={result.ai1_time.avg_time_ms:.3f} over {result.ai1_time.moves} moves, "
        f"AI2={result.ai2_time.avg_time_ms:.3f} over {result.ai2_time.moves} moves"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mandatory 6-point comparison for TicTacDoh: Negamax with alpha-beta "
            "vs Negamax without alpha-beta, with two depths, on deterministic and "
            "probabilistic variants, including average move decision times."
        )
    )
    parser.add_argument("--games", type=int, default=100, help="Games per comparison setup.")
    parser.add_argument("--depth-low", type=int, default=2, help="Lower search depth.")
    parser.add_argument("--depth-high", type=int, default=4, help="Higher search depth.")
    parser.add_argument("--miss-chance", type=float, default=0.2, help="Miss chance for probabilistic variant.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.games <= 0:
        raise ValueError("--games must be > 0")
    if args.depth_low <= 0 or args.depth_high <= 0:
        raise ValueError("depth values must be > 0")
    if args.depth_low == args.depth_high:
        raise ValueError("--depth-low and --depth-high must differ")
    if not 0.0 <= args.miss_chance <= 1.0:
        raise ValueError("--miss-chance must be in [0.0, 1.0]")

    setups = [
        ("negamax_ab", 0.0),
        ("negamax_ab", args.miss_chance),
        ("negamax_no_ab", 0.0),
        ("negamax_no_ab", args.miss_chance),
    ]

    print("Running mandatory 6-point comparisons...")
    print(
        f"Games per setup: {args.games}, depths: {args.depth_low} vs {args.depth_high}, "
        f"probabilistic miss chance: {args.miss_chance}, seed: {args.seed}"
    )

    for algo_name, miss_chance in setups:
        result = run_series(
            games=args.games,
            algorithm_name=algo_name,
            depth_ai1=args.depth_low,
            depth_ai2=args.depth_high,
            miss_chance=miss_chance,
            seed=args.seed,
        )
        print_result(result)


if __name__ == "__main__":
    main()
