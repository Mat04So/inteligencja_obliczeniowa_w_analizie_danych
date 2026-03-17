#file contain a tic tac doh game but with a probabilistic element, where players have a chance to miss their move 20%
#if a player misses their move, the opponent gets to play twice in a row
import os
import random
import sys
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from easyAI import AI_Player, Human_Player, Negamax, TwoPlayerGame


class TicTacToe(TwoPlayerGame):
    """The board positions are numbered as follows:
    1 2 3
    4 5 6
    7 8 9
    """

    def __init__(self, players, miss_chance=0.2):
        self.players = players
        self.board = [0 for i in range(9)]
        self.current_player = 1  # player 1 starts.
        self.miss_chance = miss_chance
        self.double_turn_player = None

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        self.board[int(move) - 1] = self.current_player

    def unmake_move(self, move):  # optional method (speeds up the AI)
        self.board[int(move) - 1] = 0

    def lose(self):
        """ Has the opponent "three in line ?" """
        return any(
            [
                all([(self.board[c - 1] == self.opponent_index) for c in line])
                for line in [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],  # horiz.
                    [1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9],  # vertical
                    [1, 5, 9],
                    [3, 5, 7],
                ]
            ]
        )  # diagonal

    def is_over(self):
        return (self.possible_moves() == []) or self.lose()

    def show(self):
        print(
            "\n"
            + "\n".join(
                [
                    " ".join([[".", "O", "X"][self.board[3 * j + i]] for i in range(3)])
                    for j in range(3)
                ]
            )
        )

    def scoring(self):
        return -100 if self.lose() else 0

    def play(self, nmoves=1000, verbose=True):
        history = []

        if verbose:
            self.show()

        for self.nmove in range(1, nmoves + 1):
            if self.is_over():
                break

            move = self.player.ask_move(self)
            history.append((deepcopy(self), move))

            player_before_move = self.current_player
            missed = random.random() < self.miss_chance

            if not missed:
                self.make_move(move)

            if verbose:
                if missed:
                    print(
                        "\nMove #%d: player %d misses %s :"
                        % (self.nmove, player_before_move, str(move))
                    )
                else:
                    print(
                        "\nMove #%d: player %d plays %s :"
                        % (self.nmove, player_before_move, str(move))
                    )
                self.show()

            if missed:
                self.current_player = 2 if player_before_move == 1 else 1
                self.double_turn_player = self.current_player
            elif self.double_turn_player == player_before_move:
                self.current_player = player_before_move
                self.double_turn_player = None
            else:
                self.current_player = 2 if player_before_move == 1 else 1

        history.append(deepcopy(self))
        return history


if __name__ == "__main__":
    ai_algo = Negamax(6)
    TicTacToe([Human_Player(), AI_Player(ai_algo)]).play()
