#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
import time
from typing import List, Tuple, Set
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

Square = Tuple[int, int]

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def evaluate_state_simple(self, game_state: GameState, me: int) -> float:
        """
        Simple evaluation (ignores move legality/value constraints):
        1) points difference
        2) empty-allowed squares difference
        3) finisher-units difference:
            for each allowed empty square, count how many units (row/col/region)
            would be completed if that square were filled (0..3), then sum.
        """
        board = game_state.board
        N, m, n = board.N, board.m, board.n
        opp = 2 if me == 1 else 1

        def region_id(sq: Square) -> int:
            r, c = sq
            return (r // m) * (N // n) + (c // n)

        def allowed_for(player: int) -> List[Square]:
            allowed = game_state.allowed_squares1 if player == 1 else game_state.allowed_squares2
            if allowed is None:
                # Classic mode: all squares are allowed
                return [(r, c) for r in range(N) for c in range(N)]
            return allowed

        def empty_allowed(player: int) -> List[Square]:
            return [sq for sq in allowed_for(player) if board.get(sq) == SudokuBoard.empty]

        # --- Precompute empties per row/col/region on the whole board
        row_empty = [0] * N
        col_empty = [0] * N
        reg_empty = [0] * N  # typically N regions when N = m*n

        for r in range(N):
            for c in range(N):
                if board.get((r, c)) == SudokuBoard.empty:
                    row_empty[r] += 1
                    col_empty[c] += 1
                    reg_empty[region_id((r, c))] += 1

        def finisher_units_sum(player: int) -> int:
            """
            Sum over allowed empty squares:
            +1 if it's the last empty in its row
            +1 if it's the last empty in its column
            +1 if it's the last empty in its region
            """
            total = 0
            for (r, c) in empty_allowed(player):
                rid = region_id((r, c))
                total += int(row_empty[r] == 1)
                total += int(col_empty[c] == 1)
                total += int(reg_empty[rid] == 1)
            return total
    
        # --- Components
        score_diff = game_state.scores[me - 1] - game_state.scores[opp - 1]
        open_diff = len(empty_allowed(me)) - len(empty_allowed(opp))
        finish_units_diff = finisher_units_sum(me) - finisher_units_sum(opp)

        # --- Weights (points dominant; tune if needed)
        W_SCORE = 100.0
        W_OPEN = 1.0
        W_FINISH_UNITS = 5.0

        return W_SCORE * score_diff + W_OPEN * open_diff + W_FINISH_UNITS * finish_units_diff

    # # N.B. This is a very naive implementation.
    # def compute_best_move(self, game_state: GameState) -> None:
    #     N = game_state.board.N

    #     # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
    #     def possible(i, j, value):
    #         return game_state.board.get((i, j)) == SudokuBoard.empty \
    #                and not TabooMove((i, j), value) in game_state.taboo_moves \
    #                    and (i, j) in game_state.player_squares()

    #     all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
    #                  for value in range(1, N+1) if possible(i, j, value)]
    #     move = random.choice(all_moves)
    #     self.propose_move(move)
    #     while True:
    #         time.sleep(0.2)
    #         self.propose_move(random.choice(all_moves))
    
    
    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime alpha-beta minimax with iterative deepening.

        - Core search: minimax with alpha-beta pruning.
        - Anytime behavior: call self.propose_move(best_move) whenever the
        current best root move changes, and keep searching deeper.

        This does NOT assume a known time budget; the driver is expected to stop
        this method externally at the end of the turn.
        """
        board = game_state.board
        N, m, n = board.N, board.m, board.n
        me = game_state.current_player

        taboo_set = {(tm.square, tm.value) for tm in game_state.taboo_moves}

        def region_cells(square: Square):
            r, c = square
            r0 = (r // m) * m
            c0 = (c // n) * n
            for rr in range(r0, r0 + m):
                for cc in range(c0, c0 + n):
                    yield (rr, cc)

        def used_in_row(r: int) -> Set[int]:
            return {board.get((r, c)) for c in range(N)} - {SudokuBoard.empty}

        def used_in_col(c: int) -> Set[int]:
            return {board.get((r, c)) for r in range(N)} - {SudokuBoard.empty}

        def used_in_region(square: Square) -> Set[int]:
            return {board.get(s) for s in region_cells(square)} - {SudokuBoard.empty}

        def candidates(square: Square) -> List[int]:
            r, c = square
            used = used_in_row(r) | used_in_col(c) | used_in_region(square)
            return [v for v in range(1, N + 1) if v not in used]

        # how many of (row, col, region) would become complete if we fill 'square'?
        # (depends only on emptiness, not on the value)
        def completed_units_if_filled(square: Square) -> int:
            r, c = square

            row_complete = all(
                board.get((r, cc)) != SudokuBoard.empty or (r, cc) == square
                for cc in range(N)
            )
            col_complete = all(
                board.get((rr, c)) != SudokuBoard.empty or (rr, c) == square
                for rr in range(N)
            )
            reg_complete = all(
                board.get(s) != SudokuBoard.empty or s == square
                for s in region_cells(square)
            )
            return int(row_complete) + int(col_complete) + int(reg_complete)

        # If your rules_full.pdf uses a different reward table, change this mapping.
        REWARD_BY_COMPLETIONS = (0, 1, 2, 3)

        def reward_for_square(square: Square) -> int:
            k = completed_units_if_filled(square)
            return REWARD_BY_COMPLETIONS[k]

        def legal_moves(state: GameState) -> List[Move]:
            allowed = state.player_squares()
            if allowed is None:
                allowed = [(r, c) for r in range(N) for c in range(N)]

            moves: List[Move] = []
            for sq in allowed:
                if board.get(sq) != SudokuBoard.empty:
                    continue
                for v in candidates(sq):
                    if (sq, v) in taboo_set:
                        continue
                    moves.append(Move(sq, v))

            # Move ordering helps alpha-beta pruning: complete-more-units first; tie-break by fewer candidates.
            def key(mv: Move):
                sq = mv.square
                return (completed_units_if_filled(sq), -len(candidates(sq)))

            moves.sort(key=key, reverse=True)
            return moves

        def apply_move(state: GameState, mv: Move):
            p = state.current_player
            sq = mv.square

            # compute reward BEFORE placing (square is still empty)
            rew = reward_for_square(sq)

            old_val = board.get(sq)  # should be empty
            token = (p, sq, old_val, rew)

            board.put(sq, mv.value)
            state.moves.append(mv)

            # needed for player_squares() in non-classic games
            if p == 1 and state.occupied_squares1 is not None:
                state.occupied_squares1.append(sq)
            elif p == 2 and state.occupied_squares2 is not None:
                state.occupied_squares2.append(sq)

            state.scores[p - 1] += rew
            state.current_player = 2 if p == 1 else 1
            return token

        def undo_move(state: GameState, token):
            p, sq, old_val, rew = token

            state.current_player = p
            state.scores[p - 1] -= rew

            if p == 1 and state.occupied_squares1 is not None:
                state.occupied_squares1.pop()
            elif p == 2 and state.occupied_squares2 is not None:
                state.occupied_squares2.pop()

            state.moves.pop()
            board.put(sq, old_val)

        def alphabeta(state: GameState, depth: int, alpha: float, beta: float) -> float:
            if depth == 0:
                return self.evaluate_state_simple(state, me)

            moves = legal_moves(state)
            if not moves:
                # No legal moves: losing for player to move (standard minimax convention)
                return -1e9 if state.current_player == me else 1e9

            maximizing = (state.current_player == me)

            if maximizing:
                value = -math.inf
                for mv in moves:
                    tok = apply_move(state, mv)
                    value = max(value, alphabeta(state, depth - 1, alpha, beta))
                    undo_move(state, tok)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = math.inf
                for mv in moves:
                    tok = apply_move(state, mv)
                    value = min(value, alphabeta(state, depth - 1, alpha, beta))
                    undo_move(state, tok)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value

        # --- Anytime: propose something immediately
        root_moves = legal_moves(game_state)
        if not root_moves:
            return

        best_move = root_moves[0]
        self.propose_move(best_move)

        # --- Iterative deepening: keep improving best_move; propose whenever it improves
        depth = 1
        while True:
            root_moves = legal_moves(game_state)

            best_val = -math.inf
            alpha, beta = -math.inf, math.inf

            for mv in root_moves:
                tok = apply_move(game_state, mv)
                val = alphabeta(game_state, depth - 1, alpha, beta)
                undo_move(game_state, tok)

                if val > best_val:
                    best_val = val
                    best_move = mv
                    self.propose_move(best_move)  # anytime update

                alpha = max(alpha, best_val)

            depth += 1

            # tiny yield; does not assume a known time budget
            time.sleep(0)