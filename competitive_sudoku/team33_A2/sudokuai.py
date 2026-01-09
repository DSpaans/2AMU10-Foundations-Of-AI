#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
import time
from typing import List, Tuple, Set, Optional, Dict, Any

import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

Square = Tuple[int, int]

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

        # -----------------
        # Neural net config
        # -----------------
        # A tiny MLP over hand-crafted features. The weights are persisted via
        # self.save/self.load (A2 feature), and updated online with a lightweight
        # TD(0)-style update.
        self._nn_hidden = 32
        self._nn_gamma = 0.95
        self._nn_lr = 0.01
        self._nn_scale = 0.25  # how much the NN influences the final eval

        self._nn_params: Optional[Dict[str, np.ndarray]] = None
        self._nn_state: Dict[str, Any] = {
            "turn": 0,
            "last_x": None,          # features of previous observed state
            "last_score_diff": 0.0,  # previous score diff from our perspective
            "saved_every": 5,        # save cadence (turns)
        }

        # Deterministic init for reproducibility across runs.
        self._rng = np.random.default_rng(33)

    # -------------------------
    # Neural network primitives
    # -------------------------
    def _nn_init_if_needed(self, input_dim: int) -> None:
        if self._nn_params is not None:
            return
        h = self._nn_hidden

        # Xavier-ish init; keep small outputs.
        w1 = self._rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(h, input_dim)).astype(np.float32)
        b1 = np.zeros((h,), dtype=np.float32)
        w2 = self._rng.normal(0.0, 1.0 / math.sqrt(h), size=(1, h)).astype(np.float32)
        b2 = np.zeros((1,), dtype=np.float32)
        self._nn_params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _nn_forward(self, x: np.ndarray) -> float:
        """Forward pass; returns scalar in roughly [-1, 1] due to tanh."""
        assert self._nn_params is not None
        w1, b1, w2, b2 = self._nn_params["w1"], self._nn_params["b1"], self._nn_params["w2"], self._nn_params["b2"]

        z1 = w1 @ x + b1
        a1 = np.tanh(z1)
        z2 = w2 @ a1 + b2
        out = float(np.tanh(z2[0]))
        return out

    def _nn_train_step(self, x: np.ndarray, target: float) -> None:
        """One-step SGD on squared error loss for the tiny MLP."""
        assert self._nn_params is not None
        w1, b1, w2, b2 = self._nn_params["w1"], self._nn_params["b1"], self._nn_params["w2"], self._nn_params["b2"]

        # Forward (keep intermediates)
        z1 = w1 @ x + b1
        a1 = np.tanh(z1)
        z2 = w2 @ a1 + b2
        y = np.tanh(z2[0])

        # Loss: (y - target)^2
        # Backprop through tanh
        dy = 2.0 * (y - target)
        dz2 = dy * (1.0 - y * y)

        # Gradients
        dw2 = (dz2 * a1).reshape(w2.shape)
        db2 = np.array([dz2], dtype=np.float32)

        da1 = (w2.reshape(-1) * dz2)
        dz1 = da1 * (1.0 - a1 * a1)
        dw1 = np.outer(dz1, x).astype(np.float32)
        db1 = dz1.astype(np.float32)

        lr = self._nn_lr
        self._nn_params["w2"] = (w2 - lr * dw2).astype(np.float32)
        self._nn_params["b2"] = (b2 - lr * db2).astype(np.float32)
        self._nn_params["w1"] = (w1 - lr * dw1).astype(np.float32)
        self._nn_params["b1"] = (b1 - lr * db1).astype(np.float32)

    # ---------------------
    # Feature construction
    # ---------------------
    def _features(self, game_state: GameState, me: int) -> np.ndarray:
        """Returns a small, normalized feature vector for the neural net."""
        board = game_state.board
        N, m, n = board.N, board.m, board.n
        opp = 2 if me == 1 else 1

        # Squares a given player may play on (starting region + neighbors of occupied),
        # mirroring GameState.player_squares() but parameterized by player.
        def playable_squares_for(player: int) -> Optional[List[Square]]:
            allowed = game_state.allowed_squares1 if player == 1 else game_state.allowed_squares2
            occupied = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2

            if allowed is None:
                return None

            def is_empty(sq: Square) -> bool:
                return board.get(sq) == SudokuBoard.empty

            def neighbors(square: Square):
                r, c = square
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < N and 0 <= cc < N:
                            yield (rr, cc)

            result = [sq for sq in allowed if is_empty(sq)]
            for s1 in occupied:
                for s2 in neighbors(s1):
                    if is_empty(s2):
                        result.append(s2)

            return sorted(list(set(result)))

        def empty_playable(player: int) -> List[Square]:
            playable = playable_squares_for(player)
            if playable is None:
                return [(r, c) for r in range(N) for c in range(N) if board.get((r, c)) == SudokuBoard.empty]
            return playable

        def region_id(sq: Square) -> int:
            r, c = sq
            return (r // m) * (N // n) + (c // n)

        # Precompute empties per unit
        row_empty = [0] * N
        col_empty = [0] * N
        reg_empty = [0] * N
        filled = 0
        for r in range(N):
            for c in range(N):
                if board.get((r, c)) == SudokuBoard.empty:
                    row_empty[r] += 1
                    col_empty[c] += 1
                    reg_empty[region_id((r, c))] += 1
                else:
                    filled += 1

        def finisher_units_sum(player: int) -> int:
            total = 0
            for (r, c) in empty_playable(player):
                rid = region_id((r, c))
                total += int(row_empty[r] == 1)
                total += int(col_empty[c] == 1)
                total += int(reg_empty[rid] == 1)
            return total

        # Candidate-based "forced move" counts (sudoku-consistency only)
        taboo_set = {(tm.square, tm.value) for tm in game_state.taboo_moves}

        def region_cells(square: Square):
            r, c = square
            r0 = (r // m) * m
            c0 = (c // n) * n
            for rr in range(r0, r0 + m):
                for cc in range(c0, c0 + n):
                    yield (rr, cc)

        def candidates(square: Square) -> List[int]:
            r, c = square
            used = {board.get((r, cc)) for cc in range(N)} | {board.get((rr, c)) for rr in range(N)} | {board.get(s) for s in region_cells(square)}
            used.discard(SudokuBoard.empty)
            return [v for v in range(1, N + 1) if v not in used and (square, v) not in taboo_set]

        def forced_count(player: int) -> int:
            cnt = 0
            for sq in empty_playable(player):
                if len(candidates(sq)) == 1:
                    cnt += 1
            return cnt

        # Core scalar features (normalized)
        score_diff = float(game_state.scores[me - 1] - game_state.scores[opp - 1])
        open_me = len(empty_playable(me))
        open_opp = len(empty_playable(opp))

        # Normalize to roughly [-1, 1]
        max_score_scale = 7.0 * N  # loose upper bound per completed units
        score_diff_n = score_diff / max_score_scale

        open_diff_n = (open_me - open_opp) / float(N * N)
        open_me_n = open_me / float(N * N)

        fin_diff_n = (finisher_units_sum(me) - finisher_units_sum(opp)) / float(3 * N * N)
        fin_me_n = finisher_units_sum(me) / float(3 * N * N)

        filled_frac = filled / float(N * N)
        forced_diff_n = (forced_count(me) - forced_count(opp)) / float(N * N)

        # Bias + a small set of signals
        x = np.array([
            1.0,
            score_diff_n,
            open_diff_n,
            open_me_n,
            fin_diff_n,
            fin_me_n,
            forced_diff_n,
            filled_frac,
        ], dtype=np.float32)
        return x

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

        # Squares a given player may play on (starting region + neighbors of occupied),
        # mirroring GameState.player_squares() but parameterized by player.
        def playable_squares_for(player: int) -> Optional[List[Square]]:
            allowed = game_state.allowed_squares1 if player == 1 else game_state.allowed_squares2
            occupied = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2

            if allowed is None:
                return None

            def is_empty(sq: Square) -> bool:
                return board.get(sq) == SudokuBoard.empty

            def neighbors(square: Square):
                r, c = square
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < N and 0 <= cc < N:
                            yield (rr, cc)

            result = [sq for sq in allowed if is_empty(sq)]
            for s1 in occupied:
                for s2 in neighbors(s1):
                    if is_empty(s2):
                        result.append(s2)
            return sorted(list(set(result)))

        def empty_playable(player: int) -> List[Square]:
            playable = playable_squares_for(player)
            if playable is None:
                return [(r, c) for r in range(N) for c in range(N) if board.get((r, c)) == SudokuBoard.empty]
            return playable

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
            for (r, c) in empty_playable(player):
                rid = region_id((r, c))
                total += int(row_empty[r] == 1)
                total += int(col_empty[c] == 1)
                total += int(reg_empty[rid] == 1)
            return total
    
        # --- Components
        score_diff = game_state.scores[me - 1] - game_state.scores[opp - 1]
        open_diff = len(empty_playable(me)) - len(empty_playable(opp))
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

        # -------------------------
        # Load / update NN weights
        # -------------------------
        # We load at the start of the turn. If time runs out while loading,
        # the framework will end the turn immediately after load completes.
        # (This is acceptable per the assignment description.)
        persisted = self.load()
        if isinstance(persisted, dict):
            if "nn_params" in persisted and persisted["nn_params"] is not None:
                self._nn_params = persisted["nn_params"]
            if "nn_state" in persisted and isinstance(persisted["nn_state"], dict):
                # keep defaults for missing keys
                self._nn_state.update(persisted["nn_state"])

        x_now = self._features(game_state, me)
        self._nn_init_if_needed(int(x_now.shape[0]))

        # Lightweight TD(0)-style update across turns:
        # target = r + gamma * V(s') where r is observed score-diff change.
        # This is deliberately cheap and robust.
        cur_score_diff = float(game_state.scores[me - 1] - game_state.scores[(2 if me == 1 else 1) - 1])
        v_now = self._nn_forward(x_now)

        last_x = self._nn_state.get("last_x", None)
        last_sd = float(self._nn_state.get("last_score_diff", 0.0))
        if isinstance(last_x, np.ndarray) and last_x.shape == x_now.shape:
            # Normalize reward roughly to the NN output scale
            # (score diff can jump by up to 7 points on a single move).
            r = (cur_score_diff - last_sd) / 7.0
            target = float(np.clip(r + self._nn_gamma * v_now, -1.0, 1.0))
            self._nn_train_step(last_x, target)

        # Update stored state for next call
        self._nn_state["last_x"] = x_now
        self._nn_state["last_score_diff"] = cur_score_diff
        self._nn_state["turn"] = int(self._nn_state.get("turn", 0)) + 1

        # Persist occasionally (avoid paying IO cost every turn)
        save_every = int(self._nn_state.get("saved_every", 5))
        if save_every > 0 and (self._nn_state["turn"] % save_every == 0):
            self.save({"nn_params": self._nn_params, "nn_state": self._nn_state})

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
            return [v for v in range(1, N + 1) if v not in used and (square, v) not in taboo_set]

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

        # Scoring rule from the competitive sudoku rules: 0/1/3/7 points for
        # completing 0/1/2/3 regions in a single move.
        REWARD_BY_COMPLETIONS = (0, 1, 3, 7)

        def reward_for_square(square: Square) -> int:
            k = completed_units_if_filled(square)
            return REWARD_BY_COMPLETIONS[k]

        def legal_moves(state: GameState) -> List[Move]:
            allowed = state.player_squares()
            if allowed is None:
                allowed = [(r, c) for r in range(N) for c in range(N)]

            # --- Generate moves + identify forced moves (single candidate)
            forced: List[Move] = []
            others: List[Move] = []
            for sq in allowed:
                if board.get(sq) != SudokuBoard.empty:
                    continue
                cands = candidates(sq)
                if not cands:
                    continue
                if len(cands) == 1:
                    forced.append(Move(sq, cands[0]))
                else:
                    for v in cands:
                        others.append(Move(sq, v))

            moves = forced if forced else (forced + others)

            # --- Heuristic ordering for pruning efficiency
            # Primary: complete-more-units; Secondary: prefer more constrained squares.
            def key(mv: Move):
                sq = mv.square
                # candidates(sq) is cheap enough here and helps ordering
                return (completed_units_if_filled(sq), -len(candidates(sq)))

            moves.sort(key=key, reverse=True)
            return moves

        def eval_state(state: GameState) -> float:
            """Heuristic eval + NN correction (from our fixed perspective)."""
            base = self.evaluate_state_simple(state, me)
            x = self._features(state, me)
            nn = self._nn_forward(x)
            # scale NN to be a meaningful tie-breaker but not override points
            return base + 25.0 * nn

        def select_moves_for_search(state: GameState, depth_remaining: int) -> List[Move]:
            """Heuristic + NN move ordering with a beam-width cap to reduce branching."""
            moves = legal_moves(state)
            if len(moves) <= 1:
                return moves

            # Beam width schedule: narrower deeper in the tree.
            if depth_remaining >= 3:
                beam = 14
            elif depth_remaining == 2:
                beam = 18
            else:
                beam = 26
            beam = min(beam, len(moves))

            # Refine ordering of top candidates using a quick NN look-ahead.
            refine = min(len(moves), max(beam, 30))
            head = moves[:refine]
            tail = moves[refine:]

            scored = []
            for mv in head:
                tok = apply_move(state, mv)
                v = eval_state(state)
                undo_move(state, tok)
                scored.append((v, mv))
            scored.sort(key=lambda t: t[0], reverse=True)

            ordered = [mv for _, mv in scored] + tail
            return ordered[:beam]

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
                return eval_state(state)

            moves = select_moves_for_search(state, depth)
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
        root_moves = select_moves_for_search(game_state, depth_remaining=1)
        if not root_moves:
            return

        best_move = root_moves[0]
        self.propose_move(best_move)

        # --- Iterative deepening: keep improving best_move; propose whenever it improves
        depth = 1
        MAX_DEPTH = 8  # avoid runaway recursion; framework time-limit still governs anytime behavior
        while True:
            root_moves = select_moves_for_search(game_state, depth_remaining=depth)

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

            if depth > MAX_DEPTH:
                depth = MAX_DEPTH

            # tiny yield; does not assume a known time budget
            time.sleep(0)