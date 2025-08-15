from __future__ import annotations

"""Self‑contained **python‑chess** wrapper for MuZero‑general (no extra files).

Put this file at *muzero‑general/games/chess.py* **replacing** the original
PettingZoo version.  All action mappings are generated at import‑time, so there
is **no** dependency on `chess_utils.py` or any JSON.
"""

import datetime
import pathlib
from typing import List, Tuple, Optional

import numpy as np
import chess

# -----------------------------------------------------------------------------
# Build the 4 672‑action <‑‑> move mapping at runtime --------------------------
# -----------------------------------------------------------------------------

QUEEN_DIRS = [8, -8, -1, 1, 9, 7, -7, -9]  # N, S, W, E, NE, NW, SE, SW (0 = a1)
KNIGHT_OFFSETS = [17, 15, 10, 6, -6, -10, -15, -17]  # L‑shapes

# <<< 關鍵修正 1：補全動作空間，加入兵升變為騎士 'n' >>>
PROMO_PIECES = ["q", "r", "b", "n"]  # queen, rook, bishop, knight

MOVES_TO_ACTIONS: dict[str, int] = {}
ACTIONS_TO_MOVES: dict[int, str] = {}


def _on_board(sq: int) -> bool:
    return 0 <= sq < 64


def _same_file(a: int, b: int) -> bool:
    return (a % 8) == (b % 8)


def _same_rank(a: int, b: int) -> bool:
    return (a // 8) == (b // 8)


def _build_action_tables():
    action = 0
    for from_sq in chess.SQUARES:  # 0..63
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        # --- 56 queen‑like moves ------------------------------------------------
        for dir_idx, offset in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                to_sq = from_sq + offset * dist
                if not _on_board(to_sq): break
                to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
                if dir_idx in (2, 3) and not _same_rank(from_sq, to_sq): break
                if dir_idx in (0, 1) and not _same_file(from_sq, to_sq): break
                if dir_idx in (4, 7) and abs(from_file - to_file) != abs(from_rank - to_rank): break
                if dir_idx in (5, 6) and abs(from_file - to_file) != abs(from_rank - to_rank): break
                uci = chess.Move(from_sq, to_sq).uci()
                MOVES_TO_ACTIONS[uci] = action
                ACTIONS_TO_MOVES[action] = uci
                action += 1

        # --- 8 knight moves ----------------------------------------------------
        for k_offset in KNIGHT_OFFSETS:
            to_sq = from_sq + k_offset
            if not _on_board(to_sq): continue
            if max(abs(chess.square_file(to_sq) - from_file), abs(chess.square_rank(to_sq) - from_rank)) != 2: continue
            uci = chess.Move(from_sq, to_sq).uci()
            if uci not in MOVES_TO_ACTIONS:
                MOVES_TO_ACTIONS[uci] = action
                ACTIONS_TO_MOVES[action] = uci
                action += 1

        # --- 12 (3+1) pawn promotions ------------------------------------------
                # White promotions (rank 7 → 8)
        if from_rank == 6:
            promo_dirs = [8, 7, 9]  # forward, capture‑left, capture‑right
            for d in promo_dirs:
                to_sq = from_sq + d
                if not _on_board(to_sq):
                    continue
                for p in PROMO_PIECES:
                    uci = chess.Move(from_sq, to_sq, promotion=chess.PIECE_SYMBOLS.index(p)).uci()
                    MOVES_TO_ACTIONS[uci] = action
                    ACTIONS_TO_MOVES[action] = uci
                    action += 1
        # Black promotions (rank 2 → 1)
        if from_rank == 1:
            promo_dirs = [-8, -9, -7]  # forward, capture‑right, capture‑left (from white POV)
            for d in promo_dirs:
                to_sq = from_sq + d
                if not _on_board(to_sq):
                    continue
                for p in PROMO_PIECES:
                    uci = chess.Move(from_sq, to_sq, promotion=chess.PIECE_SYMBOLS.index(p)).uci()
                    MOVES_TO_ACTIONS[uci] = action
                    ACTIONS_TO_MOVES[action] = uci
                    action += 1
    
    # <<< 關鍵修正 2：更新斷言為精確計算後的新動作總數 >>>
    # 原 1930 + 升變為騎士新增的 48 個動作 = 1976
    assert action == 1976, f"Expected 1976 actions, got {action}"


_build_action_tables()

# ------------------------ Board encoding (13 × 8 × 8) -------------------------
PIECE_PLANES = {p: i for i, p in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING], 1)}

def encode_board(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((13, 8, 8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        plane = PIECE_PLANES[piece.piece_type] - 1 + (0 if piece.color == chess.WHITE else 6)
        r, c = divmod(sq, 8)
        tensor[plane, 7 - r, c] = 1
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
    return tensor

# -----------------------------------------------------------------------------
# MuZero configuration ---------------------------------------------------------
# -----------------------------------------------------------------------------
class MuZeroConfig:
    def __init__(self):
        # === 基本 ===
        self.game = "chess"
        self.observation_shape = (13, 8, 8)
        # <<< 關鍵修正 3：更新 action_space 以匹配新的動作總數 >>>
        self.action_space = list(range(1976))
        self.players = [0, 1]
        self.stacked_observations = 0

        # === MCTS ===
        self.num_simulations = 100 
        self.max_moves = 200  # 降低最大步數以鼓勵更早結束對局
        self.discount = 1
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.25
        self.root_dirichlet_alpha = self.dirichlet_alpha
        self.root_exploration_fraction = self.exploration_fraction
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # === 網路 ===
        self.network = "resnet"
        self.blocks = 20
        self.channels = 128
        self.reduced_channels_reward = 128
        self.reduced_channels_value = 128
        self.reduced_channels_policy = 128
        self.resnet_fc_reward_layers: List[int] = []
        self.resnet_fc_value_layers: List[int] = [256]
        self.resnet_fc_policy_layers: List[int] = [256]
        # <<< 關鍵修正 4：同步 support_size 以覆蓋新的獎勵範圍 >>>
        self.support_size = 3 # 範圍為 [-2, 2], 足以覆蓋 [-1.5, 1]
        self.downsample = "resnet"

        # === Optimizer ===
        self.optimizer = "SGD"
        self.momentum = 0.9

        # === 探索溫度 ===
        self.temperature_threshold = 30000
        self.visit_softmax_temperature_fn = lambda trained_steps, **kwargs: 1.0 if trained_steps < self.temperature_threshold else 0.1

        # === 訓練 ===
        self.training_steps = int(1e5)
        self.batch_size = 256
        self.checkpoint_interval = 5000
        self.window_size = 400_000
        self.lr_init = 2e-3
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = [30000, 70000]
        self.weight_decay = 1e-4

        # === 硬體 ===
        self.max_num_gpus = 4
        self.selfplay_on_gpu = True
        self.train_on_gpu = True
        self.reanalyse_on_gpu = False
        self.num_workers = 70
        self.num_actors = 70
        self.use_last_model_value = False

        # === 其他 ===
        self.seed = 3407
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_path = pathlib.Path("results") / self.game / ts
        self.opponent = "self"
        self.muzero_player: Optional[int] = None
        self.save_model = True
        self.self_play_delay = 0
        self.PER = False
        self.PER_alpha = 0.5
        self.PER_beta = 1.0
        self.PER_epsilon = 0.1
        self.ratio = 1
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.value_loss_weight = 1.0
        self.reward_loss_weight = 1.0
        self.policy_loss_weight = 1.0
        self.training_delay = 0

# -----------------------------------------------------------------------------
# Game wrapper -----------------------------------------------------------------
# -----------------------------------------------------------------------------
class Game:
    def __init__(self, seed: int | None = None, max_moves: int = 250):
        self.board = chess.Board()
        if seed is not None:
            np.random.seed(seed)
        self.to_play_history: List[int] = []
        self.max_moves = max_moves

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply indexed *action* and return (obs, reward, done)."""
        uci = ACTIONS_TO_MOVES.get(action)
        if uci is None: raise ValueError(f"Action {action} is not defined.")
        
        move = chess.Move.from_uci(uci)
        
        # --- 诊断代码 (保持不变) ---
        if move not in self.board.legal_moves:
            print("\n" + "="*60)
            print("!!!!!! FATAL ASSERTION ERROR: MOVE IS ILLEGAL !!!!!!")
            print(f"BOARD STATE (FEN): {self.board.fen()}")
            print(f"ILLEGAL MOVE SUGGESTED BY MCTS: {uci}")
            print("="*60 + "\n")
            raise ValueError(f"Illegal move {uci} was proposed for FEN {self.board.fen()}")
        
        # --- 奖励计算逻辑 (重构后) ---
        
        # 1. 初始化奖励
        reward = 0.0
        
        # 2. 判断是否吃子 (必须在 push 之前)
        # 我们检查目标格子上是否有棋子，并且颜色与当前玩家不同
        captured_piece = self.board.piece_at(move.to_square)
        if captured_piece is not None and captured_piece.color != self.board.turn:
            piece_values = {
                chess.PAWN: 0.04,
                chess.KNIGHT: 0.12,
                chess.BISHOP: 0.12,
                chess.ROOK: 0.2,
                chess.QUEEN: 0.36,
            }
            # 只有在吃掉对方子时才加分
            reward += piece_values.get(captured_piece.piece_type, 0.0)

        # 3. 执行走法
        current_player = self.to_play()
        self.board.push(move)
        self.to_play_history.append(current_player)
        
        # 4. 判断游戏是否结束，并设定最终奖励
        done = self.board.is_game_over()
        
        if done:
            res = self.board.result()
            if res == "1-0":
                # 无论之前吃了多少子，最终奖励都是 +1.0
                reward = 1.0
            elif res == "0-1":
                # 无论之前吃了多少子，最终奖励都是 -1.0
                reward = -1.0
            else:
                # 正常和棋，最终奖励是 0.0
                reward = 0.0
        
        # 5. 检查是否达到最大步数限制
        # 这个检查必须在终局判断之后
        if len(self.to_play_history) >= self.max_moves and not done:
            done = True
            # 如果是因为步数耗尽而和棋，最终奖励被覆盖为惩罚
            reward = -0.5

        # 6. 如果游戏仍未结束，施加微小的步数惩罚
        # 这个惩罚不会覆盖吃子奖励，而是累加在上面
        if not done:
            reward -= 0.001
            
        return encode_board(self.board), reward, done

    def legal_actions(self) -> List[int]:
        # 確保不會因為動作空間不完整而返回空列表
        legal_moves = [MOVES_TO_ACTIONS[m.uci()] for m in self.board.legal_moves if m.uci() in MOVES_TO_ACTIONS]
        if not legal_moves:
             # 如果過濾後沒有合法走法，這是一個嚴重問題，但為了不假死，先返回一個隨機動作
             # 這在修正了動作空間後幾乎不可能發生
             return [list(ACTIONS_TO_MOVES.keys())[0]] if ACTIONS_TO_MOVES else []
        return legal_moves

    def to_play(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1
    
    def reset(self) -> np.ndarray:
        self.board.reset()
        self.to_play_history.clear()
        return encode_board(self.board)

    def clone(self) -> "Game":
        g = Game(max_moves=self.max_moves)
        g.board = self.board.copy(stack=True)
        g.to_play_history = list(self.to_play_history)
        return g

    # --- 其他輔助方法保持不變 ---
    def render(self): print(self.board)
    def action_to_string(self, a: int) -> str: return ACTIONS_TO_MOVES.get(a, str(a))
    def human_to_action(self) -> Optional[int]: u = input("move: ").strip(); return None if u == "quit" else MOVES_TO_ACTIONS.get(u)
    def expert_agent(self) -> Optional[int]: l = self.legal_actions(); return int(np.random.choice(l)) if l else None
    def close(self): pass