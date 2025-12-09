"""
Local-Stockfish DofBot chess controller
- Returns a FEN notation after each move
- Asks: “Your move?” or type “hint” for engine suggestion
- User can adjust difficulty from
"""
import time, json
#from Arm_Lib import Arm_Device
import chess
import os
import subprocess
from pathlib import Path
import chess.svg
from typing import List
from IPython.core.display import HTML
from IPython.core.display_functions import display

# ---------- optional pretty board ----------
try:
    HAS_SVG = True
except ImportError:
    HAS_SVG = False

# ---------- Elo-based difficulty ----------
while True:
    try:
        user_elo = int(input("Choose opponent strength (500-2000 Elo): "))
        if 500 <= user_elo <= 2000:
            break
        print("Please enter a number between 500 and 2000.")
    except ValueError:
        print("Whole numbers only.")

# very small table: depth → realistic Elo (SF 16, 1+0.1 bullet vs humans)
ELO_TO_DEPTH = {
    500: 5,  700: 6,  900: 7, 1100: 8,
    1300: 9, 1500: 10, 1700: 11, 1900: 12, 2000: 13
}
ENGINE_DEPTH = ELO_TO_DEPTH[min(ELO_TO_DEPTH.keys(), key=lambda x: abs(x - user_elo))]
print(f"Engine set to depth {ENGINE_DEPTH}  (≈ {user_elo} Elo)")

# ---------- optional blunder factor for ≤ 1100 ----------
import random
BLUNDER_PCT = max(0, (1100 - user_elo) / 7)   # 0-8.5 % random blunders
print(f"Blunder chance: {BLUNDER_PCT:.1f} %\n")


# ---------- config ----------
class ChessConfig:
    SIMULATION        = True
    ALL_SQUARES_JSON  = "all_squares_touch.json"
    ENGINE_DEPTH      = 12
    STOCKFISH_EXE     = "stockfish-windows-x86-64-avx2.exe"
    # joint presets
    NEUTRAL_POS       = [90, 150, 10, 90, 264, 180]
    NEUTRAL_POS_OPEN  = [90, 150, 10, 90, 264, 145]
    RM_POS_CLOSE      = [30, 56, 19, 42, 264, 180]
    RM_POS_OPEN       = [30, 56, 19, 42, 264, 150]

config = ChessConfig()

# ---------- arm ----------
class ArmController:
    def __init__(self):
        self.sim = config.SIMULATION
        self.log = []
        self.json = {}
        if os.path.exists(config.ALL_SQUARES_JSON):
            self.json = json.load(open(config.ALL_SQUARES_JSON))
            print(f"✓ Loaded {len(self.json)} square→servo mappings")
        else:
            print(" all_squares_touch.json missing – using neutral poses")

    def move(self, joints: List[int], ms: int = 1000):
        self.log.append(joints)
        if self.sim:
            print(f"  → Servos: {joints}  ({ms} ms)")

    def sq_pose(self, sq: str, claw: int) -> List[int]:
        sq = sq.lower()
        if sq not in self.json:
            print(f"  {sq} not mapped – fallback")
            return config.NEUTRAL_POS[:5] + [claw]
        rank = int(sq[1])
        base = self.json[sq][:5] + [claw]
        if 1 <= rank <= 3: base[1] += 3
        elif 4 <= rank <= 6: base[1] += 6
        return base

    def move_piece(self, src: str, dst: str):
        print(f" Moving {src.upper()} → {dst.upper()}")
        src_close = self.sq_pose(src, 180)
        dst_close = self.sq_pose(dst, 180)
        src_open  = self.sq_pose(src, 140); src_open[1] += 3
        dst_open  = self.sq_pose(dst, 140)

        self.move(config.NEUTRAL_POS_OPEN, 800)
        self.move(src_open, 2000)
        self.move(src_close, 500)
        self.move(config.NEUTRAL_POS, 2000)
        self.move(dst_close, 2000)
        self.move(dst_open, 500)
        self.move(config.NEUTRAL_POS, 2000)

    def remove_piece(self, sq: str):
        print(f" Removing piece at {sq.upper()}")
        src_close = self.sq_pose(sq, 180)
        src_open  = self.sq_pose(sq, 140)

        self.move(config.NEUTRAL_POS_OPEN, 2000)
        self.move(src_open, 2500)
        self.move(src_close, 500)
        self.move(config.NEUTRAL_POS, 3000)
        self.move(config.RM_POS_CLOSE, 2500)
        self.move(config.RM_POS_OPEN, 700)
        self.move(config.NEUTRAL_POS_OPEN, 2500)

# ---------- local stockfish ----------
class LocalEngine:
    def __init__(self):
        import chess.engine
        exe = Path(config.STOCKFISH_EXE)
        if not exe.exists():
            print("⬇️  Downloading Stockfish …")
            url = ("https://github.com/official-stockfish/Stockfish/releases/download/sf_16/"
                   "stockfish-windows-x86-64-avx2.exe")
            subprocess.run(
                ["powershell", "-Command", f"Invoke-WebRequest -Uri {url} -OutFile {exe}"],
                check=True
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(
            str(exe.absolute()),
            cwd=exe.parent,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        print("✅ Local Stockfish ready")

    def analyse(self, board: chess.Board, depth: int = None):
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth or config.ENGINE_DEPTH))
        score = info["score"].white()
        if score.is_mate():
            mate = score.mate()
            cp = 30_000 + mate if mate > 0 else -30_000 - abs(mate)
        else:
            cp = score.score()
        return {
            "evaluation": cp,
            "best_move":  info["pv"][0].uci() if info.get("pv") else None
        }

    def close(self):
        self.engine.quit()

# ---------- game ----------
class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.arm   = ArmController()
        self.engine = LocalEngine()
        self.history = []

    # ---------- text board ----------
    def show_board(self):
        print("\n" + "=" * 50)
        #print(self.board)  # always print text
        print("FEN:", self.board.fen())
        if HAS_SVG:
            display(HTML(chess.svg.board(self.board, size=350)))
        print("=" * 50 + "\n")

    # ---------- classify ----------
    def _classify(self, before: int, after: int) -> str:
        if before is None or after is None:
            return ""
        side = 1 if self.board.turn == chess.BLACK else -1   # who just moved
        delta = (before - after) * side                      # positive = bad
        if delta <= 30:   return " Good move"
        if delta <= 90:   return " Inaccurate"
        if delta <= 180:  return " Mistake"
        return " Blunder"

    # ---------- play one move ----------
    def make_move(self, src_dst: str) -> bool:
        """accept 'e2e4' or 'e2 e4' or SAN 'Nf3'"""
        print(f"\nProcessing: {src_dst}")
        try:
            if len(src_dst.split()) == 2:                     # e2 e4
                src, dst = src_dst.split()
                mv = chess.Move.from_uci(src + dst)
            elif len(src_dst) == 4:                           # e2e4
                mv = chess.Move.from_uci(src_dst)
            else:                                             # SAN
                mv = self.board.parse_san(src_dst)
        except ValueError:
            print("Illegal / unparseable move")
            return False

        if mv not in self.board.legal_moves:
            print("Illegal move")
            return False

        # ---- evaluate BEFORE ----
        b_eval = self.engine.analyse(self.board)["evaluation"]

        # ---- physical simulation ----
        is_cap = self.board.is_capture(mv)
        if is_cap:
            capt_sq = chess.square_name(mv.to_square)
            if self.board.is_en_passant(mv):
                capt_sq = chess.square_name(chess.square(mv.to_square % 8, mv.from_square // 8))
            self.arm.remove_piece(capt_sq)
        self.arm.move_piece(chess.square_name(mv.from_square),
                            chess.square_name(mv.to_square))

        # ---- push ----
        san = self.board.san(mv)
        self.board.push(mv)
        self.history.append(san)

        # ---- evaluate AFTER + classify ----
        a_eval = self.engine.analyse(self.board)["evaluation"]
        print(f"   Eval  {b_eval:+d} → {a_eval:+d} cp")
        print(f"   {self._classify(b_eval, a_eval)}")
        print(f"✓ Executed: {san}")
        return True

    # ---------- suggestion ----------
    def get_hint(self):
        info = self.engine.analyse(self.board)
        print(f"🤖  Suggestion: {info['best_move']}  (eval {info['evaluation']:+d} cp)")

    # ---------- reset ----------
    def reset(self):
        self.board.reset()
        self.history.clear()
        print("♟️  Board reset")

    # ---------- tidy up ----------
    def close(self):
        self.engine.close()

# ----------------  human-vs-computer driver  ----------------
def play_vs_computer():
    """Human picks a colour, engine the other."""
    import chess.engine

    game = ChessGame()

    # choose side
    while True:
        side = input("Play as [w]hite or [b]lack ? ").lower()
        if side in ("w", "white"):
            human_white = True
            break
        if side in ("b", "black"):
            human_white = False
            break
        print("Type w or b.")

    game.show_board()

    while not game.board.is_game_over():
        if game.board.turn == chess.WHITE:
            player = "White" if human_white else "Engine"
        else:
            player = "Black" if not human_white else "Engine"
        print(f"\n{player} to move")

        # ----------  human move  ----------
        if (game.board.turn == chess.WHITE) == human_white:
            while True:
                cmd = input("Your move (e2e4 / Nf3) or 'hint' / 'quit': ").strip()
                if cmd in {"q", "quit", "exit"}:
                    print("Game aborted."); return
                if cmd == "hint":
                    game.get_hint(); continue
                if game.make_move(cmd):
                    break
        # ----------  engine move  ----------
        else:
            print("Engine thinking ...")
            info = game.engine.analyse(game.board)
            best = info["best_move"]

            # shallow human-like slip
            if BLUNDER_PCT and random.random() < BLUNDER_PCT / 100:
                legal = list(game.board.legal_moves)
                weak = random.choice(legal)
                print(f"🤖  (blunder) plays {weak} instead of {best}")
                best = weak.uci()

            game.make_move(best)

        game.show_board()

    print("\nGame over – result:", game.board.result())
    game.close()


# kick-off
if __name__ == "__main__":
    play_vs_computer()