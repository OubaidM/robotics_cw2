# game_integration.py - Chess engine and game logic
import chess
import chess.engine
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
from arm_controller import ArmController


class ChessGameIntegration:
    """Bridge between Vision System and Chess Engine/Robot."""
    
    def __init__(self, vision_system, difficulty_elo=1200, use_robot=True):
        """
        Args:
            vision_system: ChessVisionSystem instance
            difficulty_elo: Engine strength (500-2000)
            use_robot: Whether to use physical robot arm
        """
        self.vision = vision_system
        self.difficulty_elo = difficulty_elo
        self.use_robot = use_robot
        
        # Create chess board - start with standard position
        self.board = chess.Board()
        
        # Initialize board state in vision system if not already done
        if self.vision.board_state is None:
            self.vision.board_state = [
                [None for _ in range(8)] for _ in range(8)
            ]
        
        # Stockfish engine (will be initialized when needed)
        self.engine = None
        self.engine_depth = self._elo_to_depth(difficulty_elo)
        
        # Arm controller
        if use_robot:
            from arm_controller import ArmController
            try:
                self.arm_controller = ArmController()
                print("✅ Robotic arm initialized")
                self.use_robot = True
            except Exception as e:
                print(f"⚠ Could not initialize robotic arm: {e}")
                print("⚠ Continuing without robot arm")
                self.use_robot = False
                self.arm_controller = None
        else:
            self.use_robot = False
            self.arm_controller = None
        
        # Game state
        self.human_color = chess.WHITE
        self.engine_thinking = False
        self.last_human_move = None
        
        # Sync vision with standard starting position
        self.update_vision_from_board()
    
    def _elo_to_depth(self, elo: int) -> int:
        """Convert Elo rating to Stockfish depth."""
        ELO_TO_DEPTH = {
            500: 5,  700: 6,  900: 7, 1100: 8,
            1300: 9, 1500: 10, 1700: 11, 1900: 12, 2000: 13
        }
        closest_elo = min(ELO_TO_DEPTH.keys(), key=lambda x: abs(x - elo))
        return ELO_TO_DEPTH[closest_elo]
    
    def update_board_from_vision(self):
        """Update internal chess.Board from vision system's FEN."""
        if self.vision.board_state is None:
            print("⚠ Vision board state is None, using standard position")
            self.board = chess.Board()
            return
        
        try:
            fen = self.vision.get_fen()
            self.board.set_fen(fen)
            print(f"✅ Board updated from vision FEN")
        except (ValueError, AttributeError) as e:
            print(f"⚠ Error setting FEN: {e}, using standard position")
            self.board = chess.Board()
    
    def update_vision_from_board(self):
        """Update vision system from internal chess.Board."""
        if self.vision.board_state is None:
            self.vision.board_state = [
                [None for _ in range(8)] for _ in range(8)
            ]
        
        fen = self.board.fen()
        print(f"Updating vision with FEN: {fen}")
        
        try:
            parts = fen.split()
            if len(parts) == 0:
                return
            
            rows = parts[0].split('/')
            
            if len(rows) != 8:
                print(f"✗ Invalid FEN: Expected 8 ranks, got {len(rows)}")
                return
            
            # Clear board
            self.vision.board_state = [[None for _ in range(8)] for _ in range(8)]
            
            for rank_idx, row in enumerate(rows):
                file_idx = 0
                for char in row:
                    if char.isdigit():
                        file_idx += int(char)
                    else:
                        color = 'W' if char.isupper() else 'B'
                        piece_type = char.upper()
                        
                        if piece_type == 'N':
                            piece_code = f"{color}N"
                        else:
                            piece_code = f"{color}{piece_type}"
                        
                        self.vision.board_state[rank_idx][file_idx] = piece_code
                        file_idx += 1
            
            if len(parts) > 1:
                self.vision.active_color = parts[1]
            if len(parts) > 2:
                self.vision.castling_rights = parts[2]
            if len(parts) > 3:
                self.vision.en_passant_target = parts[3]
            if len(parts) > 4:
                try:
                    self.vision.halfmove_clock = int(parts[4])
                except ValueError:
                    self.vision.halfmove_clock = 0
            if len(parts) > 5:
                try:
                    self.vision.fullmove_number = int(parts[5])
                except ValueError:
                    self.vision.fullmove_number = 1
            
            print("✅ Vision board state updated from FEN")
            
        except Exception as e:
            print(f"✗ Error updating vision from board: {e}")
    
    def set_difficulty(self, elo: int):
        """Change engine difficulty."""
        self.difficulty_elo = elo
        self.engine_depth = self._elo_to_depth(elo)
        print(f"Engine difficulty set to {elo} Elo (depth {self.engine_depth})")
    
    def set_human_color(self, color: str):
        """Set which color human is playing ('white' or 'black')."""
        self.human_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
        print(f"Human playing as {'white' if self.human_color == chess.WHITE else 'black'}")

    def initialize_engine(self):
        """Initialize Stockfish engine."""
        try:
            import chess.engine
            
            # Try multiple possible paths
            possible_paths = [
                "stockfish-windows-x86-64-avx2.exe",
                "stockfish-windows-x86-64.exe",
                "stockfish",
                "/usr/local/bin/stockfish",
                "/usr/bin/stockfish",
                "stockfish/stockfish-windows-x86-64-avx2.exe",
            ]
            
            engine_path = None
            for path in possible_paths:
                if Path(path).exists():
                    engine_path = path
                    break
            
            if engine_path is None:
                path = shutil.which("stockfish")
                if path:
                    engine_path = path
                else:
                    print("✗ Stockfish not found. Download from: https://stockfishchess.org/download/")
                    return False
            
            print(f"Initializing Stockfish from: {engine_path}")
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
            if self.difficulty_elo < 2000:
                self.engine.configure({"Skill Level": max(0, (self.difficulty_elo - 500) // 100)})
            
            print("✅ Stockfish engine initialized")
            return True
        except Exception as e:
            print(f"✗ Could not initialize Stockfish: {e}")
            return False
    
    def get_engine_move(self) -> Optional[chess.Move]:
        """Get engine's best move for current position."""
        if not self.engine:
            print("Engine not initialized, attempting to initialize...")
            if not self.initialize_engine():
                print("Failed to initialize engine!")
                return None
        
        print(f"🤖 Engine thinking (depth {self.engine_depth})...")
        
        try:
            legal_moves = list(self.board.legal_moves)
            print(f"Legal moves available: {len(legal_moves)}")
            
            if not legal_moves:
                print("🤖 No legal moves available!")
                return None
            
            try:
                result = self.engine.play(
                    self.board, 
                    chess.engine.Limit(depth=self.engine_depth, time=5.0)
                )
            except chess.engine.EngineTerminatedError:
                print("🤖 Engine terminated, restarting...")
                self.engine = None
                return self.get_engine_move()
            
            if result and hasattr(result, 'move') and result.move:
                move = result.move
                print(f"🤖 Engine raw suggestion: {move.uci()}")
                
                if move in legal_moves:
                    print(f"✅ Engine move is legal: {move.uci()}")
                    return move
                else:
                    print(f"⚠ Engine suggested illegal move: {move.uci()}")
                    fallback = legal_moves[0]
                    print(f"🤖 Using fallback: {fallback.uci()}")
                    return fallback
            else:
                print("🤖 Engine returned no move, using first legal move")
                return legal_moves[0]
                
        except Exception as e:
            print(f"🤖 Engine error: {e}")
            import traceback
            traceback.print_exc()
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            return None

    def process_human_move(self, move_info: Dict[str, Any]) -> bool:
        """Process a move detected by vision system."""
        print(f"\n=== PROCESSING HUMAN MOVE ===")
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        
        promotion = None
        if move_info.get('promotion'):
            promotion = getattr(chess, move_info['promotion'].upper())
        
        try:
            move = chess.Move.from_uci(f"{from_sq}{to_sq}")
            if promotion:
                move = chess.Move.from_uci(f"{from_sq}{to_sq}{promotion}")
            
            if move in self.board.legal_moves:
                print(f"✅ Legal move: {move.uci()}")
                
                self.board.push(move)
                self.last_human_move = move
                
                self.update_vision_from_board()
                
                return True
            else:
                print(f"✗ Illegal move: {move.uci()}")
                print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
                return False
                
        except ValueError as e:
            print(f"✗ Invalid move format: {e}")
            return False

    def get_engine_response(self) -> Optional[Dict[str, Any]]:
        """Get engine's response to human move."""
        print(f"\n=== ENGINE'S TURN ===")
        print(f"Current FEN: {self.board.fen()}")
        print(f"Board turn: {'White' if self.board.turn else 'Black'}")
        print(f"Human color: {'White' if self.human_color == chess.WHITE else 'Black'}")
        
        if self.board.is_game_over():
            print("Game over! No engine response needed.")
            return None
        
        is_engine_turn = (self.board.turn == chess.WHITE and self.human_color == chess.BLACK) or \
                         (self.board.turn == chess.BLACK and self.human_color == chess.WHITE)
        
        if not is_engine_turn:
            print(f"⚠ Not engine's turn! Board turn: {'White' if self.board.turn else 'Black'}")
            return None
        
        print(f"✅ It's engine's turn, getting move...")
        
        engine_move = self.get_engine_move()
        
        if not engine_move:
            print("✗ Engine returned no move")
            return None
        
        print(f"🤖 Engine suggested: {engine_move.uci()}")
        
        if engine_move not in self.board.legal_moves:
            print(f"⚠ Engine move {engine_move.uci()} is not legal!")
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                engine_move = legal_moves[0]
                print(f"🤖 Using fallback move: {engine_move.uci()}")
            else:
                print("🤖 No legal moves available!")
                return None
        
        try:
            san_notation = self.board.san(engine_move)
        except AssertionError as e:
            print(f"⚠ Error getting SAN: {e}")
            san_notation = engine_move.uci()
        
        self.board.push(engine_move)
        
        move_info = {
            'from_square': chess.square_name(engine_move.from_square),
            'to_square': chess.square_name(engine_move.to_square),
            'captured_square': None,
            'valid': True,
            'move_type': 'engine',
            'uci': engine_move.uci(),
            'san': san_notation
        }
        
        temp_board = self.board.copy()
        temp_board.pop()
        if temp_board.is_capture(engine_move):
            move_info['captured_square'] = chess.square_name(engine_move.to_square)
            captured_piece = temp_board.piece_at(engine_move.to_square)
            if captured_piece:
                move_info['captured_piece'] = captured_piece.symbol()
        
        print(f"🤖 Engine plays: {move_info['san']}")
        return move_info
    
    def execute_robot_move(self, move_info: Dict[str, Any]) -> bool:
        """Execute the move on robot arm."""
        print(f"\n=== EXECUTING ROBOT MOVE ===")
        
        # if not self.use_robot is None:
        #     print("⚠ Robot arm not available - simulating move")
        #     import time
        #     time.sleep(2)
        #     return True
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        captured_sq = move_info.get('captured_square')
        
        try:
            if captured_sq:
                print(f"🤖 Removing captured piece from {captured_sq}...")
                self.arm_controller.remove_piece(captured_sq)
                print(f"✅ Capture completed")
            
            print(f"🤖 Moving piece from {from_sq} to {to_sq}...")
            self.arm_controller.move_piece(from_sq, to_sq)
            print(f"✅ Move completed")
            
            return True
            
        except Exception as e:
            print(f"✗ Robot arm error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def should_engine_move(self):
        """Check if it's engine's turn to move."""
        human_is_white = (self.human_color == chess.WHITE)
        board_turn_is_white = self.board.turn
        
        if human_is_white:
            return not board_turn_is_white
        else:
            return board_turn_is_white

    def game_loop_iteration(self, vision_move_info: Optional[Dict[str, Any]] = None):
        """One iteration of the game loop."""
        print(f"\n=== GAME LOOP ITERATION ===")
        print(f"Board turn: {'White' if self.board.turn else 'Black'}")
        print(f"Human color: {'White' if self.human_color == chess.WHITE else 'Black'}")
        
        if vision_move_info and vision_move_info.get('valid'):
            print(f"Processing human move: {vision_move_info['from_square']}->{vision_move_info['to_square']}")
            
            if self.process_human_move(vision_move_info):
                print(f"✅ Human move processed successfully")
                print(f"New board turn: {'White' if self.board.turn else 'Black'}")
                print(f"New FEN: {self.board.fen()}")
                
                if self.board.is_game_over():
                    print("Game over after human move!")
                    return None
                
                self.update_vision_from_board()
                
                print(f"✅ Vision updated with human move")
                return {'type': 'human_move_processed', 'move': vision_move_info}
            else:
                print("✗ Failed to process human move")
        
        return None

    def get_current_turn(self):
        """Get whose turn it is."""
        return "white" if self.board.turn == chess.WHITE else "black"
    
    def is_human_turn(self):
        """Check if it's human's turn."""
        return (self.board.turn == chess.WHITE and self.human_color == chess.WHITE) or \
               (self.board.turn == chess.BLACK and self.human_color == chess.BLACK)
    
    def get_game_status(self) -> Dict[str, Any]:
        """Get current game status."""
        return {
            'fen': self.board.fen(),
            'turn': 'white' if self.board.turn else 'black',
            'is_game_over': self.board.is_game_over(),
            'result': self.board.result() if self.board.is_game_over() else None,
            'fullmove_number': self.board.fullmove_number,
            'legal_moves': [m.uci() for m in self.board.legal_moves],
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_fifty_moves': self.board.can_claim_fifty_moves(),
            'is_repetition': self.board.can_claim_threefold_repetition()
        }
    
    def close(self):
        """Cleanup."""
        if self.engine:
            self.engine.quit()
            print("✅ Engine closed")