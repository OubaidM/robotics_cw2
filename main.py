import cv2
import numpy as np
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import threading
import queue
import json
from datetime import datetime
import chess
import chess.engine
from typing import Optional, Dict, Any
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import sys
sys.path.insert(0,'/usr/local/lib/python3.11/dist-packages/Arm_Lib-0.0.5-py3.11.egg')
from Arm_Lib import Arm_Device

class ArmController:
    """Controller for the robotic arm."""
    
    def __init__(self, board_config_path="../all_squares_touch.json"):
        self.arm = Arm_Device()
        self.board = json.loads(open(board_config_path).read())
        
        # Define positions
        self.NEUTRAL_POS = [90, 150, 10, 90, 264, 180]
        self.NEUTRAL_POS_OPEN = [90, 150, 10, 90, 264, 145]
        self.RM_POS_CLOSE = [30, 56, 19, 42, 264, 180]
        self.RM_POS_OPEN = [30, 56, 19, 42, 264, 150]
        
        # Move to neutral position on initialization
        self.move_to_neutral()
    
    def move(self, joint_list, duration=1000, wait=1.0):
        """Move arm to specified joint angles."""
        self.arm.Arm_serial_servo_write6(*joint_list, duration)
        time.sleep(wait)
    
    def sq_pose(self, sq, claw):
        """Returns touch angles for square 'sq' with claw angle 'claw'."""
        rank = int(sq[1])
        base = self.board[sq][:5] + [claw]
        if 1 <= rank <= 3:
            base[1] += 3
        elif 4 <= rank <= 6:
            base[1] += 6
        # ranks 7-8: no change
        return base
    
    def move_to_neutral(self):
        """Move arm to neutral position."""
        print("🤖 Moving arm to neutral position...")
        self.move(self.NEUTRAL_POS_OPEN, 800, 1.0)
    
    def move_piece(self, src: str, dst: str):
        """Move a piece from source square to destination square."""
        print(f"🤖 Robot moving piece from {src} to {dst}...")
        
        src_close = self.sq_pose(src, 180)
        dst_close = self.sq_pose(dst, 180)
        src_open = self.sq_pose(src, 140)
        dst_open = self.sq_pose(dst, 140)
        src_open[1] += 3  # extra +3° on src_open only
        
        self.move(self.NEUTRAL_POS_OPEN, 800, 1.0)
        self.move(src_open, 2000, 2.0)
        self.move(src_close, 500, 0.5)
        self.move(self.NEUTRAL_POS, 2000, 2.0)
        self.move(dst_close, 2000, 2.0)
        self.move(dst_open, 500, 0.7)
        self.move(self.NEUTRAL_POS, 2000, 2.0)
        
        print(f"✓ Moved {src} → {dst}")
    
    def remove_piece(self, sq: str):
        """Remove a piece from the board (for captures)."""
        print(f"🤖 Robot removing piece from {sq}...")
        
        src_close = self.sq_pose(sq, 180)
        src_open = self.sq_pose(sq, 140)
        
        self.move(self.NEUTRAL_POS_OPEN, 2000, 2.0)
        self.move(src_open, 2500, 2.5)
        self.move(src_close, 500, 0.5)
        self.move(self.NEUTRAL_POS, 3000, 3.0)
        self.move(self.RM_POS_CLOSE, 2500, 2.5)
        self.move(self.RM_POS_OPEN, 700, 0.5)
        self.move(self.NEUTRAL_POS_OPEN, 2500, 2.5)
        
        print(f"✓ Removed piece from {sq}")
    
    def test_move(self):
        """Test function to verify arm is working."""
        print("🤖 Testing arm movement...")
        self.move_piece("e2", "e4")
        print("✓ Arm test completed")


@dataclass
class GameConfig:
    """Game configuration with defaults."""
    difficulty_elo: int = 1200
    human_color: str = "white"  # "white" or "black"
    camera_index: int = 0
    display_scale: float = 0.5
    stockfish_path: str = "stockfish-windows-x86-64-avx2.exe"
    calibration_file: str = "calibration_data.json"
    game_config_file: str = "game_config.json"
    
    @classmethod
    def load(cls, filename=None):
        """Load configuration from file."""
        if filename is None:
            filename = cls().game_config_file
        
        config_path = Path(filename)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        return cls()
    
    def save(self, filename=None):
        """Save configuration to file."""
        if filename is None:
            filename = self.game_config_file
        
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class ChessGameIntegration:
    """Bridge between Vision System and ChessGame/Stockfish/Robot."""
    
    def __init__(self, vision_system, difficulty_elo=1200, use_robot=False):
        """
        Args:
            vision_system: Your ChessVisionSystem instance
            difficulty_elo: Engine strength (500-2000)
            use_robot: Whether to use physical robot arm
        """
        self.vision = vision_system
        self.difficulty_elo = difficulty_elo
        self.use_robot = use_robot
        
        # Create chess board - start with standard position
        self.board = chess.Board()  # Standard starting position
        
        # Initialize board state in vision system if not already done
        if self.vision.board_state is None:
            # Create empty board state (will be populated later)
            self.vision.board_state = [
                [None for _ in range(8)] for _ in range(8)
            ]
        
        # Stockfish engine (will be initialized when needed)
        self.engine = None
        self.engine_depth = self._elo_to_depth(difficulty_elo)
        
        # Arm controller
        if use_robot:
            try:
                self.arm_controller = ArmController()
                print("✓ Robotic arm initialized")
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
        self.human_color = chess.WHITE  # Human plays white by default
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
        # Find closest Elo in the table
        closest_elo = min(ELO_TO_DEPTH.keys(), key=lambda x: abs(x - elo))
        return ELO_TO_DEPTH[closest_elo]
    
    def update_board_from_vision(self):
        """Update internal chess.Board from vision system's FEN."""
        # First check if vision system has valid board state
        if self.vision.board_state is None:
            print("⚠ Vision board state is None, using standard position")
            self.board = chess.Board()
            return
        
        try:
            fen = self.vision.get_fen()
            self.board.set_fen(fen)
            print(f"✓ Board updated from vision FEN")
        except (ValueError, AttributeError) as e:
            print(f"⚠ Error setting FEN: {e}, using standard position")
            self.board = chess.Board()
    
    def update_vision_from_board(self):
        """Update vision system from internal chess.Board."""
        if self.vision.board_state is None:
            # Initialize empty board
            self.vision.board_state = [
                [None for _ in range(8)] for _ in range(8)
            ]
        
        fen = self.board.fen()
        print(f"Updating vision with FEN: {fen}")
        
        # Parse FEN and update vision board state
        try:
            # Split FEN into components
            parts = fen.split()
            if len(parts) == 0:
                return
            
            # Parse piece placement
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
                        # Convert FEN piece to internal representation
                        if char.isupper():
                            color = 'W'
                        else:
                            color = 'B'
                        
                        piece_type = char.upper()
                        
                        if piece_type == 'N':
                            piece_code = f"{color}N"
                        else:
                            piece_code = f"{color}{piece_type}"
                        
                        self.vision.board_state[rank_idx][file_idx] = piece_code
                        file_idx += 1
            
            # Update other FEN fields if they exist
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
            
            print("✓ Vision board state updated from FEN")
            
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
    
# In ChessGameIntegration class, update initialize_engine method:

    def initialize_engine(self):
        """Initialize Stockfish engine."""
        try:
            import chess.engine
            
            # Try multiple possible paths
            possible_paths = [
                "stockfish-windows-x86-64-avx2.exe",  # Windows <-- What we are using at time of testing
                "stockfish-windows-x86-64.exe",       # Windows alternative
                "stockfish",                          # Linux/Mac
                "/usr/local/bin/stockfish",           # Mac Homebrew
                "/usr/bin/stockfish",                 # Linux system
                "stockfish/stockfish-windows-x86-64-avx2.exe",  # Subfolder
            ]
            
            engine_path = None
            for path in possible_paths:
                if Path(path).exists():
                    engine_path = path
                    break
            
            if engine_path is None:
                # Try to find stockfish in PATH
                import shutil
                path = shutil.which("stockfish")
                if path:
                    engine_path = path
                else:
                    print("✗ Stockfish not found. Please download from: https://stockfishchess.org/download/")
                    print("Place it in the same folder as this script or in your PATH.")
                    return False
            
            print(f"Initializing Stockfish from: {engine_path}")
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
            # Configure engine difficulty
            if self.difficulty_elo < 2000:
                # Set lower skill level for weaker play
                self.engine.configure({"Skill Level": max(0, (self.difficulty_elo - 500) // 100)})
            
            print("✓ Stockfish engine initialized")
            return True
        except Exception as e:
            print(f"✗ Could not initialize Stockfish: {e}")
            print("Please download Stockfish from: https://stockfishchess.org/download/")
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
            # Get all legal moves first
            legal_moves = list(self.board.legal_moves)
            print(f"Legal moves available: {len(legal_moves)}")
            
            if not legal_moves:
                print("🤖 No legal moves available!")
                return None
            
            # Try to get engine move with timeout
            try:
                result = self.engine.play(
                    self.board, 
                    chess.engine.Limit(depth=self.engine_depth, time=5.0)  # Add time limit
                )
            except chess.engine.EngineTerminatedError:
                print("🤖 Engine terminated, restarting...")
                self.engine = None
                return self.get_engine_move()  # Try again
            
            if result and hasattr(result, 'move') and result.move:
                move = result.move
                print(f"🤖 Engine raw suggestion: {move.uci()}")
                
                # Double-check the move is legal
                if move in legal_moves:
                    print(f"✓ Engine move is legal: {move.uci()}")
                    return move
                else:
                    print(f"⚠ Engine suggested illegal move: {move.uci()}")
                    print(f"⚠ Legal moves: {[m.uci() for m in legal_moves[:10]]}...")
                    # Choose first legal move as fallback
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
            # Return first legal move as fallback
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            return None
    def process_human_move(self, move_info: Dict[str, Any]) -> bool:
        """
        Process a move detected by vision system.
        Returns True if move was valid and processed.
        """
        print(f"\n=== PROCESSING HUMAN MOVE ===")
        
        # Convert vision move to chess.Move
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        
        # Handle promotion (if we detect it)
        promotion = None
        if move_info.get('promotion'):
            promotion = getattr(chess, move_info['promotion'].upper())
        
        try:
            move = chess.Move.from_uci(f"{from_sq}{to_sq}")
            if promotion:
                move = chess.Move.from_uci(f"{from_sq}{to_sq}{promotion}")
            
            # Check if move is legal
            if move in self.board.legal_moves:
                print(f"✓ Legal move: {move.uci()}")
                
                # Update internal board
                self.board.push(move)
                self.last_human_move = move
                
                # Update vision system with the move (it should already know, but sync)
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
        """
        Get engine's response to human move and return move info for vision.
        """
        print(f"\n=== ENGINE'S TURN ===")
        print(f"Current FEN: {self.board.fen()}")
        print(f"Board turn: {'White' if self.board.turn else 'Black'}")
        print(f"Human color: {'White' if self.human_color == chess.WHITE else 'Black'}")
        
        # Check if game is over
        if self.board.is_game_over():
            print("Game over! No engine response needed.")
            return None
        
        # Check if it's actually engine's turn
        is_engine_turn = (self.board.turn == chess.WHITE and self.human_color == chess.BLACK) or \
                         (self.board.turn == chess.BLACK and self.human_color == chess.WHITE)
        
        if not is_engine_turn:
            print(f"⚠ Not engine's turn! Board turn: {'White' if self.board.turn else 'Black'}")
            return None
        
        print(f"✓ It's engine's turn, getting move...")
        
        # Get engine move
        engine_move = self.get_engine_move()
        
        if not engine_move:
            print("✗ Engine returned no move")
            return None
        
        print(f"🤖 Engine suggested: {engine_move.uci()}")
        
        # Verify move is legal
        if engine_move not in self.board.legal_moves:
            print(f"⚠ Engine move {engine_move.uci()} is not legal!")
            print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
            # Fall back to first legal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                engine_move = legal_moves[0]
                print(f"🤖 Using fallback move: {engine_move.uci()}")
            else:
                print("🤖 No legal moves available!")
                return None
        
        # Get SAN notation BEFORE pushing the move
        try:
            san_notation = self.board.san(engine_move)
        except AssertionError as e:
            print(f"⚠ Error getting SAN: {e}")
            print("Using UCI notation instead")
            san_notation = engine_move.uci()
        
        # Apply engine move to board
        self.board.push(engine_move)
        
        # Create move info for vision system
        move_info = {
            'from_square': chess.square_name(engine_move.from_square),
            'to_square': chess.square_name(engine_move.to_square),
            'captured_square': None,
            'valid': True,
            'move_type': 'engine',
            'uci': engine_move.uci(),
            'san': san_notation
        }
        
        # Check if capture (need to check before the move was made)
        # We need to check the board state before the move
        temp_board = self.board.copy()
        temp_board.pop()  # Go back to before the move
        if temp_board.is_capture(engine_move):
            move_info['captured_square'] = chess.square_name(engine_move.to_square)
            captured_piece = temp_board.piece_at(engine_move.to_square)
            if captured_piece:
                move_info['captured_piece'] = captured_piece.symbol()
        
        print(f"🤖 Engine plays: {move_info['san']}")
        return move_info
    
    def execute_robot_move(self, move_info: Dict[str, Any]) -> bool:
        """
        Execute the move on robot arm.
        Blocks until the move is complete.
        """
        print(f"\n=== EXECUTING ROBOT MOVE ===")
        
        if not self.use_robot or self.arm_controller is None:
            print("⚠ Robot arm not available - simulating move")
            import time
            time.sleep(2)  # Simulate robot moving
            return True
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        captured_sq = move_info.get('captured_square')
        
        try:
            # Handle capture first (remove captured piece)
            if captured_sq:
                print(f"🤖 Removing captured piece from {captured_sq}...")
                self.arm_controller.remove_piece(captured_sq)
                print(f"✓ Capture completed")
            
            # Move the piece
            print(f"🤖 Moving piece from {from_sq} to {to_sq}...")
            self.arm_controller.move_piece(from_sq, to_sq)
            print(f"✓ Move completed")
            
            return True
            
        except Exception as e:
            print(f"✗ Robot arm error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def should_engine_move(self):
        """Check if it's engine's turn to move."""
        # If human is white and it's black's turn, engine should move
        # If human is black and it's white's turn, engine should move
        human_is_white = (self.human_color == chess.WHITE)
        board_turn_is_white = self.board.turn
        
        if human_is_white:
            # Human is white, engine is black
            # Engine should move when it's black's turn (board.turn == chess.BLACK)
            return not board_turn_is_white  # board.turn == chess.BLACK
        else:
            # Human is black, engine is white
            # Engine should move when it's white's turn (board.turn == chess.WHITE)
            return board_turn_is_white  # board.turn == chess.WHITE
    
# Update the game_loop_iteration method in ChessGameIntegration:

    def game_loop_iteration(self, vision_move_info: Optional[Dict[str, Any]] = None):
        """
        One iteration of the game loop.
        Can be called after vision detects a move, or periodically.
        """
        print(f"\n=== GAME LOOP ITERATION ===")
        print(f"Board turn: {'White' if self.board.turn else 'Black'}")
        print(f"Human color: {'White' if self.human_color == chess.WHITE else 'Black'}")
        
        # If vision detected a human move, process it
        if vision_move_info and vision_move_info.get('valid'):
            print(f"Processing human move: {vision_move_info['from_square']}→{vision_move_info['to_square']}")
            
            if self.process_human_move(vision_move_info):
                print(f"✓ Human move processed successfully")
                print(f"New board turn: {'White' if self.board.turn else 'Black'}")
                print(f"New FEN: {self.board.fen()}")
                
                # Check if game is over after human move
                if self.board.is_game_over():
                    print("Game over after human move!")
                    return None
                
                # Update vision system immediately
                self.update_vision_from_board()
                
                print(f"✓ Vision updated with human move")
                return {'type': 'human_move_processed', 'move': vision_move_info}
            else:
                print("✗ Failed to process human move")
        
        return None
    def get_current_turn(self):
        """Get whose turn it is."""
        if self.board.turn == chess.WHITE:
            return "white"
        else:
            return "black"
    
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
            print("✓ Engine closed")

class ChessVisionSystem:
    def __init__(self, model_path="robot_code/models/best_yollov11s.pt"):
        """Initialize the chess vision system."""
        self.model = YOLO(model_path)
        self.homography_matrix = None
        self.square_centers = None
        self.square_boundaries = None
        self.board_state = None
        
        # Track detected pieces per square
        self.detected_pieces_state = {}
        self.prev_detected_state = {}
        
        # ===== NEW: Complete FEN tracking =====
        self.active_color = 'w'  # 'w' or 'b'
        self.castling_rights = 'KQkq'
        self.en_passant_target = '-'
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # ===== NEW: Game history =====
        self.move_history = []
        self.position_history = []
        self.game_headers = {
            'Event': 'Robot Chess Game',
            'Site': 'Local',
            'Date': datetime.now().strftime('%Y.%m.%d'),
            'Round': '1',
            'White': 'Human',
            'Black': 'Robot',
            'Result': '*'
        }
        
        # Standard starting position
        self.initial_position = [
            ['BR', 'BN', 'BB', 'BQ', 'BK', 'BB', 'BN', 'BR'],
            ['BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP'],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            ['WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP'],
            ['WR', 'WN', 'WB', 'WQ', 'WK', 'WB', 'WN', 'WR'],
        ]

    def calibrate_board(self, image):
        """
        Interactive calibration: user clicks 4 outer corners of the board.
        Click order: top-left, top-right, bottom-right, bottom-left
        """
        print("\n=== BOARD CALIBRATION ===")
        print("Click the 4 OUTER corners of the board in this order:")
        print("1. Top-left corner")
        print("2. Top-right corner")
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("Press 'r' to reset, 'q' to quit")
        
        corners = []
        clone = image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(clone, str(len(corners)), (x+10, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(corners) > 1:
                    cv2.line(clone, corners[-2], corners[-1], (0, 255, 0), 2)
                if len(corners) == 4:
                    cv2.line(clone, corners[-1], corners[0], (0, 255, 0), 2)
                cv2.imshow("Calibration", clone)
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.imshow("Calibration", clone)
        
        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                corners = []
                clone = image.copy()
                cv2.imshow("Calibration", clone)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        
        # Calculate homography matrix
        src_pts = np.array(corners, dtype=np.float32)
        # Map to a normalized 800x800 board in transformed space
        dst_pts = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Generate 64 square centers and boundaries
        self._generate_square_grid()
        
        print("✓ Board calibrated successfully!")
        return True
    
    def _generate_square_grid(self):
        """Generate the 64 square centers and boundaries in transformed space."""
        square_size = 100  # 800/8
        self.square_centers = {}
        self.square_boundaries = {}
        
        files = 'abcdefgh'
        for rank in range(8):  # 8 to 1
            for file_idx in range(8):  # a to h
                square_name = f"{files[file_idx]}{8-rank}"
                
                # Center of square in transformed space
                center_x = file_idx * square_size + square_size / 2
                center_y = rank * square_size + square_size / 2
                self.square_centers[square_name] = (center_x, center_y)
                
                # Boundaries (left, top, right, bottom)
                self.square_boundaries[square_name] = (
                    file_idx * square_size,
                    rank * square_size,
                    (file_idx + 1) * square_size,
                    (rank + 1) * square_size
                )

    # ===== COMPLETE FEN MANAGEMENT =====
    
    def board_to_fen(self):
        """Convert current board state to complete FEN notation."""
        # Handle None board_state
        if self.board_state is None:
            print("⚠ Warning: board_state is None, returning starting position FEN")
            return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # 1. Piece placement
        fen_rows = []
        for rank in self.board_state:
            empty_count = 0
            row_str = ""
            
            for piece in rank:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    
                    color = piece[0]
                    piece_type = piece[1]
                    
                    fen_piece = piece_type.upper() if color == 'W' else piece_type.lower()
                    if piece_type == 'N':
                        fen_piece = 'N' if color == 'W' else 'n'
                    
                    row_str += fen_piece
            
            if empty_count > 0:
                row_str += str(empty_count)
            
            fen_rows.append(row_str)
        
        fen_position = '/'.join(fen_rows)
        
        # 2. Active color (default to 'w' if not set)
        active_color = getattr(self, 'active_color', 'w')
        
        # 3. Castling availability (default to 'KQkq' if not set)
        castling_rights = getattr(self, 'castling_rights', 'KQkq')
        
        # 4. En passant target square (default to '-' if not set)
        en_passant_target = getattr(self, 'en_passant_target', '-')
        
        # 5. Halfmove clock (default to 0 if not set)
        halfmove_clock = getattr(self, 'halfmove_clock', 0)
        
        # 6. Fullmove number (default to 1 if not set)
        fullmove_number = getattr(self, 'fullmove_number', 1)
        
        full_fen = f"{fen_position} {active_color} {castling_rights} {en_passant_target} {halfmove_clock} {fullmove_number}"
        
        return full_fen
    
    def load_fen(self, fen_string):
        """Load board state from a FEN string."""
        print(f"\n=== LOADING FEN ===")
        print(f"FEN: {fen_string}")
        
        parts = fen_string.strip().split()
        
        if len(parts) < 1:
            print("✗ Invalid FEN: No position data")
            return False
        
        # 1. Parse piece placement
        rows = parts[0].split('/')
        
        if len(rows) != 8:
            print(f"✗ Invalid FEN: Expected 8 ranks, got {len(rows)}")
            return False
        
        # Clear board
        self.board_state = [[None for _ in range(8)] for _ in range(8)]
        
        for rank_idx, row in enumerate(rows):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    # Convert FEN piece to internal representation
                    if char.isupper():
                        color = 'W'
                    else:
                        color = 'B'
                    
                    piece_type = char.upper()
                    
                    if piece_type == 'N':
                        piece_code = f"{color}N"
                    else:
                        piece_code = f"{color}{piece_type}"
                    
                    self.board_state[rank_idx][file_idx] = piece_code
                    file_idx += 1
        
        # 2. Parse active color
        if len(parts) > 1:
            self.active_color = parts[1]
        
        # 3. Parse castling rights
        if len(parts) > 2:
            self.castling_rights = parts[2]
        
        # 4. Parse en passant target
        if len(parts) > 3:
            self.en_passant_target = parts[3]
        
        # 5. Parse halfmove clock
        if len(parts) > 4:
            try:
                self.halfmove_clock = int(parts[4])
            except ValueError:
                self.halfmove_clock = 0
        
        # 6. Parse fullmove number
        if len(parts) > 5:
            try:
                self.fullmove_number = int(parts[5])
            except ValueError:
                self.fullmove_number = 1
        
        print(f"✓ FEN loaded successfully")
        return True
    
    def get_fen(self):
        """Get current FEN."""
        return self.board_to_fen()
    
    # ===== MOVE HISTORY TRACKING =====
    
    def add_move_to_history(self, move_info, fen_before, fen_after):
        """Record a move in history."""
        move_record = {
            'move_number': self.fullmove_number,
            'from_square': move_info['from_square'],
            'to_square': move_info['to_square'],
            'captured': move_info.get('captured_square'),
            'piece': self._get_piece_at_square(move_info['from_square'], fen_before),
            'fen_before': fen_before,
            'fen_after': fen_after,
            'timestamp': time.time(),
            'color': self.active_color,
            'move_type': move_info.get('move_type', 'unknown'),
            'uci': f"{move_info['from_square']}{move_info['to_square']}"
        }
        
        self.move_history.append(move_record)
        
        # Also record position for repetition detection
        self.position_history.append(fen_before.split()[0])
        
        print(f"📝 Move recorded: {move_record['piece']} {move_record['from_square']}→{move_record['to_square']}")
    
    def _get_piece_at_square(self, square, fen):
        """Extract piece from FEN at given square."""
        rows = fen.split()[0].split('/')
        file_idx = ord(square[0]) - ord('a')
        rank_idx = 8 - int(square[1])
        
        row = rows[rank_idx]
        col = 0
        
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                if col == file_idx:
                    return char.upper() if char.isupper() else char.lower()
                col += 1
        
        return None
    
    def get_move_history_display(self):
        """Return formatted move history for display."""
        if not self.move_history:
            return []
        
        formatted = []
        white_moves = [m for m in self.move_history if m['color'] == 'w']
        black_moves = [m for m in self.move_history if m['color'] == 'b']
        
        for i in range(max(len(white_moves), len(black_moves))):
            line = f"{i+1}."
            if i < len(white_moves):
                line += f" {white_moves[i]['uci']}"
            if i < len(black_moves):
                line += f" {black_moves[i]['uci']}"
            formatted.append(line)
        
        return formatted
    
    def export_pgn(self):
        """Export game to PGN format."""
        pgn_lines = []
        
        # Add headers
        for key, value in self.game_headers.items():
            pgn_lines.append(f'[{key} "{value}"]')
        
        pgn_lines.append('')
        
        # Add moves
        moves_display = self.get_move_history_display()
        pgn_lines.extend(moves_display)
        
        return '\n'.join(pgn_lines)
    
    def save_game(self, filename=None):
        """Save game to PGN file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"chess_game_{timestamp}.pgn"
        
        pgn_content = self.export_pgn()
        
        with open(filename, 'w') as f:
            f.write(pgn_content)
        
        print(f"✓ Game saved to {filename}")
        return filename
    
    def undo_last_move(self):
        """Revert the last move."""
        if not self.move_history:
            print("No moves to undo")
            return False
        
        last_move = self.move_history.pop()
        
        # Load the FEN from before the move
        self.load_fen(last_move['fen_before'])
        
        # Remove from position history
        if self.position_history:
            self.position_history.pop()
        
        print(f"↶ Undid move: {last_move['piece']} {last_move['from_square']}→{last_move['to_square']}")
        return True
    
    # ===== UPDATED BOARD STATE MANAGEMENT =====
    
    def update_board_state(self, move_info):
        """Update internal board state and record move."""
        if not move_info['valid']:
            print("✗ Cannot update: Invalid move")
            return False
        
        # Get FEN before move
        fen_before = self.board_to_fen()
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        captured_sq = move_info.get('captured_square')
        
        from_rank, from_file = self._square_to_indices(from_sq)
        to_rank, to_file = self._square_to_indices(to_sq)
        
        # Get the moving piece
        piece = self.board_state[from_rank][from_file]
        if not piece:
            print(f"⚠ ERROR: No piece at {from_sq} in board_state!")
            return False
        
        # Move piece
        self.board_state[from_rank][from_file] = None
        self.board_state[to_rank][to_file] = piece
        
        # Handle capture
        if captured_sq:
            cap_rank, cap_file = self._square_to_indices(captured_sq)
            captured_piece = self.board_state[cap_rank][cap_file]
            self.board_state[cap_rank][cap_file] = None
            move_info['captured_piece'] = captured_piece
            self.halfmove_clock = 0  # Reset on capture
        else:
            self.halfmove_clock += 1
        
        # Update en passant
        self._update_en_passant(from_sq, to_sq, piece)
        
        # Update castling rights
        self._update_castling_rights(from_sq, to_sq, piece)
        
        # Update active color
        self.active_color = 'b' if self.active_color == 'w' else 'w'
        
        # Increment fullmove number after black moves
        if self.active_color == 'w':
            self.fullmove_number += 1
        
        # Get FEN after move
        fen_after = self.board_to_fen()
        
        # Record move in history
        self.add_move_to_history(move_info, fen_before, fen_after)
        
        print(f"✓ Board state updated: {from_sq} → {to_sq}")
        return True
    
    def _update_en_passant(self, from_sq, to_sq, piece):
        """Update en passant target square."""
        piece_type = piece[1]
        
        if piece_type == 'P':  # Pawn move
            from_rank = int(from_sq[1])
            to_rank = int(to_sq[1])
            
            # Pawn moved two squares forward
            if abs(to_rank - from_rank) == 2:
                # Set en passant target to the square behind the pawn
                if piece[0] == 'W':  # White pawn
                    self.en_passant_target = f"{from_sq[0]}{int(from_sq[1]) + 1}"
                else:  # Black pawn
                    self.en_passant_target = f"{from_sq[0]}{int(from_sq[1]) - 1}"
            else:
                self.en_passant_target = '-'
        else:
            self.en_passant_target = '-'
    
    def _update_castling_rights(self, from_sq, to_sq, piece):
        """Update castling availability."""
        piece_type = piece[1]
        
        if piece_type == 'K':  # King moved
            if piece[0] == 'W':  # White king
                self.castling_rights = self.castling_rights.replace('K', '').replace('Q', '')
            else:  # Black king
                self.castling_rights = self.castling_rights.replace('k', '').replace('q', '')
        
        elif piece_type == 'R':  # Rook moved
            if from_sq == 'a1':  # White queenside rook
                self.castling_rights = self.castling_rights.replace('Q', '')
            elif from_sq == 'h1':  # White kingside rook
                self.castling_rights = self.castling_rights.replace('K', '')
            elif from_sq == 'a8':  # Black queenside rook
                self.castling_rights = self.castling_rights.replace('q', '')
            elif from_sq == 'h8':  # Black kingside rook
                self.castling_rights = self.castling_rights.replace('k', '')
    


    def transform_point(self, point):
        """Transform a point from original image to board grid."""
        if self.homography_matrix is None:
            return None
        
        # Add homogeneous coordinate
        point_homo = np.array([point[0], point[1], 1.0])
        
        # Transform
        transformed = self.homography_matrix @ point_homo
        
        # Convert back to Cartesian coordinates
        transformed = transformed / transformed[2]
        
        return (transformed[0], transformed[1])
    
    def detect_pieces(self, image, conf_threshold=0.25):
        """Run YOLO on raw image."""
        if self.homography_matrix is None:
            print("⚠ Cannot detect pieces: Board not calibrated")
            return []
        
        # Run YOLO detection
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Transform center to board coordinates
                transformed = self.transform_point((center_x, center_y))
                if transformed is None:
                    continue
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get class name from model
                class_name = self.model.names[class_id]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'transformed_center': transformed,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        return detections
    
    def calculate_occupancy(self, detections):
        """Calculate which squares are occupied."""
        if self.square_boundaries is None:
            print("⚠ Cannot calculate occupancy: Square grid not generated")
            return {}, {}
        
        occupancy = {}  # Map square name -> list of detections in that square
        square_detections = {}  # Map square name -> best detection in that square
        
        for detection in detections:
            center = detection['transformed_center']
            
            # Find which square this detection belongs to
            for square_name, bounds in self.square_boundaries.items():
                left, top, right, bottom = bounds
                
                if left <= center[0] <= right and top <= center[1] <= bottom:
                    if square_name not in occupancy:
                        occupancy[square_name] = []
                    occupancy[square_name].append(detection)
                    
                    # Keep only the highest confidence detection per square
                    if square_name not in square_detections:
                        square_detections[square_name] = detection
                    elif detection['confidence'] > square_detections[square_name]['confidence']:
                        square_detections[square_name] = detection
                    
                    break
        
        return occupancy, square_detections
    
    def _calculate_overlap_percentage(self, bbox1, bbox2):
        """Calculate percentage of bbox1 that overlaps with bbox2."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # Check if there is any intersection
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        # Calculate areas
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        if bbox1_area == 0:
            return 0.0
        
        return (inter_area / bbox1_area) * 100
    
    def get_occupied_squares(self, occupancy):
        """From occupancy dict, return set of squares that are occupied."""
        return set(occupancy.keys())
    
    def initialize_board_state(self, image):
        """Initialize the board with standard starting position."""
        if self.homography_matrix is None:
            print("⚠ Cannot initialize: Board not calibrated")
            return False
        
        print("\n=== BOARD INITIALIZATION ===")
        
        # Detect pieces
        detections = self.detect_pieces(image)
        
        if not detections:
            print("⚠ No pieces detected, using standard starting position")
            self.board_state = [row[:] for row in self.initial_position]
            return True
        
        # Calculate occupancy
        occupancy, square_detections = self.calculate_occupancy(detections)
        
        # Initialize board state to empty
        self.board_state = [[None for _ in range(8)] for _ in range(8)]
        
        # Map class names to piece codes
        class_to_piece = {
            'white_pawn': 'WP', 'white_rook': 'WR', 'white_knight': 'WN',
            'white_bishop': 'WB', 'white_queen': 'WQ', 'white_king': 'WK',
            'black_pawn': 'BP', 'black_rook': 'BR', 'black_knight': 'BN',
            'black_bishop': 'BB', 'black_queen': 'BQ', 'black_king': 'BK'
        }
        
        # Fill board state based on detections
        for square_name, detection in square_detections.items():
            class_name = detection['class_name']
            
            if class_name in class_to_piece:
                piece_code = class_to_piece[class_name]
                
                # Convert square name to indices
                file_idx = ord(square_name[0]) - ord('a')
                rank_idx = 8 - int(square_name[1])
                
                self.board_state[rank_idx][file_idx] = piece_code
        
        # Reset FEN tracking
        self.active_color = 'w'
        self.castling_rights = 'KQkq'
        self.en_passant_target = '-'
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        print("✓ Board state initialized from vision")
        return True
    

    def detect_move(self, image):
        """Detect what move was made."""
        if self.homography_matrix is None:
            return {'valid': False, 'error': 'Board not calibrated'}
        
        if self.board_state is None:
            return {'valid': False, 'error': 'Board state not initialized'}
        
        # Detect pieces in current frame
        detections = self.detect_pieces(image)
        
        if not detections:
            return {'valid': False, 'error': 'No pieces detected'}
        
        # Calculate occupancy
        occupancy, square_detections = self.calculate_occupancy(detections)
        current_occupied = self.get_occupied_squares(occupancy)
        
        # Initialize previous state if needed
        if not hasattr(self, 'prev_detected_state') or self.prev_detected_state is None:
            self.prev_detected_state = {}
        
        if not hasattr(self, 'detected_pieces_state') or self.detected_pieces_state is None:
            self.detected_pieces_state = {}
        
        # If we don't have previous state, initialize it and return
        if not self.detected_pieces_state:
            self.detected_pieces_state = square_detections.copy()
            return {'valid': False, 'error': 'No previous state to compare - initializing'}
        
        # Determine which squares changed
        prev_occupied = set(self.detected_pieces_state.keys())
        current_occupied = set(square_detections.keys())
        
        squares_left = prev_occupied - current_occupied
        squares_arrived = current_occupied - prev_occupied
        
        # Simple move detection: one square left and one square arrived
        if len(squares_left) == 1 and len(squares_arrived) == 1:
            from_square = list(squares_left)[0]
            to_square = list(squares_arrived)[0]
            
            # Get piece types to verify it's a valid move
            if from_square in self.detected_pieces_state:
                piece_info = self.detected_pieces_state[from_square]
                piece_class = piece_info['class_name']
            else:
                piece_class = "unknown"
            
            move_info = {
                'from_square': from_square,
                'to_square': to_square,
                'captured_square': None,
                'valid': True,
                'move_type': 'human',
                'piece': piece_class
            }
            
            # Check for capture (if to_square was occupied in previous state)
            if to_square in self.detected_pieces_state:
                move_info['captured_square'] = to_square
                captured_piece = self.detected_pieces_state[to_square]
                move_info['captured_piece'] = captured_piece['class_name']
            
            # Update detected state for next comparison
            self.prev_detected_state = self.detected_pieces_state.copy()
            self.detected_pieces_state = square_detections.copy()
            
            print(f"✓ Move detected: {from_square}→{to_square} ({piece_class})")
            return move_info
        
        # Update detected state even if no move detected
        self.prev_detected_state = self.detected_pieces_state.copy()
        self.detected_pieces_state = square_detections.copy()
        
        return {'valid': False, 'error': 'Could not detect a valid move'}


    def reset_detection_state(self):
        """Reset detection state to force re-initialization on next move."""
        print("🔄 Resetting vision detection state...")
        self.detected_pieces_state = {}
        self.prev_detected_state = {}
        print("✓ Detection state reset - next move will initialize fresh state")

    def sync_detection_state(self, frame):
        """Sync detection state with current board position."""
        print("🔄 Syncing vision detection state with current board...")
        
        # Detect pieces in current frame
        detections = self.detect_pieces(frame)
        if not detections:
            print("⚠ No detections to sync with")
            return False
        
        # Calculate occupancy
        occupancy, square_detections = self.calculate_occupancy(detections)
        
        # Update detection state
        self.detected_pieces_state = square_detections.copy()
        self.prev_detected_state = square_detections.copy()
        
        print(f"✓ Detection state synced with {len(square_detections)} detected pieces")
        return True
    
    def _square_to_indices(self, square_name):
        """Convert square name to array indices."""
        file_char = square_name[0]
        rank_char = square_name[1]
        
        file_idx = ord(file_char) - ord('a')  # a=0, b=1, ..., h=7
        rank_idx = 8 - int(rank_char)  # 1=7, 2=6, ..., 8=0
        
        return rank_idx, file_idx
    
    def visualize_detection(self, image, detections, occupancy):
        """Visualize detections and grid overlay."""
        if self.homography_matrix is None:
            return image
        
        # Create a copy of the image
        vis = image.copy()
        
        # Draw transformed grid if available
        if self.square_boundaries:
            # We need to draw grid lines in the original image space
            # For simplicity, we'll create a transformed version for visualization
            
            # Create a blank image for the transformed board
            board_img = np.zeros((800, 800, 3), dtype=np.uint8)
            
            # Draw grid lines
            for i in range(9):  # 8 squares + 1 extra line
                # Vertical lines
                cv2.line(board_img, (i*100, 0), (i*100, 800), (100, 100, 100), 2)
                # Horizontal lines
                cv2.line(board_img, (0, i*100), (800, i*100), (100, 100, 100), 2)
            
            # Draw square labels
            for rank in range(8):
                for file_idx in range(8):
                    square_name = f"{chr(ord('a') + file_idx)}{8-rank}"
                    
                    # Draw square name
                    cv2.putText(board_img, square_name, 
                               (file_idx*100 + 5, rank*100 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Transform board image back to original perspective
            inv_homography = cv2.getPerspectiveTransform(
                np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32),
                np.array(self._get_corners_from_homography(self.homography_matrix), dtype=np.float32)
            )
            
            board_warped = cv2.warpPerspective(board_img, inv_homography, 
                                              (vis.shape[1], vis.shape[0]))
            
            # Blend with original image
            vis = cv2.addWeighted(vis, 0.7, board_warped, 0.3, 0)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if 'white' in class_name else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = map(int, detection['center'])
            cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw occupancy
        if occupancy:
            for square_name, detections_list in occupancy.items():
                if detections_list:
                    # Draw square name with occupancy count
                    rank_idx, file_idx = self._square_to_indices(square_name)
                    
                    # We need to transform square center back to image coordinates
                    center_x, center_y = self.square_centers[square_name]
                    
                    # Transform back to original image
                    inv_homography = cv2.getPerspectiveTransform(
                        np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32),
                        np.array(self._get_corners_from_homography(self.homography_matrix), dtype=np.float32)
                    )
                    
                    center_homo = np.array([center_x, center_y, 1.0])
                    center_img = inv_homography @ center_homo
                    center_img = center_img / center_img[2]
                    
                    cv2.putText(vis, square_name, 
                               (int(center_img[0]) - 20, int(center_img[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis
    
    def _get_corners_from_homography(self, homography_matrix):
        """Get original corners from homography matrix."""
        # This is a simplified method - in practice you'd want to store original corners
        # For now, we'll estimate corners
        
        # Define transformed corners
        transformed_corners = np.array([
            [0, 0],
            [800, 0],
            [800, 800],
            [0, 800]
        ], dtype=np.float32)
        
        # Estimate original corners by inverting the transformation
        inv_homography = np.linalg.inv(homography_matrix)
        
        original_corners = []
        for corner in transformed_corners:
            corner_homo = np.array([corner[0], corner[1], 1.0])
            original = inv_homography @ corner_homo
            original = original / original[2]
            original_corners.append([original[0], original[1]])
        
        return original_corners


# ===== GUI APPLICATION =====

class ChessVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Vision System")
        self.root.geometry("1200x800")
        
        # Initialize vision system
        self.vision = ChessVisionSystem("best_yolov12s.pt")
        
        # Camera setup
        self.cap = None
        self.camera_active = False
        self.display_scale = 0.5
        
        # Game logic connection
        self.game_logic = None
        self.difficulty = 1200  # Default Elo
        
        # Threading
        self.camera_queue = queue.Queue(maxsize=1)
        self.running = False
        
        # Create GUI
        self.setup_gui()
        
        # Start camera thread
        self.start_camera()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera label
        self.camera_label = ttk.Label(left_panel, text="Camera Feed", font=('Arial', 12, 'bold'))
        self.camera_label.pack(pady=5)
        
        # Camera display
        self.camera_canvas = tk.Canvas(left_panel, bg='black')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        right_panel = ttk.Frame(main_container, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Game controls frame
        control_frame = ttk.LabelFrame(right_panel, text="Game Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Calibration button
        self.calibrate_btn = ttk.Button(control_frame, text="Calibrate Board", 
                                        command=self.calibrate_board)
        self.calibrate_btn.pack(fill=tk.X, pady=5)
        
        # Initialize button
        self.init_btn = ttk.Button(control_frame, text="Initialize Board", 
                                  command=self.initialize_board)
        self.init_btn.pack(fill=tk.X, pady=5)
        
        # Detect move button
        self.detect_btn = ttk.Button(control_frame, text="Detect Move", 
                                     command=self.detect_move, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=5)
        
        # Undo button
        self.undo_btn = ttk.Button(control_frame, text="Undo Last Move", 
                                   command=self.undo_move, state=tk.DISABLED)
        self.undo_btn.pack(fill=tk.X, pady=5)
        
        # Difficulty frame
        diff_frame = ttk.LabelFrame(right_panel, text="Game Difficulty", padding=10)
        diff_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Difficulty label
        ttk.Label(diff_frame, text="Robot Elo Rating:").pack(anchor=tk.W)
        
        # Difficulty slider
        self.difficulty_var = tk.IntVar(value=1200)
        self.difficulty_scale = ttk.Scale(diff_frame, from_=800, to=2800, 
                                         variable=self.difficulty_var,
                                         command=self.update_difficulty_label)
        self.difficulty_scale.pack(fill=tk.X, pady=5)
        
        # Difficulty value label
        self.diff_label = ttk.Label(diff_frame, text="1200 Elo")
        self.diff_label.pack()
        
        # FEN frame
        fen_frame = ttk.LabelFrame(right_panel, text="FEN Position", padding=10)
        fen_frame.pack(fill=tk.X, pady=(0, 10))
        
        # FEN entry
        self.fen_var = tk.StringVar()
        fen_entry = ttk.Entry(fen_frame, textvariable=self.fen_var)
        fen_entry.pack(fill=tk.X, pady=5)
        
        # FEN buttons
        fen_btn_frame = ttk.Frame(fen_frame)
        fen_btn_frame.pack(fill=tk.X)
        
        ttk.Button(fen_btn_frame, text="Load FEN", 
                  command=self.load_fen).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(fen_btn_frame, text="Copy FEN", 
                  command=self.copy_fen).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Game info frame
        info_frame = ttk.LabelFrame(right_panel, text="Game Info", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Move history
        ttk.Label(info_frame, text="Move History:").pack(anchor=tk.W)
        
        self.history_listbox = tk.Listbox(info_frame, height=10)
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Game info labels
        info_text = tk.Text(info_frame, height=6, font=('Courier', 10))
        info_text.pack(fill=tk.X, pady=5)
        info_text.insert('1.0', "Game not started")
        info_text.config(state=tk.DISABLED)
        self.info_text = info_text
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_difficulty_label(self, *args):
        """Update difficulty label when slider moves."""
        elo = self.difficulty_var.get()
        self.diff_label.config(text=f"{elo} Elo")
        
        # Update game logic difficulty if connected
        if self.game_logic:
            self.game_logic.set_difficulty(elo)
    
    def start_camera(self):
        """Start camera capture thread."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.camera_active = True
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        self.root.after(10, self.update_camera_display)
    
    def camera_loop(self):
        """Camera capture loop (runs in thread)."""
        while self.running:
            if self.camera_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Put frame in queue (discard old if full)
                    if self.camera_queue.full():
                        try:
                            self.camera_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.camera_queue.put(frame.copy())
            time.sleep(0.03)
    
    def update_camera_display(self):
        """Update camera display in GUI."""
        try:
            frame = self.camera_queue.get_nowait()
            
            # Resize for display
            h, w = frame.shape[:2]
            display_size = (int(w * self.display_scale), int(h * self.display_scale))
            display_frame = cv2.resize(frame, display_size)
            
            # Convert to RGB for tkinter
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.camera_canvas.image = photo  # Keep reference
            
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self.update_camera_display)
    
    def get_current_frame(self):
        """Get current frame from camera."""
        try:
            return self.camera_queue.get_nowait()
        except queue.Empty:
            return None
    
    def calibrate_board(self):
        """Calibrate board corners."""
        frame = self.get_current_frame()
        if frame is None:
            messagebox.showerror("Error", "No camera frame available!")
            return
        
        # Run calibration in a thread to avoid freezing GUI
        def calibrate_thread():
            self.status_var.set("Calibrating board...")
            success = self.vision.calibrate_board(frame)
            
            self.root.after(0, lambda: self.on_calibration_complete(success))
        
        threading.Thread(target=calibrate_thread, daemon=True).start()
    
    def on_calibration_complete(self, success):
        """Callback after calibration."""
        if success:
            messagebox.showinfo("Success", "Board calibrated successfully!")
            self.status_var.set("Board calibrated")
        else:
            messagebox.showerror("Error", "Calibration failed!")
            self.status_var.set("Calibration failed")
    
    def initialize_board(self):
        """Initialize board state."""
        frame = self.get_current_frame()
        if frame is None:
            messagebox.showerror("Error", "No camera frame available!")
            return
        
        def init_thread():
            self.status_var.set("Initializing board...")
            success = self.vision.initialize_board_state(frame)
            
            self.root.after(0, lambda: self.on_init_complete(success))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_init_complete(self, success):
        """Callback after initialization."""
        if success:
            self.detect_btn.config(state=tk.NORMAL)
            self.undo_btn.config(state=tk.NORMAL)
            self.update_game_info()
            messagebox.showinfo("Success", "Board initialized!")
            self.status_var.set("Board initialized")
        else:
            messagebox.showerror("Error", "Initialization failed!")
            self.status_var.set("Initialization failed")
    
    def detect_move(self):
        """Detect and process a move."""
        frame = self.get_current_frame()
        if frame is None:
            messagebox.showerror("Error", "No camera frame available!")
            return
        
        def detect_thread():
            self.status_var.set("Detecting move...")
            move_info = self.vision.detect_move(frame)
            
            self.root.after(0, lambda: self.on_move_detected(move_info))
        
        threading.Thread(target=detect_thread, daemon=True).start()
    
    def on_move_detected(self, move_info):
        """Callback after move detection."""
        if move_info['valid']:
            # Update board state
            self.vision.update_board_state(move_info)
            
            # Update GUI
            self.update_game_info()
            
            # Get FEN and send to game logic
            fen = self.vision.get_fen()
            self.fen_var.set(fen)
            
            # Send to robot logic (you'll implement this)
            if self.game_logic:
                self.game_logic.process_human_move(fen)
            
            self.status_var.set(f"Move detected: {move_info['from_square']}→{move_info['to_square']}")
        else:
            messagebox.showwarning("Invalid Move", move_info.get('error', 'Unknown error'))
            self.status_var.set("Invalid move detected")
    
    def undo_move(self):
        """Undo last move."""
        if self.vision.undo_last_move():
            self.update_game_info()
            self.fen_var.set(self.vision.get_fen())
            self.status_var.set("Last move undone")
        else:
            messagebox.showinfo("Info", "No moves to undo")
    
    def load_fen(self):
        """Load FEN from entry."""
        fen = self.fen_var.get().strip()
        if not fen:
            fen = simpledialog.askstring("Load FEN", "Enter FEN position:")
            if fen:
                self.fen_var.set(fen)
            else:
                return
        
        if self.vision.load_fen(fen):
            self.update_game_info()
            self.status_var.set("FEN loaded successfully")
        else:
            messagebox.showerror("Error", "Invalid FEN format!")
    
    def copy_fen(self):
        """Copy FEN to clipboard."""
        fen = self.vision.get_fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(fen)
        self.status_var.set("FEN copied to clipboard")
    
    def update_game_info(self):
        """Update game information display."""
        # Update history
        self.history_listbox.delete(0, tk.END)
        history = self.vision.get_move_history_display()
        for move in history[-10:]:  # Show last 10 moves
            self.history_listbox.insert(tk.END, move)
        
        # Update info text
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        fen = self.vision.get_fen()
        info = f"FEN: {fen}\n"
        info += f"Active color: {'White' if self.vision.active_color == 'w' else 'Black'}\n"
        info += f"Move number: {self.vision.fullmove_number}\n"
        info += f"Total moves: {len(self.vision.move_history)}\n"
        info += f"Castling: {self.vision.castling_rights or 'None'}"
        
        self.info_text.insert('1.0', info)
        self.info_text.config(state=tk.DISABLED)
    
    def save_game(self):
        """Save game to PGN file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if filename:
            self.vision.save_game(filename)
            self.status_var.set(f"Game saved to {filename}")
    
    def on_closing(self):
        """Cleanup on window close."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ===== GAME LOGIC INTERFACE =====

class GameLogic:
    """Interface between vision and robot logic."""
    def __init__(self, vision_system):
        self.vision = vision_system
        self.engine = None  # Chess engine (Stockfish, etc.)
        self.difficulty = 1200
        self.is_thinking = False
    
    def set_difficulty(self, elo):
        """Set engine difficulty level."""
        self.difficulty = elo
        if self.engine:
            # Configure engine with new difficulty
            pass
    
    def process_human_move(self, fen):
        """Process human move and get robot response."""
        print(f"Human move detected. FEN: {fen}")
        
        # Here you would:
        # 1. Send FEN to chess engine
        # 2. Get best move from engine at current difficulty
        # 3. Send move to robot arm
        # 4. Update vision with robot's move
        
        # For now, just print what would happen
        print(f"Engine (Elo {self.difficulty}) thinking...")
        
        # Simulate engine thinking
        import random
        time.sleep(1)  # Simulate thinking time
        
        # Get a random move (in real implementation, use chess engine)
        possible_moves = ['e2e4', 'd2d4', 'g1f3', 'c2c4']
        robot_move = random.choice(possible_moves)
        
        print(f"Engine plays: {robot_move}")
        
        # Update vision with robot's move
        self.update_vision_with_robot_move(robot_move)
    
    def update_vision_with_robot_move(self, uci_move):
        """Update vision system with robot's move."""
        from_sq = uci_move[0:2]
        to_sq = uci_move[2:4]
        
        move_info = {
            'from_square': from_sq,
            'to_square': to_sq,
            'captured_square': None,
            'valid': True,
            'move_type': 'robot'
        }
        
        self.vision.update_board_state(move_info)


# ===== MAIN APPLICATION =====

# Replace your current main() function with this:

def main():
    # Load configuration
    config = GameConfig.load()
    print(f"✓ Loaded configuration:")
    print(f"  Difficulty: {config.difficulty_elo} Elo")
    print(f"  Human color: {config.human_color}")
    print(f"  Display scale: {config.display_scale}")
    
    # Initialize camera
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Discard first few frames
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        # Try default camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open any camera")
            return
    
    # Initialize vision system
    vision = ChessVisionSystem(model_path="robot_code/models/best_yollov11s.pt")
    
    # Initialize game integration
    # Ask if user wants to use robot arm
    print("\n=== ROBOT ARM SETUP ===")
    use_robot_input = input("Use robotic arm? (y/n): ").lower().strip()
    use_robot = use_robot_input == 'y'
    
    if use_robot:
        print("✓ Using robotic arm")
    else:
        print("✓ Running in simulation mode (no physical robot)")
    
    # Initialize game integration
    game = ChessGameIntegration(vision, config.difficulty_elo, use_robot=use_robot)
    game.set_human_color(config.human_color)  # Set from config
    
    print(f"\n=== GAME SETUP ===")
    print(f"Human playing as: {config.human_color}")
    print(f"Engine playing as: {'black' if config.human_color == 'white' else 'white'}")
    print(f"Starting position FEN: {game.board.fen()}")
    
    # Check for saved calibration
    calibration_path = Path(config.calibration_file)
    calibration_loaded = False
    
    if calibration_path.exists():
        print("\n=== LOADING SAVED CALIBRATION ===")
        try:
            with open(calibration_path, 'r') as f:
                calib_data = json.load(f)
                homography_matrix = np.array(calib_data['homography_matrix'])
                vision.homography_matrix = homography_matrix
                vision._generate_square_grid()
                calibration_loaded = True
                print("✓ Calibration loaded from file")
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
    
    if not calibration_loaded:
        print("\n=== CALIBRATION REQUIRED ===")
        print("Press SPACE to capture image for calibration...")
        
        calibrated = False
        while not calibrated:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = cv2.resize(frame, 
                                      (int(frame.shape[1] * config.display_scale), 
                                       int(frame.shape[0] * config.display_scale)))
            cv2.imshow("Chess Vision - Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                cv2.destroyWindow("Chess Vision - Calibration")
                if vision.calibrate_board(frame):
                    # Save calibration
                    try:
                        calib_data = {
                            'homography_matrix': vision.homography_matrix.tolist(),
                            'timestamp': time.time()
                        }
                        with open(config.calibration_file, 'w') as f:
                            json.dump(calib_data, f, indent=2)
                        print(f"✓ Calibration saved to {config.calibration_file}")
                    except Exception as e:
                        print(f"✗ Could not save calibration: {e}")
                    
                    calibrated = True
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    
    # Initialize board state
    print("\n=== BOARD INITIALIZATION ===")
    print("Set up the board in standard starting position")
    print("and press SPACE when ready...")
    
    initialized = False
    while not initialized:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = cv2.resize(frame, 
                                  (int(frame.shape[1] * config.display_scale), 
                                   int(frame.shape[0] * config.display_scale)))
        
        # Add instructions
        cv2.putText(display_frame, "Set up board and press SPACE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Human: {config.human_color} | Engine: {config.difficulty_elo} Elo", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Board must be in standard starting position", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Chess Vision - Initialization", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            cv2.destroyWindow("Chess Vision - Initialization")
            
            # Try to initialize from vision
            vision_success = vision.initialize_board_state(frame)
            
            # Always start with standard position in chess engine
            game.board = chess.Board()  # Reset to standard position
            game.update_vision_from_board()  # Sync vision with standard position
            
            # Force vision to have standard position
            vision.load_fen(game.board.fen())
            
            # Initialize detection state
            detections = vision.detect_pieces(frame)
            occupancy, square_detections = vision.calculate_occupancy(detections)
            vision.detected_pieces_state = square_detections.copy()
            vision.prev_detected_state = square_detections.copy()
            
            # Visualize
            vis = vision.visualize_detection(frame, detections, occupancy)
            vis_display = cv2.resize(vis, 
                                    (int(vis.shape[1] * config.display_scale), 
                                     int(vis.shape[0] * config.display_scale)))
            cv2.imshow("Initial Board", vis_display)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            
            initialized = True
            print("✓ Board initialized with standard starting position")
            print(f"Starting FEN: {game.board.fen()}")
            
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    # Create a simple control window for settings -- deprecated: tk not working well with multi-threading
    # create_control_window(config, game)
    
    # Main game loop
    print("\n" + "="*50)
    print("GAME STARTED!")
    print("="*50)
    print("\nControls:")
    print("SPACE - Detect your move")
    print("d - Change difficulty")
    print("c - Toggle human color")
    print("s - Save settings")
    print("ESC - Quit")
    
    move_count = 0
    game_running = True
    engine_thinking = False
    
    while game_running:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break
        
        # Get current game status
        status = game.get_game_status()
        current_turn = status['turn']
        human_turn = 'white' if config.human_color == 'white' else 'black'
        is_human_turn = (current_turn == human_turn)
        
        # Display
        display_frame = cv2.resize(frame, 
                                  (int(frame.shape[1] * config.display_scale), 
                                   int(frame.shape[0] * config.display_scale)))
        
        # Add game info overlay
        cv2.putText(display_frame, f"Turn: {current_turn.capitalize()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Human: {config.human_color.capitalize()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Elo: {config.difficulty_elo} | Moves: {move_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if engine_thinking:
            cv2.putText(display_frame, "🤖 Engine thinking...", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif is_human_turn and not status['is_game_over']:
            cv2.putText(display_frame, "Your turn - Press SPACE after moving", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif not status['is_game_over']:
            cv2.putText(display_frame, "🤖 Engine's turn", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        
        if status['is_game_over']:
            cv2.putText(display_frame, f"GAME OVER: {status['result']}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press ESC to quit", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Chess Vision - Game", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key - ALWAYS WORK
            print("\nESC pressed - Exiting game...")
            game_running = False
            break
        
        # If it's engine's turn and not already thinking, make engine move
               # If it's engine's turn and not already thinking, make engine move
        should_engine_move = game.should_engine_move()
        
        if should_engine_move and not status['is_game_over'] and not engine_thinking:
            print(f"\n=== ENGINE'S TURN ===")
            engine_thinking = True
            
            # Show "Engine thinking" message
            for _ in range(30):  # Wait 0.5 seconds
                ret, frame = cap.read()
                if ret:
                    display_frame = cv2.resize(frame, 
                                              (int(frame.shape[1] * config.display_scale), 
                                               int(frame.shape[0] * config.display_scale)))
                    cv2.putText(display_frame, f"Turn: {'White' if game.board.turn else 'Black'}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "🤖 Engine thinking...", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                    cv2.imshow("Chess Vision - Game", display_frame)
                    cv2.waitKey(16)
            
            # Get engine move
            engine_move_info = game.get_engine_response()
            
            if engine_move_info:
                print(f"🤖 Engine plays: {engine_move_info.get('san', 'Unknown')}")
                
                # Update vision with engine move
                vision.update_board_state(engine_move_info)
                
                # Execute robot move - THIS WILL BLOCK UNTIL COMPLETE
                print("🤖 Robot executing move...")
                robot_success = game.execute_robot_move(engine_move_info)
                
                if robot_success:
                    print("✓ Robot move completed")
                    move_count += 0.5
                    
                    # Sync vision detection state
                    ret, sync_frame = cap.read()
                    if ret:
                        vision.sync_detection_state(sync_frame)
                    
                    # Show move visualization briefly
                    ret, frame = cap.read()
                    if ret:
                        detections = vision.detect_pieces(frame)
                        occupancy, _ = vision.calculate_occupancy(detections)
                        vis = vision.visualize_detection(frame, detections, occupancy)
                        vis_display = cv2.resize(vis, 
                                                (int(vis.shape[1] * config.display_scale), 
                                                 int(vis.shape[0] * config.display_scale)))
                        cv2.putText(vis_display, f"Engine: {engine_move_info['from_square']}→{engine_move_info['to_square']}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Engine Move", vis_display)
                        cv2.waitKey(1000)
                        cv2.destroyWindow("Engine Move")
                else:
                    print("✗ Robot move failed!")
            else:
                print("⚠ No engine move received")
            
            engine_thinking = False
            continue  # Go back to start of loop to refresh display
        
        if key == ord(' '):  # Space key
            if status['is_game_over']:
                print("Game is over! Press ESC to quit.")
                continue
                
            if not is_human_turn:
                print(f"⚠ Not your turn! Current turn: {current_turn}, You are: {human_turn}")
                continue
                
            # Human makes a move
            print(f"\n=== MOVE {int(move_count) + 1} ===")
            print(f"Human ({human_turn}) to move...")
            
            move_info = vision.detect_move(frame)
            
            if move_info['valid']:
                print(f"✓ Human move detected: {move_info['from_square']}→{move_info['to_square']}")
                
                # Process in game integration
                result = game.game_loop_iteration(move_info)
                
                if result:
                    print(f"✓ Engine will respond next turn")
                    move_count += 0.5  # Count half move
                    
                    # Sync vision detection state after human move
                    vision.sync_detection_state(frame)
                    
                    # Show move on screen
                    detections = vision.detect_pieces(frame)
                    occupancy, _ = vision.calculate_occupancy(detections)
                    vis = vision.visualize_detection(frame, detections, occupancy)
                    vis_display = cv2.resize(vis, 
                                            (int(vis.shape[1] * config.display_scale), 
                                             int(vis.shape[0] * config.display_scale)))
                    
                    # Add human move overlay
                    from_sq = move_info['from_square']
                    to_sq = move_info['to_square']
                    cv2.putText(vis_display, f"Human: {from_sq}→{to_sq}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Your Move", vis_display)
                    cv2.waitKey(1500)
                    cv2.destroyWindow("Your Move")
                else:
                    print("✓ Human move processed")
                    move_count += 0.5  # Count half move
            else:
                print(f"✗ Invalid move: {move_info.get('error', 'Unknown error')}")
        
        elif key == ord('d'):  # Change difficulty
            print("\n=== CHANGE DIFFICULTY ===")
            print(f"Current difficulty: {config.difficulty_elo} Elo")
            
            # Create a simple input dialog
            cv2.destroyWindow("Chess Vision - Game")
            cv2.namedWindow("Set Difficulty")
            cv2.resizeWindow("Set Difficulty", 400, 150)
            
            # Create a black image for the dialog
            dialog_img = np.zeros((150, 400, 3), dtype=np.uint8)
            cv2.putText(dialog_img, "Enter new Elo (500-2000):", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(dialog_img, f"Current: {config.difficulty_elo}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(dialog_img, "Press Enter to confirm, ESC to cancel", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow("Set Difficulty", dialog_img)
            
            # Simple input handling
            input_text = str(config.difficulty_elo)
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                
                if key2 == 13:  # Enter key
                    try:
                        new_elo = int(input_text)
                        if 500 <= new_elo <= 2000:
                            config.difficulty_elo = new_elo
                            game.set_difficulty(new_elo)
                            print(f"✓ Difficulty set to {new_elo} Elo")
                        else:
                            print("✗ Elo must be between 500 and 2000")
                    except ValueError:
                        print("✗ Invalid number")
                    break
                elif key2 == 27:  # ESC key
                    print("✗ Difficulty change cancelled")
                    break
                elif key2 == 8:  # Backspace
                    if input_text:
                        input_text = input_text[:-1]
                elif 48 <= key2 <= 57:  # Number keys
                    input_text += chr(key2)
                
                # Update display
                dialog_img = np.zeros((150, 400, 3), dtype=np.uint8)
                cv2.putText(dialog_img, "Enter new Elo (500-2000):", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(dialog_img, f"Current: {config.difficulty_elo}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(dialog_img, f"New: {input_text}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(dialog_img, "Press Enter to confirm, ESC to cancel", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.imshow("Set Difficulty", dialog_img)
            
            cv2.destroyWindow("Set Difficulty")
            pass
            
        elif key == ord('c'):  # Toggle human color
            if config.human_color == "white":
                config.human_color = "black"
                game.set_human_color("black")
                print(f"✓ Human color changed to {config.human_color}")
            else:
                config.human_color = "white"
                game.set_human_color("white")
                print(f"✓ Human color changed to {config.human_color}")
            
            # Refresh status after color change
            status = game.get_game_status()
            print(f"Game status after color change:")
            print(f"  Turn: {status['turn']}")
            print(f"  Human: {config.human_color}")
            print(f"  FEN: {status['fen']}")
    
    # Cleanup
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    game.close()
    
    # Save configuration
    config.save()
    print(f"\nConfiguration saved to {config.game_config_file}")
    print("Thanks for playing!")

def create_control_window(config, game):
    """Create a simple control window for settings."""
    
    print("\n=== GAME CONTROLS ===")
    print("Press 'd' to change difficulty")
    print("Press 'c' to change human color")
    print("Press 's' to save settings")
    print("Press ESC to quit")
    
    # We'll handle controls in the main OpenCV loop instead
    return None

if __name__ == "__main__":
    main()