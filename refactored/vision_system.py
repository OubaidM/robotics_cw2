# vision_system.py - Chess vision and detection
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime


class ChessVisionSystem:
    def __init__(self, model_path="robot_code\models\best_yolov11s.pt"):
        """Initialize the chess vision system."""
        self.model = YOLO(model_path)
        self.homography_matrix = None
        self.square_centers = None
        self.square_boundaries = None
        self.board_state = None
        
        # Track detected pieces per square
        self.detected_pieces_state = {}
        self.prev_detected_state = {}
        
        # Complete FEN tracking
        self.active_color = 'w'
        self.castling_rights = 'KQkq'
        self.en_passant_target = '-'
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # Game history
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
        """Interactive calibration: user clicks 4 outer corners of the board."""
        print("\n=== BOARD CALIBRATION ===")
        print("Click the 4 OUTER corners of the board in this order:")
        print("1. Top-left corner")
        print("2. Top-right corner")
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("Press 'r' to reset, 'q' to quit")

        corners = []

        # ---- SCALE FOR SMALL SCREEN ----
        # You can adjust this number depending on Pi screen resolution
        display_scale = 0.5  # 0.5 = half size, try 0.6 or 0.4 if needed

        small = cv2.resize(image, (0, 0), fx=display_scale, fy=display_scale)
        clone = small.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                # Convert from scaled coords -> original coords
                orig_x = int(x / display_scale)
                orig_y = int(y / display_scale)

                corners.append((orig_x, orig_y))

                # Draw marks on the small image (scaled)
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(clone, str(len(corners)), (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if len(corners) > 1:
                    # Draw lines on scaled preview
                    prev_scaled = (int(corners[-2][0] * display_scale),
                                int(corners[-2][1] * display_scale))
                    curr_scaled = (x, y)
                    cv2.line(clone, prev_scaled, curr_scaled, (0, 255, 0), 2)

                if len(corners) == 4:
                    first_scaled = (int(corners[0][0] * display_scale),
                                    int(corners[0][1] * display_scale))
                    cv2.line(clone, (x, y), first_scaled, (0, 255, 0), 2)

                cv2.imshow("Calibration", clone)

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.imshow("Calibration", clone)

        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                corners = []
                clone = small.copy()
                cv2.imshow("Calibration", clone)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()

        # ---- USE THE ORIGINAL RESOLUTION POINTS FOR HOMOGRAPHY ----
        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        self._generate_square_grid()

        print("? Board calibrated successfully!")
        return True

    
    def _generate_square_grid(self):
        """Generate the 64 square centers and boundaries in transformed space."""
        square_size = 100  # 800/8
        self.square_centers = {}
        self.square_boundaries = {}
        
        files = 'abcdefgh'
        for rank in range(8):
            for file_idx in range(8):
                square_name = f"{files[file_idx]}{8-rank}"
                
                center_x = file_idx * square_size + square_size / 2
                center_y = rank * square_size + square_size / 2
                self.square_centers[square_name] = (center_x, center_y)
                
                self.square_boundaries[square_name] = (
                    file_idx * square_size,
                    rank * square_size,
                    (file_idx + 1) * square_size,
                    (rank + 1) * square_size
                )

    def board_to_fen(self):
        """Convert current board state to complete FEN notation."""
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
        
        active_color = getattr(self, 'active_color', 'w')
        castling_rights = getattr(self, 'castling_rights', 'KQkq')
        en_passant_target = getattr(self, 'en_passant_target', '-')
        halfmove_clock = getattr(self, 'halfmove_clock', 0)
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
        
        # Parse piece placement
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
                    color = 'W' if char.isupper() else 'B'
                    piece_type = char.upper()
                    
                    if piece_type == 'N':
                        piece_code = f"{color}N"
                    else:
                        piece_code = f"{color}{piece_type}"
                    
                    self.board_state[rank_idx][file_idx] = piece_code
                    file_idx += 1
        
        if len(parts) > 1:
            self.active_color = parts[1]
        if len(parts) > 2:
            self.castling_rights = parts[2]
        if len(parts) > 3:
            self.en_passant_target = parts[3]
        if len(parts) > 4:
            try:
                self.halfmove_clock = int(parts[4])
            except ValueError:
                self.halfmove_clock = 0
        if len(parts) > 5:
            try:
                self.fullmove_number = int(parts[5])
            except ValueError:
                self.fullmove_number = 1
        
        print(f"✅ FEN loaded successfully")
        return True
    
    def get_fen(self):
        """Get current FEN."""
        return self.board_to_fen()
    
    def transform_point(self, point):
        """Transform a point from original image to board grid."""
        if self.homography_matrix is None:
            return None
        
        point_homo = np.array([point[0], point[1], 1.0])
        transformed = self.homography_matrix @ point_homo
        transformed = transformed / transformed[2]
        
        return (transformed[0], transformed[1])
    
    def detect_pieces(self, image, conf_threshold=0.25):
        """Run YOLO on raw image."""
        if self.homography_matrix is None:
            print("⚠ Cannot detect pieces: Board not calibrated")
            return []
        
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                transformed = self.transform_point((center_x, center_y))
                if transformed is None:
                    continue
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
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
        
        occupancy = {}
        square_detections = {}
        
        for detection in detections:
            center = detection['transformed_center']
            
            for square_name, bounds in self.square_boundaries.items():
                left, top, right, bottom = bounds
                
                if left <= center[0] <= right and top <= center[1] <= bottom:
                    if square_name not in occupancy:
                        occupancy[square_name] = []
                    occupancy[square_name].append(detection)
                    
                    if square_name not in square_detections:
                        square_detections[square_name] = detection
                    elif detection['confidence'] > square_detections[square_name]['confidence']:
                        square_detections[square_name] = detection
                    
                    break
        
        return occupancy, square_detections
    
    def get_occupied_squares(self, occupancy):
        """From occupancy dict, return set of squares that are occupied."""
        return set(occupancy.keys())
    
    def initialize_board_state(self, image):
        """Initialize the board with standard starting position."""
        if self.homography_matrix is None:
            print("⚠ Cannot initialize: Board not calibrated")
            return False
        
        print("\n=== BOARD INITIALIZATION ===")
        
        detections = self.detect_pieces(image)
        
        if not detections:
            print("⚠ No pieces detected, using standard starting position")
            self.board_state = [row[:] for row in self.initial_position]
            return True
        
        occupancy, square_detections = self.calculate_occupancy(detections)
        
        self.board_state = [[None for _ in range(8)] for _ in range(8)]
        
        class_to_piece = {
            'white_pawn': 'WP', 'white_rook': 'WR', 'white_knight': 'WN',
            'white_bishop': 'WB', 'white_queen': 'WQ', 'white_king': 'WK',
            'black_pawn': 'BP', 'black_rook': 'BR', 'black_knight': 'BN',
            'black_bishop': 'BB', 'black_queen': 'BQ', 'black_king': 'BK'
        }
        
        for square_name, detection in square_detections.items():
            class_name = detection['class_name']
            
            if class_name in class_to_piece:
                piece_code = class_to_piece[class_name]
                
                file_idx = ord(square_name[0]) - ord('a')
                rank_idx = 8 - int(square_name[1])
                
                self.board_state[rank_idx][file_idx] = piece_code
        
        self.active_color = 'w'
        self.castling_rights = 'KQkq'
        self.en_passant_target = '-'
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        print("✅ Board state initialized from vision")
        return True

    def detect_move(self, image):
        """
        Detect a move using authoritative board_state + YOLO occupancy.
        Includes:
        - simple moves
        - captures via YOLO class change
        - fallback: if exactly one legal capture square is occupied, use that
        - safe dictionary access (no KeyError)
        """

        print("\n=== DETECTING MOVE ===")

        # --- 0) Save previous YOLO state before processing ---
        prev_yolo = self.prev_detected_state.copy() if hasattr(self, "prev_detected_state") else {}

        # --- 1) Run YOLO detection ---
        detections = self.detect_pieces(image)
        occupancy, detected_state = self.calculate_occupancy(detections)
        current_occupied = self.get_occupied_squares(occupancy)

        # --- 2) Build expected occupied squares from authoritative board_state ---
        expected_occupied = set()
        for r in range(8):
            for f in range(8):
                if self.board_state[r][f] is not None:
                    expected_occupied.add(f"{'abcdefgh'[f]}{8-r}")

        print("Expected occupied:", expected_occupied)
        print("Current occupied :", current_occupied)

        disappeared = expected_occupied - current_occupied
        appeared    = current_occupied - expected_occupied

        print("Disappeared:", disappeared)
        print("Appeared:", appeared)

        # Utility
        def idx(sq): return self._square_to_indices(sq)

        # --- LEGAL TARGET GENERATOR ---
        def get_legal_targets(piece, from_sq):
            f_idx = ord(from_sq[0]) - ord('a')
            r_num = int(from_sq[1])
            color = piece[0]
            ptype = piece[1]

            def inside(f, r): return 0 <= f < 8 and 1 <= r <= 8

            targets = []

            # Knight
            if ptype == 'N':
                for df, dr in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
                    nf, nr = f_idx + df, r_num + dr
                    if inside(nf, nr):
                        targets.append(chr(nf + ord('a')) + str(nr))

            # Pawn captures
            elif ptype == 'P':
                step = 1 if color == 'W' else -1
                for df in (-1, 1):
                    nf, nr = f_idx + df, r_num + step
                    if inside(nf, nr):
                        targets.append(chr(nf + ord('a')) + str(nr))

            # Sliding pieces
            elif ptype in ('B','R','Q'):
                rays = []
                if ptype in ('B','Q'): rays += [(1,1),(1,-1),(-1,1),(-1,-1)]
                if ptype in ('R','Q'): rays += [(1,0),(-1,0),(0,1),(0,-1)]

                for df, dr in rays:
                    nf, nr = f_idx + df, r_num + dr
                    while inside(nf, nr):
                        targets.append(chr(nf + ord('a')) + str(nr))

                        # stop ray if piece blocks path
                        bi, bj = 8 - nr, nf
                        if self.board_state[bi][bj] is not None:
                            break
                        nf += df
                        nr += dr

            # King moves
            elif ptype == 'K':
                for df, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nf, nr = f_idx + df, r_num + dr
                    if inside(nf, nr):
                        targets.append(chr(nf + ord('a')) + str(nr))

            return targets

        # --- CASE 1: Simple move ---
        if len(disappeared) == 1 and len(appeared) == 1:
            from_sq = list(disappeared)[0]
            to_sq   = list(appeared)[0]

            ti, tj = idx(to_sq)
            capture = self.board_state[ti][tj] is not None

            # Sync YOLO
            self.prev_detected_state = detected_state.copy()
            self.detected_pieces_state = detected_state.copy()

            return {
                "valid": True,
                "from_square": from_sq,
                "to_square": to_sq,
                "captured_square": to_sq if capture else None,
                "move_type": "capture" if capture else "simple"
            }

        # --- CASE 2: Capture (1 disappeared, 0 appeared) ---
        if len(disappeared) == 1 and len(appeared) == 0:
            from_sq = list(disappeared)[0]
            print("Pattern: 1 disappeared, 0 appeared → potential capture")

            fr, fc = idx(from_sq)
            piece = self.board_state[fr][fc]

            if not piece:
                return {"valid": False, "error": f"No piece at {from_sq} in board_state"}

            legal_targets = get_legal_targets(piece, from_sq)
            print("Legal capture targets:", legal_targets)

            changed = []

            # --- YOLO CLASS CHANGE (with safe access) ---
            for sq in legal_targets:
                prev = prev_yolo.get(sq)
                curr = detected_state.get(sq)

                # Case 1: Both squares have YOLO data
                if prev and curr:
                    prev_class = prev.get("class")
                    curr_class = curr.get("class")
                    if prev_class != curr_class:
                        print(f"Class changed at {sq}: {prev_class} -> {curr_class}")
                        changed.append(sq)

                # Case 2: Previously had piece, now empty
                elif prev and not curr:
                    prev_class = prev.get("class")
                    print(f"{sq} lost YOLO class (prev={prev_class})")
                    changed.append(sq)

            # --- If exactly ONE YOLO-change → capture square ---
            if len(changed) == 1:
                to_sq = changed[0]
                self.prev_detected_state = detected_state.copy()
                self.detected_pieces_state = detected_state.copy()

                return {
                    "valid": True,
                    "from_square": from_sq,
                    "to_square": to_sq,
                    "captured_square": to_sq,
                    "move_type": "capture"
                }

            # --- FALLBACK: Only one legal target currently occupied ---
            occupied_targets = [sq for sq in legal_targets if sq in current_occupied]

            if len(occupied_targets) == 1:
                to_sq = occupied_targets[0]
                print(f"Fallback: Only one legal target occupied → {to_sq}")

                self.prev_detected_state = detected_state.copy()
                self.detected_pieces_state = detected_state.copy()

                return {
                    "valid": True,
                    "from_square": from_sq,
                    "to_square": to_sq,
                    "captured_square": to_sq,
                    "move_type": "capture"
                }

            # --- STILL AMBIGUOUS ---
            print("Could not determine capture square by YOLO or fallback.")
            self.prev_detected_state = detected_state.copy()
            self.detected_pieces_state = detected_state.copy()

            return {"valid": False, "error": "Capture detected but destination unclear"}

        # --- UNKNOWN PATTERN ---
        print(f"⚠ Unexpected pattern: {len(disappeared)} disappeared, {len(appeared)} appeared")

        self.prev_detected_state = detected_state.copy()
        self.detected_pieces_state = detected_state.copy()

        return {
            "valid": False,
            "error": f"Unexpected pattern: {len(disappeared)} disappeared, {len(appeared)} appeared"
        }


    def reset_detection_state(self):
        """Reset detection state to force re-initialization on next move."""
        print("🔄 Resetting vision detection state...")
        self.detected_pieces_state = {}
        self.prev_detected_state = {}
        print("✅ Detection state reset")

    def sync_detection_state(self, frame):
        """Sync detection state with current board position - IMPROVED VERSION."""
        print("🔄 Syncing vision detection state with current board...")
        
        # Multiple detection attempts for reliability
        all_detections = []
        for attempt in range(3):
            detections = self.detect_pieces(frame)
            if detections:
                all_detections.extend(detections)
            time.sleep(0.1)
        
        if not all_detections:
            print("⚠ No detections to sync with")
            return False
        
        # Use most confident detections
        occupancy, square_detections = self.calculate_occupancy(all_detections)
        
        # Update detection state
        self.detected_pieces_state = square_detections.copy()
        self.prev_detected_state = square_detections.copy()
        
        print(f"✅ Detection state synced with {len(square_detections)} detected pieces")
        return True
    
    def _square_to_indices(self, square_name):
        """Convert square name to array indices."""
        file_char = square_name[0]
        rank_char = square_name[1]
        
        file_idx = ord(file_char) - ord('a')
        rank_idx = 8 - int(rank_char)
        
        return rank_idx, file_idx
    
    def update_board_state(self, move_info):
        """Update internal board state and record move."""
        if not move_info['valid']:
            print("✗ Cannot update: Invalid move")
            return False
        
        fen_before = self.board_to_fen()
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        captured_sq = move_info.get('captured_square')
        
        from_rank, from_file = self._square_to_indices(from_sq)
        to_rank, to_file = self._square_to_indices(to_sq)
        
        piece = self.board_state[from_rank][from_file]
        if not piece:
            print(f"⚠ ERROR: No piece at {from_sq} in board_state!")
            return False
        
        self.board_state[from_rank][from_file] = None
        self.board_state[to_rank][to_file] = piece
        
        if captured_sq:
            cap_rank, cap_file = self._square_to_indices(captured_sq)
            captured_piece = self.board_state[cap_rank][cap_file]
            self.board_state[cap_rank][cap_file] = None
            move_info['captured_piece'] = captured_piece
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        self._update_en_passant(from_sq, to_sq, piece)
        self._update_castling_rights(from_sq, to_sq, piece)
        
        self.active_color = 'b' if self.active_color == 'w' else 'w'
        
        if self.active_color == 'w':
            self.fullmove_number += 1
        
        fen_after = self.board_to_fen()
        
        print(f"✅ Board state updated: {from_sq} → {to_sq}")
        return True
    
    def _update_en_passant(self, from_sq, to_sq, piece):
        """Update en passant target square."""
        piece_type = piece[1]
        
        if piece_type == 'P':
            from_rank = int(from_sq[1])
            to_rank = int(to_sq[1])
            
            if abs(to_rank - from_rank) == 2:
                if piece[0] == 'W':
                    self.en_passant_target = f"{from_sq[0]}{int(from_sq[1]) + 1}"
                else:
                    self.en_passant_target = f"{from_sq[0]}{int(from_sq[1]) - 1}"
            else:
                self.en_passant_target = '-'
        else:
            self.en_passant_target = '-'
    
    def _update_castling_rights(self, from_sq, to_sq, piece):
        """Update castling availability."""
        piece_type = piece[1]
        
        if piece_type == 'K':
            if piece[0] == 'W':
                self.castling_rights = self.castling_rights.replace('K', '').replace('Q', '')
            else:
                self.castling_rights = self.castling_rights.replace('k', '').replace('q', '')
        
        elif piece_type == 'R':
            if from_sq == 'a1':
                self.castling_rights = self.castling_rights.replace('Q', '')
            elif from_sq == 'h1':
                self.castling_rights = self.castling_rights.replace('K', '')
            elif from_sq == 'a8':
                self.castling_rights = self.castling_rights.replace('q', '')
            elif from_sq == 'h8':
                self.castling_rights = self.castling_rights.replace('k', '')
    
    def visualize_detection(self, image, detections, occupancy):
        """Visualize detections and grid overlay."""
        if self.homography_matrix is None:
            return image
        
        vis = image.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = (0, 255, 0) if 'white' in class_name else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            center_x, center_y = map(int, detection['center'])
            cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return vis