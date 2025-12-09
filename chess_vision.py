import cv2
import numpy as np
from ultralytics import YOLO
import time

class ChessVisionSystem:
    def __init__(self, model_path="best_yolov12s.pt"):
        """Initialize the chess vision system."""
        self.model = YOLO(model_path)
        self.homography_matrix = None
        self.square_centers = None
        self.square_boundaries = None
        self.board_state = None
        
        # Track detected pieces per square (for capture detection)
        self.detected_pieces_state = {}  # square_name -> {'class': str, 'conf': float}
        self.prev_detected_state = {}    # Previous state for comparison
        
        # Standard chess starting position (rank 8 to rank 1)
        self.initial_position = [
            ['BR', 'BN', 'BB', 'BQ', 'BK', 'BB', 'BN', 'BR'],  # Rank 8
            ['BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP'],  # Rank 7
            [None, None, None, None, None, None, None, None],   # Rank 6
            [None, None, None, None, None, None, None, None],   # Rank 5
            [None, None, None, None, None, None, None, None],   # Rank 4
            [None, None, None, None, None, None, None, None],   # Rank 3
            ['WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP'],  # Rank 2
            ['WR', 'WN', 'WB', 'WQ', 'WK', 'WB', 'WN', 'WR'],  # Rank 1
        ]

    # def reduce_glare(self, image):
    #     """Reduce glare and improve piece detection."""
    #     # Convert to LAB color space - L channel contains brightness information
    #     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #     l_channel, a, b = cv2.split(lab)
        
    #     # Apply CLAHE to L channel only (not to color channels)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     cl = clahe.apply(l_channel)
        
    #     # Merge the CLAHE enhanced L-channel back with a and b channels
    #     limg = cv2.merge([cl, a, b])
        
    #     # Convert back to BGR
    #     enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    #     return enhanced
            
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
    
    def transform_point(self, point):
        """Transform a point from original image coords to board grid coords."""
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        return tuple(transformed[0][0])
    
    def detect_pieces(self, image, conf_threshold=0.25):
        """
        Run YOLO on raw image and return bounding boxes.
        Returns: list of dicts with 'bbox' (x1,y1,x2,y2), 'conf', 'class'
        """
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls]
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'conf': conf,
                    'class': cls_name,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                })
        
        return detections
    
    def calculate_occupancy(self, detections):
        """
        Calculate which squares are occupied based on bbox overlap.
        Returns: tuple of (occupancy_dict, detected_state_dict)
        - occupancy_dict: square_name -> list of detections
        - detected_state_dict: square_name -> {'class': str, 'conf': float} (highest conf detection)
        """
        occupancy = {sq: [] for sq in self.square_centers.keys()}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Transform bbox corners to board space
            tl = self.transform_point((x1, y1))
            tr = self.transform_point((x2, y1))
            bl = self.transform_point((x1, y2))
            br = self.transform_point((x2, y2))
            
            # Get bbox in transformed space
            trans_x1 = min(tl[0], tr[0], bl[0], br[0])
            trans_y1 = min(tl[1], tr[1], bl[1], br[1])
            trans_x2 = max(tl[0], tr[0], bl[0], br[0])
            trans_y2 = max(tl[1], tr[1], bl[1], br[1])
            
            trans_bbox = (trans_x1, trans_y1, trans_x2, trans_y2)
            det['transformed_bbox'] = trans_bbox
            
            # Check overlap with each square
            for square_name, (sq_x1, sq_y1, sq_x2, sq_y2) in self.square_boundaries.items():
                overlap = self._calculate_overlap_percentage(trans_bbox, (sq_x1, sq_y1, sq_x2, sq_y2))
                
                if overlap >= 60.0:  # 60% threshold
                    occupancy[square_name].append({
                        'detection': det,
                        'overlap': overlap
                    })
        
        # Create detected state (highest confidence per square)
        detected_state = {}
        for square_name, detections_list in occupancy.items():
            if detections_list:
                # Take highest confidence detection
                best_det = max(detections_list, key=lambda d: d['detection']['conf'])
                detected_state[square_name] = {
                    'class': best_det['detection']['class'],
                    'conf': best_det['detection']['conf']
                }
        
        return occupancy, detected_state
    
    def _calculate_overlap_percentage(self, bbox1, bbox2):
        """Calculate percentage of bbox1 that overlaps with bbox2."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        if bbox1_area == 0:
            return 0.0
        
        return (intersection_area / bbox1_area) * 100.0
    
    def get_occupied_squares(self, occupancy):
        """
        From occupancy dict, return set of squares that are occupied.
        If multiple detections on one square, keep highest confidence.
        """
        occupied = set()
        
        for square_name, detections in occupancy.items():
            if len(detections) > 0:
                occupied.add(square_name)
        
        return occupied
    
    def initialize_board_state(self, image):
        """
        Initialize the board with standard starting position.
        Verify that 32 pieces are detected in correct ranks.
        """
        print("\n=== INITIALIZING BOARD STATE ===")
        
        detections = self.detect_pieces(image)
        print(f"Detected {len(detections)} pieces")
        
        occupancy, detected_state = self.calculate_occupancy(detections)
        self.detected_pieces_state = detected_state  # Save initial state
        self.prev_detected_state = detected_state.copy()
        
        occupied_squares = self.get_occupied_squares(occupancy)
        
        # Verify pieces in starting ranks
        rank_1_2 = {f"{f}{r}" for f in 'abcdefgh' for r in '12'}
        rank_7_8 = {f"{f}{r}" for f in 'abcdefgh' for r in '78'}
        
        pieces_rank_1_2 = occupied_squares & rank_1_2
        pieces_rank_7_8 = occupied_squares & rank_7_8
        
        print(f"Pieces on ranks 1-2: {len(pieces_rank_1_2)}/16")
        print(f"Pieces on ranks 7-8: {len(pieces_rank_7_8)}/16")
        
        if len(pieces_rank_1_2) != 16 or len(pieces_rank_7_8) != 16:
            print("⚠ WARNING: Expected 16 pieces on ranks 1-2 and 16 on ranks 7-8")
            print("Proceeding anyway, but board may not be set up correctly.")
        
        # Initialize with standard position (ignore YOLO classifications)
        self.board_state = [row[:] for row in self.initial_position]
        
        print("✓ Board state initialized to starting position")
        return True
    
    def detect_move(self, image):
        """
        Detect what move was made by comparing current occupancy to expected state.
        Uses detected piece classes to identify captures.
        Returns: dict with 'from_square', 'to_square', 'captured_square' (or None)
        """
        print("\n=== DETECTING MOVE ===")
        
        detections = self.detect_pieces(image)
        occupancy, detected_state = self.calculate_occupancy(detections)
        
        # Save current state for next detection
        self.prev_detected_state = self.detected_pieces_state.copy()
        self.detected_pieces_state = detected_state
        
        current_occupied = self.get_occupied_squares(occupancy)
        
        # Expected occupied squares from board_state
        expected_occupied = set()
        for rank_idx in range(8):
            for file_idx in range(8):
                if self.board_state[rank_idx][file_idx] is not None:
                    square_name = f"{'abcdefgh'[file_idx]}{8-rank_idx}"
                    expected_occupied.add(square_name)
        
        # Find differences
        disappeared = expected_occupied - current_occupied  # Pieces that left
        appeared = current_occupied - expected_occupied      # Pieces that arrived
        
        print(f"Disappeared from: {disappeared}")
        print(f"Appeared at: {appeared}")
        
        # ===== CASE 1: Simple move (no capture) =====
        if len(disappeared) == 1 and len(appeared) == 1:
            from_sq = list(disappeared)[0]
            to_sq = list(appeared)[0]
            
            # Check if it might actually be a capture (if to_sq had a piece before)
            to_rank, to_file = self._square_to_indices(to_sq)
            had_piece_before = self.board_state[to_rank][to_file] is not None
            
            if had_piece_before:
                # Actually a capture!
                return {
                    'from_square': from_sq,
                    'to_square': to_sq,
                    'captured_square': to_sq,
                    'valid': True,
                    'move_type': 'capture'
                }
            else:
                # Simple move
                return {
                    'from_square': from_sq,
                    'to_square': to_sq,
                    'captured_square': None,
                    'valid': True,
                    'move_type': 'simple'
                }
        
        # ===== CASE 2: Capture (1 disappeared, 0 appeared) =====
        elif len(disappeared) == 1 and len(appeared) == 0:
            from_sq = list(disappeared)[0]
            
            print(f"Detected pattern: 1 disappeared, 0 appeared")
            print(f"Looking for capture destination...")
            
            # Find which square changed its detected piece class
            changed_squares = []
            
            for square in current_occupied:
                # Only check squares that were occupied before
                if square not in expected_occupied:
                    continue
                    
                # Check if this square's detected piece changed
                if square in self.prev_detected_state and square in detected_state:
                    prev_class = self.prev_detected_state[square]['class']
                    curr_class = detected_state[square]['class']
                    
                    if prev_class != curr_class:
                        changed_squares.append(square)
                        print(f"  Square {square}: {prev_class} -> {curr_class}")
            
            print(f"Squares with changed pieces: {changed_squares}")
            
            if len(changed_squares) == 1:
                # Found the capture destination!
                to_sq = changed_squares[0]
                
                # Verify this is a valid capture
                from_rank, from_file = self._square_to_indices(from_sq)
                to_rank, to_file = self._square_to_indices(to_sq)
                
                moved_piece = self.board_state[from_rank][from_file]
                captured_piece = self.board_state[to_rank][to_file]
                
                if moved_piece and captured_piece and moved_piece[0] != captured_piece[0]:
                    print(f"✓ Capture detected: {moved_piece} from {from_sq} captures {captured_piece} on {to_sq}")
                    return {
                        'from_square': from_sq,
                        'to_square': to_sq,
                        'captured_square': to_sq,
                        'valid': True,
                        'move_type': 'capture'
                    }
            
            # If no class change detected, try to find destination by process of elimination
            if len(changed_squares) == 0 or len(changed_squares) > 1:
                print("No clear class change, trying alternative detection...")
                
                # Look for squares that are occupied AND had opponent pieces
                possible_destinations = []
                
                from_rank, from_file = self._square_to_indices(from_sq)
                moved_piece = self.board_state[from_rank][from_file]
                
                if not moved_piece:
                    return {'valid': False, 'error': f'No piece at {from_sq} in board_state'}
                
                moved_color = moved_piece[0]  # 'W' or 'B'
                
                for square in current_occupied:
                    if square in expected_occupied:
                        sq_rank, sq_file = self._square_to_indices(square)
                        piece_was_there = self.board_state[sq_rank][sq_file]
                        
                        if piece_was_there and piece_was_there[0] != moved_color:
                            possible_destinations.append(square)
                
                print(f"Possible capture destinations (opponent pieces): {possible_destinations}")
                
                if len(possible_destinations) == 1:
                    to_sq = possible_destinations[0]
                    return {
                        'from_square': from_sq,
                        'to_square': to_sq,
                        'captured_square': to_sq,
                        'valid': True,
                        'move_type': 'capture'
                    }
                elif len(possible_destinations) > 1:
                    # Multiple possibilities - try to pick the closest one
                    from_file_idx = ord(from_sq[0]) - ord('a')
                    from_rank_idx = int(from_sq[1])
                    
                    def square_distance(sq1, sq2):
                        f1 = ord(sq1[0]) - ord('a')
                        r1 = int(sq1[1])
                        f2 = ord(sq2[0]) - ord('a')
                        r2 = int(sq2[1])
                        return abs(f1 - f2) + abs(r1 - r2)
                    
                    # Pick the closest square (for pawns/knights this works)
                    closest_sq = min(possible_destinations, 
                                    key=lambda sq: square_distance(from_sq, sq))
                    
                    print(f"Multiple possibilities, choosing closest: {closest_sq}")
                    return {
                        'from_square': from_sq,
                        'to_square': closest_sq,
                        'captured_square': closest_sq,
                        'valid': True,
                        'move_type': 'capture',
                        'note': 'Multiple possibilities detected'
                    }
        
        # ===== CASE 3: Other patterns (castling, promotion, etc.) =====
        else:
            print(f"⚠ Unexpected pattern: {len(disappeared)} disappeared, {len(appeared)} appeared")
            
            # Debug info
            if disappeared:
                print(f"Disappeared squares: {disappeared}")
                for sq in disappeared:
                    rank, file = self._square_to_indices(sq)
                    piece = self.board_state[rank][file]
                    print(f"  {sq}: {piece}")
            
            if appeared:
                print(f"Appeared squares: {appeared}")
                for sq in appeared:
                    if sq in detected_state:
                        print(f"  {sq}: {detected_state[sq]['class']}")
            
            return {
                'from_square': None,
                'to_square': None,
                'captured_square': None,
                'valid': False,
                'error': f'Unexpected pattern: {len(disappeared)} disappeared, {len(appeared)} appeared'
            }
        
        # Fallback
        return {
            'from_square': None,
            'to_square': None,
            'captured_square': None,
            'valid': False,
            'error': 'Could not determine move'
        }
    
    def update_board_state(self, move_info):
        """Update internal board state based on detected move."""
        if not move_info['valid']:
            print("✗ Cannot update: Invalid move")
            return False
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        captured_sq = move_info.get('captured_square')
        
        print(f"Updating board: {from_sq} -> {to_sq}")
        if captured_sq:
            print(f"  Captured on: {captured_sq}")
        
        from_rank, from_file = self._square_to_indices(from_sq)
        to_rank, to_file = self._square_to_indices(to_sq)
        
        # Get the moving piece
        piece = self.board_state[from_rank][from_file]
        if not piece:
            print(f"⚠ ERROR: No piece at {from_sq} according to board_state!")
            return False
        
        # Move piece
        self.board_state[from_rank][from_file] = None
        self.board_state[to_rank][to_file] = piece
        
        # Handle capture (might be same as to_sq for standard captures)
        if captured_sq:
            cap_rank, cap_file = self._square_to_indices(captured_sq)
            self.board_state[cap_rank][cap_file] = None
        
        print(f"✓ Board state updated")
        return True
    
    def _square_to_indices(self, square_name):
        """Convert square name (e.g., 'e4') to array indices (rank, file)."""
        file_idx = ord(square_name[0]) - ord('a')
        rank_idx = 8 - int(square_name[1])
        return rank_idx, file_idx
    
    def board_to_fen(self):
        """Convert current board state to FEN notation."""
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
                    
                    # Convert piece code to FEN notation
                    # WP -> P, BP -> p, WN -> N, BN -> n, etc.
                    color = piece[0]
                    piece_type = piece[1]
                    
                    fen_piece = piece_type.upper() if color == 'W' else piece_type.lower()
                    if piece_type == 'N':
                        fen_piece = 'N' if color == 'W' else 'n'
                    
                    row_str += fen_piece
            
            if empty_count > 0:
                row_str += str(empty_count)
            
            fen_rows.append(row_str)
        
        # Join ranks with '/'
        fen_position = '/'.join(fen_rows)
        
        # For now, just return position part of FEN
        # Full FEN includes: position, active color, castling, en passant, halfmove, fullmove
        # You can extend this based on your needs
        return fen_position + " w KQkq - 0 1"  # Placeholder for other FEN fields
    
    def visualize_detection(self, image, detections, occupancy):
        """Visualize detections and grid overlay for debugging."""
        vis_img = image.copy()
        # vis_img = self.reduce_glare(image.copy())
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.putText(vis_img, label, (int(x1), int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw grid (inverse transform square corners back to original image)
        inv_homography = np.linalg.inv(self.homography_matrix)
        
        for i in range(9):  # 9 lines for 8 squares
            # Horizontal lines
            pt1 = np.array([[[0, i * 100]]], dtype=np.float32)
            pt2 = np.array([[[800, i * 100]]], dtype=np.float32)
            
            pt1_img = cv2.perspectiveTransform(pt1, inv_homography)[0][0]
            pt2_img = cv2.perspectiveTransform(pt2, inv_homography)[0][0]
            
            cv2.line(vis_img, tuple(pt1_img.astype(int)), tuple(pt2_img.astype(int)), (255, 0, 0), 1)
            
            # Vertical lines
            pt1 = np.array([[[i * 100, 0]]], dtype=np.float32)
            pt2 = np.array([[[i * 100, 800]]], dtype=np.float32)
            
            pt1_img = cv2.perspectiveTransform(pt1, inv_homography)[0][0]
            pt2_img = cv2.perspectiveTransform(pt2, inv_homography)[0][0]
            
            cv2.line(vis_img, tuple(pt1_img.astype(int)), tuple(pt2_img.astype(int)), (255, 0, 0), 1)
        
        return vis_img


# ============= MAIN USAGE EXAMPLE =============

def main():
    # Initialize camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change to your camera index
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try for higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    DISPLAY_SCALE = 0.5  # Scale down for display purposes



    # Discard first few frames
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
     
    # Initialize vision system
    vision = ChessVisionSystem(model_path="best_yolov12s.pt")
    
    # Step 1: Calibrate board
    print("Press SPACE to capture image for calibration...")
    while True:
        time.sleep(0.5)  # Let camera adjust
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            cv2.destroyWindow("Camera Feed")
            if vision.calibrate_board(frame):
                break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Step 2: Initialize board state
    print("\nPress SPACE to capture image for board initialization...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            cv2.destroyWindow("Camera Feed")
            vision.initialize_board_state(frame)
            
            # Visualize initial detection
            detections = vision.detect_pieces(frame)
            occupancy, _ = vision.calculate_occupancy(detections)
            vis = vision.visualize_detection(frame, detections, occupancy)
            cv2.imshow("Initial Board", vis)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Step 3: Game loop
    print("\n=== GAME LOOP ===")
    print("After robot moves, press SPACE to detect opponent's move")
    print("Press 'f' to show current FEN")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ===== ADD THIS LINE =====
        DISPLAY_SCALE = 0.5  # Half size - adjust as needed (0.33, 0.25, etc.)
        # =========================
        
        # ===== REPLACE THIS LINE =====
        # OLD: cv2.imshow("Camera Feed", frame)
        # NEW:
        display_frame = cv2.resize(frame, 
                                (int(frame.shape[1] * DISPLAY_SCALE), 
                                int(frame.shape[0] * DISPLAY_SCALE)))
        cv2.imshow("Camera Feed", display_frame)
        # =============================
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Detect move - STILL USE ORIGINAL FRAME!
            move_info = vision.detect_move(frame)  # <-- Important: Use 'frame', not 'display_frame'
            
            if move_info['valid']:
                print(f"\n✓ Valid move detected!")
                print(f"  From: {move_info['from_square']}")
                print(f"  To: {move_info['to_square']}")
                if move_info['captured_square']:
                    print(f"  Captured: {move_info['captured_square']}")
                if 'move_type' in move_info:
                    print(f"  Type: {move_info['move_type']}")
                
                vision.update_board_state(move_info)
                
                # Show FEN
                fen = vision.board_to_fen()
                print(f"\nCurrent FEN: {fen}")
                
                # Visualize - STILL USE ORIGINAL FRAME for visualization
                detections = vision.detect_pieces(frame)
                occupancy, _ = vision.calculate_occupancy(detections)
                vis = vision.visualize_detection(frame, detections, occupancy)  # <-- Use 'frame'
                
                # ===== OPTIONAL: Also resize the visualization window =====
                vis_display = cv2.resize(vis, 
                                        (int(vis.shape[1] * DISPLAY_SCALE), 
                                        int(vis.shape[0] * DISPLAY_SCALE)))
                cv2.imshow("Detection", vis_display)
                # ==========================================================
                
                cv2.waitKey(2000)
                cv2.destroyWindow("Detection")
                
            else:
                print(f"\n✗ Invalid move detected!")
                print(f"  Error: {move_info.get('error', 'Unknown error')}")
                print("  Please reset the board to previous state")
        
        elif key == ord('f'):
            fen = vision.board_to_fen()
            print(f"\nCurrent FEN: {fen}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()