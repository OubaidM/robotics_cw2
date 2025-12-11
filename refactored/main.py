# main.py - Entry point and main game loop
import cv2
import numpy as np
import time
from pathlib import Path

from config import GameConfig
from vision_system import ChessVisionSystem
from game_integration import ChessGameIntegration

def main():
    # Load configuration
    config = GameConfig.load()
    print(f"✅ Loaded configuration:")
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
        cap = cv2.VideoCapture(config.camera_index)
        if not cap.isOpened():
            print("Error: Could not open any camera")
            return
    
    # Initialize vision system
    vision = ChessVisionSystem(model_path="robot_code/models/best_yollov11s.pt")

    
    # Initialize game integration
    # print("\n=== ROBOT ARM SETUP ===")
    # use_robot_input = input("Use robotic arm? (y/n): ").lower().strip()
    # use_robot = use_robot_input == 'y'
    
    # if use_robot:
    #     print("✅ Using robotic arm")
    # else:
    #     print("✅ Running in simulation mode (no physical robot)")

    use_robot=True
    
    game = ChessGameIntegration(vision, config.difficulty_elo, use_robot=True)
    game.set_human_color(config.human_color)
    
    print(f"\n=== GAME SETUP ===")
    print(f"Human playing as: {config.human_color}")
    print(f"Engine playing as: {'black' if config.human_color == 'white' else 'white'}")
    print(f"Starting position FEN: {game.board.fen()}")
    
    # Load or perform calibration
    if not load_or_calibrate(vision, config, cap):
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Initialize board state
    if not initialize_board(vision, game, config, cap):
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Run main game loop
    run_game_loop(vision, game, config, cap)
    
    # Cleanup
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    game.close()
    
    config.save()
    print(f"\nConfiguration saved to {config.game_config_file}")
    print("Thanks for playing!")


def load_or_calibrate(vision, config, cap):
    """Load saved calibration or perform new calibration."""
    calibration_path = Path(config.calibration_file)
    
    if calibration_path.exists():
        print("\n=== LOADING SAVED CALIBRATION ===")
        try:
            import json
            with open(calibration_path, 'r') as f:
                calib_data = json.load(f)
                vision.homography_matrix = np.array(calib_data['homography_matrix'])
                vision._generate_square_grid()
                print("✅ Calibration loaded from file")
                return True
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
    
    print("\n=== CALIBRATION REQUIRED ===")
    print("Press SPACE to capture image for calibration...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        
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
                    import json
                    calib_data = {
                        'homography_matrix': vision.homography_matrix.tolist(),
                        'timestamp': time.time()
                    }
                    with open(config.calibration_file, 'w') as f:
                        json.dump(calib_data, f, indent=2)
                    print(f"✅ Calibration saved to {config.calibration_file}")
                except Exception as e:
                    print(f"✗ Could not save calibration: {e}")
                return True
        elif key == ord('q'):
            return False
    
    return False


def initialize_board(vision, game, config, cap):
    """Initialize board state with standard starting position."""
    import chess
    
    print("\n=== BOARD INITIALIZATION ===")
    print("Set up the board in standard starting position")
    print("and press SPACE when ready...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        
        display_frame = cv2.resize(frame, 
                                  (int(frame.shape[1] * config.display_scale), 
                                   int(frame.shape[0] * config.display_scale)))
        
        cv2.putText(display_frame, "Set up board and press SPACE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Human: {config.human_color} | Engine: {config.difficulty_elo} Elo", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Chess Vision - Initialization", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            cv2.destroyWindow("Chess Vision - Initialization")
            
            # Initialize with standard position
            game.board = chess.Board()
            game.update_vision_from_board()
            vision.load_fen(game.board.fen())
            print("VISION START BOARD:")
            for r in vision.board_state:
                print(r)
            
            # Initialize detection state
            detections = vision.detect_pieces(frame)
            occupancy, square_detections = vision.calculate_occupancy(detections)
            vision.detected_pieces_state = square_detections.copy()
            vision.prev_detected_state = square_detections.copy()
            
            # Show visualization
            vis = vision.visualize_detection(frame, detections, occupancy)
            vis_display = cv2.resize(vis, 
                                    (int(vis.shape[1] * config.display_scale), 
                                     int(vis.shape[0] * config.display_scale)))
            cv2.imshow("Initial Board", vis_display)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            
            print("✅ Board initialized with standard starting position")
            print(f"Starting FEN: {game.board.fen()}")
            return True
            
        elif key == ord('q'):
            return False
    
    return False


def run_game_loop(vision, game, config, cap):
    """Main game loop."""
    print("\n" + "="*50)
    print("GAME STARTED!")
    print("="*50)
    print("\nControls:")
    print("SPACE - Detect your move")
    print("d - Change difficulty")
    print("c - Toggle human color")
    print("ESC - Quit")
    
    move_count = 0
    game_running = True
    engine_thinking = False
    waiting_for_human = False
    
    while game_running:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break
        
        status = game.get_game_status()
        current_turn = status['turn']
        human_turn = 'white' if config.human_color == 'white' else 'black'
        is_human_turn = (current_turn == human_turn)
        
        # Display frame with overlays
        display_frame = render_game_display(frame, config, status, human_turn, 
                                           engine_thinking, move_count)
        cv2.imshow("Chess Vision - Game", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nESC pressed - Exiting game...")
            break
        
        # Handle engine's turn
        should_engine_move = game.should_engine_move()
        
        if should_engine_move and not status['is_game_over'] and not engine_thinking:
            engine_thinking = True
            waiting_for_human = False
            
            # Show thinking message
            show_engine_thinking(cap, config, game)
            
            # Get and execute engine move
            engine_move_info = game.get_engine_response()
            
            if engine_move_info:
                print(f"🤖 Engine plays: {engine_move_info.get('san', 'Unknown')}")
                
                
                # Execute robot move (blocks until complete)
                print("🤖 Robot executing move...")
                robot_success = game.execute_robot_move(engine_move_info)
                
                if robot_success:
                    print("✅ Robot move completed")
                    move_count += 0.5
                    
                    # CRITICAL: Wait for robot to settle and re-sync vision
                    print("⏳ Waiting for board to settle...")
                    time.sleep(2.0)

                    for _ in range(10):  
                        cap.read()
                        time.sleep(0.03)
                    

                    # Re-read a frame after robot finishes moving
                    ret, frame2 = cap.read()
                    if ret:
                        print("🤖 Detecting bot move via vision...")
                        bot_vision_move = vision.detect_move(frame2)

                        if bot_vision_move["valid"]:
                            print("🤖 Bot move confirmed:", bot_vision_move["from_square"], "to", bot_vision_move["to_square"])
                            vision.update_board_state(bot_vision_move)
                        else:
                            print("⚠ Could not visually confirm bot move:", bot_vision_move["error"])

                    
                    # Show move visualization
                    show_move_visualization(cap, config, vision, engine_move_info, "Engine")
                    
                    waiting_for_human = True
                else:
                    print("✗ Robot move failed!")
            else:
                print("⚠ No engine move received")
            
            engine_thinking = False
            continue
        
        # Handle human move detection
        if key == ord(' '):
            if status['is_game_over']:
                print("Game is over! Press ESC to quit.")
                continue
                
            if not is_human_turn:
                print(f"⚠ Not your turn! Current turn: {current_turn}, You are: {human_turn}")
                continue
            
            print(f"\n=== MOVE {int(move_count) + 1} ===")
            print(f"Human ({human_turn}) to move...")
            
            move_info = vision.detect_move(frame)
            
            if move_info['valid']:
                print(f"✅ Human move detected: {move_info['from_square']} to {move_info['to_square']}")
                
                result = game.game_loop_iteration(move_info)
                
                if result:
                    print(f"✅ Engine will respond next turn")
                    move_count += 0.5
                    
                    # Sync vision after human move
                    vision.reset_detection_state()
                    vision.sync_detection_state(frame)

                    
                    # Show move visualization
                    show_move_visualization(cap, config, vision, move_info, "Human")
                    waiting_for_human = False
                else:
                    print("✅ Human move processed")
                    move_count += 0.5
            else:
                print(f"✗ Invalid move: {move_info.get('error', 'Unknown error')}")
        
        elif key == ord('d'):
            handle_difficulty_change(config, game)
        
        elif key == ord('c'):
            handle_color_change(config, game)


def render_game_display(frame, config, status, human_turn, engine_thinking, move_count):
    """Render game display with overlays."""
    display_frame = cv2.resize(frame, 
                              (int(frame.shape[1] * config.display_scale), 
                               int(frame.shape[0] * config.display_scale)))
    
    current_turn = status['turn']
    
    cv2.putText(display_frame, f"Turn: {current_turn.capitalize()}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Human: {config.human_color.capitalize()}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, f"Elo: {config.difficulty_elo} | Moves: {move_count}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if engine_thinking:
        cv2.putText(display_frame, "🤖 Engine thinking...", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    elif (current_turn == human_turn) and not status['is_game_over']:
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
    
    return display_frame


def show_engine_thinking(cap, config, game):
    """Show engine thinking animation."""
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            display_frame = cv2.resize(frame, 
                                      (int(frame.shape[1] * config.display_scale), 
                                       int(frame.shape[0] * config.display_scale)))
            cv2.putText(display_frame, f"Turn: {'White' if game.board.turn else 'Black'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "🤖 Engine thinking...", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.imshow("Chess Vision - Game", display_frame)
            cv2.waitKey(16)


def show_move_visualization(cap, config, vision, move_info, player):
    """Show move visualization."""
    ret, frame = cap.read()
    if ret:
        detections = vision.detect_pieces(frame)
        occupancy, _ = vision.calculate_occupancy(detections)
        vis = vision.visualize_detection(frame, detections, occupancy)
        vis_display = cv2.resize(vis, 
                                (int(vis.shape[1] * config.display_scale), 
                                 int(vis.shape[0] * config.display_scale)))
        
        from_sq = move_info['from_square']
        to_sq = move_info['to_square']
        cv2.putText(vis_display, f"{player}: {from_sq}→{to_sq}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(f"{player} Move", vis_display)
        cv2.waitKey(1500)
        cv2.destroyWindow(f"{player} Move")


def handle_difficulty_change(config, game):
    """Handle difficulty change."""
    print(f"\nCurrent difficulty: {config.difficulty_elo} Elo")
    print("Enter new Elo (500-2000): ", end='')
    
    try:
        new_elo = int(input())
        if 500 <= new_elo <= 2000:
            config.difficulty_elo = new_elo
            game.set_difficulty(new_elo)
            print(f"✅ Difficulty set to {new_elo} Elo")
        else:
            print("✗ Elo must be between 500 and 2000")
    except ValueError:
        print("✗ Invalid number")


def handle_color_change(config, game):
    """Handle human color change."""
    if config.human_color == "white":
        config.human_color = "black"
        game.set_human_color("black")
    else:
        config.human_color = "white"
        game.set_human_color("white")
    
    print(f"✅ Human color changed to {config.human_color}")


if __name__ == "__main__":
    main()