# arm_controller.py - Robot arm control
import json
import time
import sys  # Path setup first
sys.path.insert(0, "/usr/local/lib/python3.11/dist-packages/Arm_Lib-0.0.5-py3.11.egg")
from Arm_Lib import Arm_Device  # Import after


class ArmController:
    """Controller for the robotic arm."""
    
    def __init__(self, board_config_path="all_squares_touch.json"):
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
            base[1] += 4
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
        src_open[1] += 2
        
        self.move(self.NEUTRAL_POS_OPEN, 800, 1.0)
        self.move(src_open, 2000, 2.0)
        self.move(src_close, 500, 0.5)
        self.move(self.NEUTRAL_POS, 2000, 2.0)
        self.move(dst_close, 2000, 2.0)
        self.move(dst_open, 500, 0.7)
        self.move(self.NEUTRAL_POS, 2000, 2.0)
        
        print(f"✅ Moved {src} -> {dst}")
    
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
        
        print(f"✅ Removed piece from {sq}")
    
    def test_move(self):
        """Test function to verify arm is working."""
        print("🤖 Testing arm movement...")
        self.move_piece("e2", "e4")
        print("✅ Arm test completed")