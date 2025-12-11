import sys
import os

# Force Arm_Lib egg path
egg_path = "/usr/local/lib/python3.11/dist-packages/Arm_Lib-0.0.5-py3.11.egg"
if egg_path not in sys.path:
    sys.path.insert(0, egg_path)

# Also add parent directory
parent_dir = "/usr/local/lib/python3.11/dist-packages"
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"? Added Arm_Lib paths:")
print(f"  - {egg_path}")
print(f"  - {parent_dir}")

# Now import
from Arm_Lib import Arm_Device
print("? Arm_Lib imported successfully")