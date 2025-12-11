# config.py - Configuration management
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GameConfig:
    """Game configuration with defaults."""
    difficulty_elo: int = 1200
    human_color: str = "black"  # "white" or "black"
    camera_index: int = 0
    display_scale: float = 0.5
    stockfish_path: str = "stockfish"
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