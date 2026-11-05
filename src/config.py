from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectConfig:
    # Directories
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT_DIR / "data" / "processed"
    CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
    
    # Skeleton Graph Topology
    # 11 upper-body pose nodes + 21 left-hand nodes + 21 right-hand nodes = 53 nodes
    NUM_NODES: int = 53
    IN_CHANNELS: int = 3          # (x, y, z) normalized coordinates
    SEQ_LENGTH: int = 30          # 30 frames per sign window (~1 second at 30 FPS)
    
    # Vocabulary classes
    CLASSES: tuple = (
        "HELLO", "THANK_YOU", "HELP", "DOCTOR", "PAIN",
        "WATER", "YES", "NO", "PLEASE", "EMERGENCY"
    )
    
    # Training Hyperparameters
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 25
    TEMPORAL_KERNEL_SIZE: int = 9
    DROPOUT: float = 0.25

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

cfg = ProjectConfig()
cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)