import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import cfg

class SignSkeletonDataset(Dataset):
    """
    Loads (N, C, T, V) coordinate tensors and applies on-the-fly 3D geometric
    augmentations during training.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.features = features.astype(np.float32)  # Shape: (N, 3, T, 53)
        self.labels = labels.astype(np.int64)        # Shape: (N,)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def _rotate_y_axis(self, x: np.ndarray, max_deg: float = 12.0) -> np.ndarray:
        """Rotates (C=3, T, V) skeleton around the vertical Y axis by a random angle."""
        theta = np.radians(np.random.uniform(-max_deg, max_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([
            [ cos_t, 0.0, sin_t],
            [   0.0, 1.0,   0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float32)
        # tensordot over channel axis 0: (3, 3) x (3, T, V) -> (3, T, V)
        return np.tensordot(rot_matrix, x, axes=([1], [0]))

    def __getitem__(self, idx: int):
        x = self.features[idx].copy()  # (3, T, V)
        y = self.labels[idx]

        if self.augment:
            x = self._rotate_y_axis(x)
            # Add minor Gaussian tracking jitter
            noise = np.random.normal(0.0, 0.008, size=x.shape).astype(np.float32)
            x = x + noise

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def generate_synthetic_dataset(num_samples: int = 320):
    """
    Generates structured synthetic class-distinct trajectories of shape (N, 3, T, V)
    so the repository can be trained and tested out-of-the-box in VS Code.
    """
    rng = np.random.default_rng(seed=42)
    X = np.zeros((num_samples, cfg.IN_CHANNELS, cfg.SEQ_LENGTH, cfg.NUM_NODES), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    t_vals = np.linspace(0, 2 * np.pi, cfg.SEQ_LENGTH, dtype=np.float32)

    for i in range(num_samples):
        cls_id = i % cfg.num_classes
        y[i] = cls_id
        freq = 1.0 + 0.35 * cls_id
        phase = 0.5 * cls_id
        # Inject class-specific harmonic motion into hand nodes (indices 11..52)
        for v in range(11, cfg.NUM_NODES):
            X[i, 0, :, v] = np.sin(freq * t_vals + phase + v * 0.05)
            X[i, 1, :, v] = np.cos(freq * t_vals - phase + v * 0.05)
            X[i, 2, :, v] = 0.5 * np.sin(0.5 * freq * t_vals)
        # Add realistic sample variance
        X[i] += rng.normal(0, 0.12, size=X[i].shape).astype(np.float32)

    return X, y