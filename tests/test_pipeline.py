import numpy as np
import torch
from src.config import cfg
from src.graph.adjacency import build_normalized_adjacency
from src.extract_landmarks import normalize_skeleton_frame
from src.models.sign_classifier import STGCNSignClassifier

def test_adjacency_matrix_shape_and_symmetry():
    A = build_normalized_adjacency(cfg.NUM_NODES)
    assert A.shape == (2, 53, 53)
    # Partition 1 must be symmetric
    assert torch.allclose(A[1], A[1].T, atol=1e-6)

def test_coordinate_scale_and_translation_invariance():
    sample = np.random.randn(53, 3).astype(np.float32)
    # Shift by +100 units and scale by 4x
    transformed = (sample * 4.0) + 100.0
    norm1 = normalize_skeleton_frame(sample)
    norm2 = normalize_skeleton_frame(transformed)
    assert np.allclose(norm1, norm2, atol=1e-4)

def test_stgcn_forward_pass():
    model = STGCNSignClassifier()
    x = torch.randn(4, cfg.IN_CHANNELS, cfg.SEQ_LENGTH, cfg.NUM_NODES)
    out = model(x)
    assert out.shape == (4, cfg.num_classes)