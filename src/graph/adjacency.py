import numpy as np
import torch
from src.config import cfg
from src.graph.skeleton import get_anatomical_edges

def build_normalized_adjacency(num_nodes: int = cfg.NUM_NODES) -> torch.Tensor:
    """
    Constructs a two-partition spatial adjacency tensor of shape (2, V, V):
      Partition 0: Identity matrix (self-loop features)
      Partition 1: Symmetric degree-normalized neighbor adjacency matrix
    """
    I = np.eye(num_nodes, dtype=np.float32)

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in get_anatomical_edges():
        A[u, v] = 1.0
        A[v, u] = 1.0

    degree = np.sum(A, axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = np.power(degree[mask], -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    adj_tensor = np.stack([I, A_norm], axis=0)
    return torch.from_numpy(adj_tensor)
