import torch
import torch.nn as nn
from src.config import cfg
from src.graph.adjacency import build_normalized_adjacency
from src.models.st_gcn_block import STGCNBlock

class STGCNSignClassifier(nn.Module):
    """
    Full Spatio-Temporal Graph Convolutional Network for Sign Language Recognition.
    Input Tensor: (Batch, Channels=3, Frames=T, Vertices=53)
    Output Tensor: (Batch, Num_Classes)
    """
    def __init__(
        self,
        in_channels: int = cfg.IN_CHANNELS,
        num_classes: int = cfg.num_classes,
        num_nodes: int = cfg.NUM_NODES
    ):
        super().__init__()
        # Register adjacency matrix as a persistent non-trainable buffer
        A = build_normalized_adjacency(num_nodes)
        self.register_buffer("A", A)

        # Input Batch Normalization across (C * V) channels
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, num_nodes=num_nodes, stride=1, dropout=0.1),
            STGCNBlock(64, 64, num_nodes=num_nodes, stride=1),
            STGCNBlock(64, 128, num_nodes=num_nodes, stride=2),
            STGCNBlock(128, 256, num_nodes=num_nodes, stride=2)
        ])

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape

        # Normalize input coordinates per vertex channel
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # Back to (N, C, T, V)

        for layer in self.layers:
            x = layer(x, self.A)

        # Global Average Pooling across Time (T) and Vertices (V)
        x_pooled = x.mean(dim=[2, 3])  # Shape: (N, 256)
        logits = self.classifier(x_pooled)
        return logits