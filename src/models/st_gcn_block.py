import torch
import torch.nn as nn
from src.config import cfg

class SpatialGraphConv(nn.Module):
    """
    Performs spatial message passing across anatomical joints using partitioned
    normalized adjacency matrices.
    """
    def __init__(self, in_channels: int, out_channels: int, num_subsets: int = 2):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, T, V), A shape: (K, V, V)
        N, _, T, V = x.shape
        x_proj = self.conv(x)  # (B, K * C_out, T, V)
        x_proj = x_proj.view(N, self.num_subsets, -1, T, V)
        # Aggregate spatial neighbor states: (B, K, C_out, T, V) x (K, V, W) -> (B, C_out, T, W)
        out = torch.einsum("nkctv,kvw->nctw", x_proj, A)
        return out.contiguous()

class STGCNBlock(nn.Module):
    """
    Combines Spatial Graph Convolution with a 1D Temporal Convolution and
    a residual skip connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int = cfg.NUM_NODES,
        stride: int = 1,
        temporal_kernel: int = cfg.TEMPORAL_KERNEL_SIZE,
        dropout: float = cfg.DROPOUT
    ):
        super().__init__()
        pad_t = (temporal_kernel - 1) // 2

        self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets=2)
        # Learnable edge importance weighting
        self.edge_importance = nn.Parameter(torch.ones(2, num_nodes, num_nodes))

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(pad_t, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x_spatial = self.gcn(x, A * self.edge_importance)
        x_temporal = self.tcn(x_spatial)
        return self.act(x_temporal + res)