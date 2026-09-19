"""ParticleNet-Lite architecture.

Pure PyTorch implementation of ParticleNet-Lite EdgeConv GNN for jet flavour tagging.
Operates on constituent spatial coordinates (Delta eta, Delta phi) and feature channels.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance matrix.

    Args:
        x: [B, N, C]

    Returns:
        dist: [B, N, N]
    """
    x_inner = torch.bmm(x, x.transpose(1, 2))  # [B, N, N]
    x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    dist = x_sq - 2 * x_inner + x_sq.transpose(1, 2)
    return torch.clamp(dist, min=0.0)


def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors in coordinate space.

    Args:
        points: [B, N, 2] (Delta eta, Delta phi)
        k: number of neighbors

    Returns:
        idx: [B, N, k] indices of nearest neighbors
    """
    dist = pairwise_distance(points)  # [B, N, N]
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return idx


def get_graph_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Construct EdgeConv graph feature: [x_i, x_j - x_i].

    Args:
        x: [B, N, C]
        idx: [B, N, k]

    Returns:
        feature: [B, 2*C, N, k]
    """
    B, N, C = x.shape
    k = idx.shape[-1]

    # Gather neighbor features
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, N, k, C]
    x_expanded = x.unsqueeze(1).expand(-1, N, -1, -1)       # [B, N, N, C]
    neighbors = torch.gather(x_expanded, 2, idx_expanded)   # [B, N, k, C]

    central = x.unsqueeze(2).expand(-1, -1, k, -1)          # [B, N, k, C]
    edge_feature = torch.cat([central, neighbors - central], dim=-1)  # [B, N, k, 2*C]

    # Permute to [B, 2*C, N, k] for Conv2d
    return edge_feature.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    """EdgeConv block with batch norm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels_list: List[int]):
        super().__init__()
        layers = []
        c_in = 2 * in_channels
        for c_out in out_channels_list:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ])
            c_in = c_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, C]
            idx: [B, N, k]

        Returns:
            out: [B, N, C_out]
        """
        edge_feat = get_graph_feature(x, idx)  # [B, 2*C, N, k]
        h = self.mlp(edge_feat)                # [B, C_out, N, k]
        # Max-pooling over k neighbors
        out = torch.max(h, dim=-1)[0]          # [B, C_out, N]
        return out.transpose(1, 2).contiguous()  # [B, N, C_out]


class ParticleNetLite(nn.Module):
    """ParticleNet-Lite model for 3-class jet tagging."""

    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 3,
        k_neighbors: int = 8,
        conv_params: Optional[List[List[int]]] = None,
        fc_params: Optional[List[Tuple[int, float]]] = None,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        if conv_params is None:
            conv_params = [[32, 32], [64, 64]]
        if fc_params is None:
            fc_params = [(128, 0.1)]

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.edge_convs = nn.ModuleList()
        in_c = input_dim
        for c_list in conv_params:
            self.edge_convs.append(EdgeConvBlock(in_c, c_list))
            in_c = c_list[-1]

        # Fully connected projection head
        fc_layers = []
        c_fc_in = in_c
        for h_dim, drop in fc_params:
            fc_layers.extend([
                nn.Linear(c_fc_in, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            ])
            c_fc_in = h_dim
        fc_layers.append(nn.Linear(c_fc_in, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(
        self,
        features: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        vectors: Optional[torch.Tensor] = None,
        high_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with uniform interface.

        Args:
            features: [B, N, C]
            points: [B, N, 2] (Delta eta, Delta phi)
            mask: [B, N]
        """
        B, N, C = features.shape

        if points is None:
            points = features[:, :, :2]

        if mask is None:
            mask = torch.ones((B, N), device=features.device, dtype=torch.float32)

        # Batch norm on input features: [B*N, C]
        mask_flat = mask.view(-1) > 0
        feat_flat = features.view(-1, C)
        if torch.any(mask_flat):
            feat_flat_norm = torch.zeros_like(feat_flat)
            feat_flat_norm[mask_flat] = self.input_bn(feat_flat[mask_flat])
            h = feat_flat_norm.view(B, N, C)
        else:
            h = features

        k = min(self.k_neighbors, N - 1) if N > 1 else 1

        # Dynamic graph convolutions
        for conv in self.edge_convs:
            idx = knn(points, k=k)
            h = conv(h, idx)
            # Mask out invalid padded particles
            h = h * mask.unsqueeze(-1)

        # Global average pooling over real constituents
        mask_sum = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)  # [B, 1]
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask_sum         # [B, C_out]

        # Logits [B, num_classes]
        logits = self.classifier(pooled)
        return logits
