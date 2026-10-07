"""LorentzNet-small architecture.

Lorentz-equivariant and invariant neural network for jet flavour tagging.
Operates on particle 4-momenta (px, py, pz, E) and constituent scalar features.
Reference: Gong et al., LorentzNet, JHEP 2022.
"""

from typing import Optional, Sequence
import torch
import torch.nn as nn

from piwpct.schema import FEATURE_INDEX, FEATURE_NAMES


def minkowski_dot(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Compute Minkowski inner product: E1*E2 - px1*px2 - py1*py2 - pz1*pz2.

    Args:
        p1: [..., 4] (px, py, pz, E)
        p2: [..., 4] (px, py, pz, E)
    """
    px1, py1, pz1, e1 = p1[..., 0], p1[..., 1], p1[..., 2], p1[..., 3]
    px2, py2, pz2, e2 = p2[..., 0], p2[..., 1], p2[..., 2], p2[..., 3]
    return e1 * e2 - (px1 * px2 + py1 * py2 + pz1 * pz2)


def psi(minkowski_prod: torch.Tensor) -> torch.Tensor:
    """Sign-preserving logarithm of Minkowski metric for numerical scaling."""
    sign = torch.sign(minkowski_prod)
    return sign * torch.log(torch.abs(minkowski_prod) + 1.0)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block."""

    def __init__(self, n_scalar: int, n_hidden: int, dropout: float = 0.1, c_weight: float = 1e-3):
        super().__init__()
        self.c_weight = c_weight
        # Message MLP: takes (h_i, h_j, psi(p_i, p_j)) -> message
        self.phi_m = nn.Sequential(
            nn.Linear(2 * n_scalar + 1, n_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
        )

        # Coordinate vector weighting MLP
        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, 1),
        )

        # Scalar update MLP
        self.phi_h = nn.Sequential(
            nn.Linear(n_scalar + n_hidden, n_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_scalar),
        )

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple:
        """Forward pass.

        Args:
            h: [B, N, n_scalar]
            x: [B, N, 4] 4-momenta
            mask: [B, N]
        """
        B, N, _ = h.shape

        # Pairwise Minkowski dot products
        # x: [B, N, 1, 4], [B, 1, N, 4]
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)
        m_dot = minkowski_dot(x_i, x_j)             # [B, N, N]
        psi_dot = psi(m_dot).unsqueeze(-1)          # [B, N, N, 1]

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, S]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, S]

        m_input = torch.cat([h_i, h_j, psi_dot], dim=-1)  # [B, N, N, 2S+1]
        msg = self.phi_m(m_input)                         # [B, N, N, n_hidden]

        # Pairwise mask
        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1)  # [B, N, N, 1]
        msg = msg * pair_mask

        # Equivariant 4-momentum update
        w_x = self.phi_x(msg) * self.c_weight            # [B, N, N, 1]
        delta_x = (w_x * x_j).sum(dim=2)                 # [B, N, 4]
        new_x = (x + delta_x) * mask.unsqueeze(-1)

        # Invariant scalar update
        agg_msg = msg.sum(dim=2)                         # [B, N, n_hidden]
        new_h = (h + self.phi_h(torch.cat([h, agg_msg], dim=-1))) * mask.unsqueeze(-1)

        return new_h, new_x


class LorentzNetSmall(nn.Module):
    """Compact LorentzNet model for a configurable jet classification target."""

    def __init__(
        self,
        n_scalar: int = len(FEATURE_NAMES),
        scalar_feature_names: Optional[Sequence[str]] = None,
        n_hidden: int = 64,
        n_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        names = tuple(scalar_feature_names or FEATURE_NAMES)
        if len(names) != n_scalar:
            raise ValueError(
                f"n_scalar={n_scalar} disagrees with scalar_feature_names length {len(names)}."
            )
        unknown = [name for name in names if name not in FEATURE_INDEX]
        if unknown:
            raise ValueError(f"Unknown LorentzNet scalar feature names: {unknown}.")
        self.scalar_feature_names = names
        self.scalar_feature_indices = tuple(FEATURE_INDEX[name] for name in names)
        self.n_scalar = len(names)
        self.embedding = nn.Sequential(
            nn.Linear(n_scalar, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.layers = nn.ModuleList([
            LGEB(n_hidden, n_hidden, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        vectors: Optional[torch.Tensor] = None,
        high_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with uniform interface."""
        B, N, C = features.shape

        if mask is None:
            mask = torch.ones((B, N), device=features.device, dtype=torch.float32)

        if C != len(FEATURE_NAMES):
            raise ValueError(
                f"LorentzNet expects the {len(FEATURE_NAMES)}-feature schema, received {C} channels."
            )
        if vectors is None:
            raise ValueError("LorentzNet requires explicit raw four-vectors; it does not infer them from features.")
        if vectors.shape != (B, N, 4):
            raise ValueError(f"LorentzNet vectors must have shape {(B, N, 4)}, got {tuple(vectors.shape)}.")
        scalars = features[:, :, list(self.scalar_feature_indices)]

        h = self.embedding(scalars) * mask.unsqueeze(-1)
        x = vectors

        for layer in self.layers:
            h, x = layer(h, x, mask)

        # Global sum/average pooling over valid constituents
        mask_sum = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        logits = self.classifier(pooled)
        return logits
