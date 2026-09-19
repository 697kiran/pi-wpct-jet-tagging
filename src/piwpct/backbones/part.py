"""Particle Transformer (ParT) architecture.

Transformer-based architecture for jet tagging with multi-head self-attention
over constituent particles, constituent masking, and classification head.
Reference: Qu, Li, Qian, Particle Transformer for Jet Tagging, ICML 2022.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParticleTransformerBlock(nn.Module):
    """Transformer encoder block for jet constituents."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, embed_dim]
            key_padding_mask: [B, N] boolean mask (True for padded positions)
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ParticleTransformer(nn.Module):
    """Particle Transformer (ParT) model for 3-class jet flavour tagging."""

    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Learnable class token for global jet representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            ParticleTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
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
        B, N, _ = features.shape

        if mask is None:
            mask = torch.ones((B, N), device=features.device, dtype=torch.float32)

        # Project constituent features
        x = self.input_proj(features)  # [B, N, embed_dim]

        # Prepend class token: [B, 1 + N, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Build key_padding_mask: True where padding (mask == 0)
        # Class token at index 0 is always valid (False)
        cls_mask = torch.ones((B, 1), device=mask.device, dtype=mask.dtype)
        combined_mask = torch.cat((cls_mask, mask), dim=1)
        key_padding_mask = (combined_mask == 0)  # [B, 1 + N]

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        # Extract normalized class token representation
        cls_out = self.norm(x[:, 0])  # [B, embed_dim]
        logits = self.head(cls_out)   # [B, num_classes]
        return logits
