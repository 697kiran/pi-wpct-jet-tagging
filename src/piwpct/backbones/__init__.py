"""Backbone model architectures for jet flavour tagging.

Provides uniform forward interface:
    forward(features, points=None, mask=None, vectors=None, high_level=None) -> logits [B, 3]
"""

from piwpct.backbones.gbt import GBTModel
from piwpct.backbones.lorentznet_small import LorentzNetSmall
from piwpct.backbones.part import ParticleTransformer
from piwpct.backbones.particlenet_lite import ParticleNetLite


def build_backbone(backbone_name: str, config: dict):
    """Instantiate a backbone model by name."""
    name = backbone_name.lower().replace("-", "_")
    if name in {"pn_lite", "particlenet_lite"}:
        return ParticleNetLite(
            input_dim=config.get("input_dim", 16),
            num_classes=config.get("num_classes", 3),
            k_neighbors=config.get("k_neighbors", 8),
            conv_params=config.get("conv_params", [[32, 32], [64, 64]]),
            fc_params=config.get("fc_params", [[128, 0.1]]),
        )
    elif name in {"lorentznet", "lorentznet_small"}:
        return LorentzNetSmall(
            n_scalar=config.get("n_scalar", 12),
            n_hidden=config.get("n_hidden", 64),
            n_layers=config.get("n_layers", 3),
            num_classes=config.get("num_classes", 3),
            dropout=config.get("dropout", 0.1),
        )
    elif name in {"part", "particle_transformer"}:
        return ParticleTransformer(
            input_dim=config.get("input_dim", 16),
            num_classes=config.get("num_classes", 3),
            embed_dim=config.get("embed_dim", 64),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 3),
            mlp_ratio=config.get("mlp_ratio", 2.0),
            dropout=config.get("dropout", 0.1),
        )
    elif name in {"gbt", "xgboost", "lightgbm"}:
        return GBTModel(
            backend=config.get("backend", "xgboost"),
            params=config.get("params", {}),
        )
    else:
        raise ValueError(f"Unknown backbone architecture: {backbone_name}")


__all__ = [
    "ParticleNetLite",
    "LorentzNetSmall",
    "ParticleTransformer",
    "GBTModel",
    "build_backbone",
]
