"""Backbone contract checks for 17 scalars, explicit points, p4s and masks."""

import pytest
import torch

from piwpct.backbones import build_backbone
from piwpct.schema import FEATURE_NAMES


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("pn_lite", {"input_dim": 17, "num_classes": 3, "k_neighbors": 2,
                     "conv_params": [[8]], "fc_params": [[8, 0.0]]}),
        ("part", {"input_dim": 17, "num_classes": 3, "embed_dim": 16,
                  "num_heads": 4, "num_layers": 1, "dropout": 0.0}),
        ("lorentznet_small", {"n_scalar": 17, "scalar_feature_names": list(FEATURE_NAMES),
                               "n_hidden": 8, "n_layers": 1, "num_classes": 3, "dropout": 0.0}),
    ],
)
def test_neural_backbones_accept_authoritative_inputs(name, config):
    model = build_backbone(name, config)
    features = torch.randn(2, 4, 17)
    points = torch.randn(2, 4, 2)
    vectors = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    logits = model(features=features, points=points, vectors=vectors, mask=mask)
    assert logits.shape == (2, 3)


def test_lorentznet_requires_named_scalar_width_and_separate_four_vectors():
    model = build_backbone(
        "lorentznet_small",
        {"n_scalar": 17, "scalar_feature_names": list(FEATURE_NAMES), "n_hidden": 8, "n_layers": 1},
    )
    with pytest.raises(ValueError, match="requires explicit raw four-vectors"):
        model(features=torch.zeros(1, 2, 17), mask=torch.ones(1, 2))
    with pytest.raises(ValueError, match="expects the 17-feature schema"):
        model(features=torch.zeros(1, 2, 16), vectors=torch.zeros(1, 2, 4))
