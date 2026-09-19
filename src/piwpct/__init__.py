"""PI-WPCT: Physics-Impact Weighted Working-Point Conformal Training.

An extension of Working-Point Conformal Training (WPCT) for 3-class jet flavour tagging
(light, charm, b) on JetClass with exact Mondrian conformal calibration and
periodic, non-differentiable downstream risk allocation.
"""

__version__ = "0.1.0"
__author__ = "Amrita Vishwa Vidyapeetham - Team 6 (Guide: Dr. Vivek Menon)"

__all__ = [
    "data",
    "metrics",
    "conformal",
    "losses",
    "risk_allocator",
    "train",
]
