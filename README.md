# Physics-Impact Weighted Working-Point Conformal Training (PI-WPCT)

[![CI](https://github.com/697kiran/pi-wpct-jet-tagging/actions/workflows/ci.yml/badge.svg)](https://github.com/697kiran/pi-wpct-jet-tagging/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Physics-Impact Weighted Working-Point Conformal Training (PI-WPCT) for Robust and Calibration-Aware Jet Flavour Tagging at the LHC.**  
*Amrita Vishwa Vidyapeetham, Amritapuri — 23AID498 Project (Team 6)*  
*Guide: Dr. Vivek Menon*  
*Team Members: R Kiran, Kurt Sony Rebello, Adityan Rajesh P*

---

## Overview

PI-WPCT extends **Working-Point Conformal Training (WPCT)** for three-class jet flavour tagging ($b$-jet vs $c$-jet vs light-jet) using the JetClass dataset.

While standard WPCT employs fixed or manually-tuned loss penalty weights for light and charm mistags, **PI-WPCT introduces a periodic, non-differentiable risk-allocation mechanism**. At working-point $w$, the algorithm evaluates the local sensitivity of a downstream analysis proxy (selected $b$-jet purity $P(w)$) to competing background mistags and weights them by their empirical validation variance:

$$R_i(w) = \left[\frac{\partial P}{\partial r_i}\right]^2 \cdot \text{Var}[r_i(w)], \quad i \in \{\text{light}, \text{charm}\}$$

$$\lambda_{\text{light}}(w) = \lambda_{\text{bg}} \cdot \frac{R_{\text{light}}}{R_{\text{light}} + R_{\text{charm}}}, \quad \lambda_{\text{charm}}(w) = \lambda_{\text{bg}} \cdot \frac{R_{\text{charm}}}{R_{\text{light}} + R_{\text{charm}}}$$

This preserves the original WPCT loss skeleton, model backbones, exact post-training Mondrian conformal calibration, and robustness-testing framework while making background penalization adapt dynamically to the working point and the evolving validation risk structure.

---

## Key Methodological Principles & Hard Guarantees

1. **Strict 4-Way Split & Zero Leakage**:
   - `train`: Gradient optimization only. Never touched by risk allocation or calibration logic.
   - `validation`: Model selection and periodic risk-weight refresh (PI-WPCT only). Never used for headline test intervals.
   - `calibration`: Held out exclusively for exact post-training Mondrian conformal calibration. Never accessed during training.
   - `test`: Final headline evaluation only. Never used for tuning.
   - Automated runtime checks (`SplitAuditor`) fail immediately upon any split leakage.
2. **Strictly Non-Differentiable Risk Allocation**:
   - Executed under `torch.no_grad()` on validation data only at $K$-epoch boundaries.
   - Imposes zero gradient path through the downstream risk calculation.
3. **Automated Null-Result / Failure Surface**:
   - Explicitly detects flat risk ratios across candidate working points.
   - Requires dynamic gains to exceed paired bootstrap 95% confidence intervals to claim a performance advantage.
   - Explicitly clarifies that PI-WPCT is a lightweight loss re-weighting method and does *not* claim INFERNO/SANNT-scale full downstream statistical profile likelihood optimization.

---

## Repository Structure

```
pi-wpct/
├── configs/                             # Predeclared YAML configs
│   ├── data.yaml                        # Split fractions, pilot parameters, yields
│   ├── backbones/                       # {gbt, pn_lite, lorentznet_small, part}.yaml
│   ├── regimes/                         # {baseline_ce, ce_temp_scaling, ce_mondrian_cp, conftr, wpct, pi_wpct}.yaml
│   └── experiment.yaml                  # Seeds, target WPs, K refresh interval
├── src/piwpct/
│   ├── data.py                          # JetClass filtering, tanh(d0)/tanh(dz), canonical splits
│   ├── metrics.py                       # WP metrics, bootstrap CIs, McNemar, ECE
│   ├── conformal.py                     # Exact class-conditional Mondrian conformal calibration
│   ├── losses.py                        # WPCTLoss, PIWPCTLoss, ConfTrLoss, BaselineCELoss
│   ├── risk_allocator.py                # Periodic non-differentiable risk allocation
│   ├── train.py                         # Shared training loop & checkpoint manager
│   ├── backbones/                       # ParticleNet-Lite, LorentzNet-small, ParT, GBT
│   ├── evaluation/                      # Reporting, robustness perturbations, publication plots
│   └── utils/                           # Seeding, split integrity auditor, trajectory logging
├── scripts/
│   ├── build_splits.py                  # Generates and hashes canonical 4-way split
│   ├── run_pretraining_diagnostic.py    # Mandatory pre-training risk diagnostic (§6 & §9.2)
│   ├── run_regime.py                    # Train and evaluate single regime
│   └── run_full_ablation_matrix.py      # Run full ablation grid and comparative report
├── tests/                               # Comprehensive unit and leakage test suite
├── docs/
│   ├── ARCHITECTURE.md                  # 7-layer architecture and contracts
│   └── REPRODUCIBILITY.md               # Step-by-step reproduction instructions
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/697kiran/pi-wpct-jet-tagging.git
cd pi-wpct-jet-tagging
pip install -e .[dev]
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Generate Canonical Split

```bash
python scripts/build_splits.py --pilot --output-dir data/splits --seed 42
```

### 4. Run Pre-Training Diagnostic Check

```bash
python scripts/run_pretraining_diagnostic.py --config configs/experiment.yaml
```

### 5. Train and Evaluate PI-WPCT

```bash
python scripts/run_regime.py --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
```

### 6. Run Full Ablation Matrix (WPCT vs PI-WPCT)

```bash
python scripts/run_full_ablation_matrix.py --backbone pn_lite --regimes baseline_ce wpct pi_wpct --epochs 3
```

---

## Experimental Regimes

| Regime | Description | Conformal Calibration |
|---|---|---|
| `baseline_ce` | Standard Cross-Entropy with raw softmax threshold | None |
| `ce_temp_scaling` | Cross-Entropy with validation-fitted post-hoc temperature scaling | None |
| `ce_mondrian_cp` | Standard Cross-Entropy followed by exact Mondrian conformal calibration | Exact Mondrian (cal split) |
| `conftr` | Differentiable conformal training (Stutz et al., ICLR 2022) | Exact Mondrian (cal split) |
| `wpct` | Working-Point loss with fixed/manually-tuned background mistag weights | Exact Mondrian (cal split) |
| `pi_wpct` | WPCT with periodic validation physics-impact risk allocation | Exact Mondrian (cal split) |

---

## References

1. Qu, H., Li, C. and Qian, S. et al., *JetClass: a large-scale dataset for high-energy particle physics machine learning*, Zenodo DOI: [10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768).
2. Stutz, D. et al., *Learning Optimal Conformal Classifiers*, ICLR 2022.
3. Ding, T. et al., *Class-conditional conformal prediction with finite-sample coverage guarantees*.
4. Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J., *Conformal prediction beyond exchangeability*, Annals of Statistics, 2023.
5. de Castro, L. and Cranmer, K., *INFERNO: Inference-Aware Neural Optimisation*, EPJ Web of Conferences, 2019. *(Cited as adjacent prior art; PI-WPCT does not differentiate through downstream statistical fits)*.
