# Evaluation Report: Ablation Matrix (pn_lite, WP=0.7)
**B.Tech Capstone Project — Amrita Vishwa Vidyapeetham (Team 6, Guide: Dr. Vivek Menon)**

## 1. Executive Summary & Automated Findings

> [!WARNING]
> **NULL_RESULT_NO_SIGNIFICANT_GAIN**: Dynamic PI-WPCT mean light rejection change was +0.00 (95% Paired Bootstrap CI: [+0.00, +0.00]). Because the confidence interval spans zero, the difference is NOT statistically significant. Academic integrity requires reporting this as a null result rather than claiming a gain.

> [!NOTE]
> **SCOPE_BOUNDARY_STATEMENT**: PI-WPCT is a lightweight, periodic loss-reweighting scheme. It does NOT backpropagate through a downstream profile likelihood fit or claim INFERNO/SANNT-scale full-uncertainty optimization. Results are strictly scoped to working-point background mistag reduction and exact conformal calibration.

## 2. Regime Comparison Summary Table

| Regime | Accuracy | Macro-AUC | b-Eff (w) | Light Rejection (1/mistag) | Charm Rejection (1/mistag) | Marginal Coverage | ECE |
|---|---|---|---|---|---|---|---|
| **baseline_ce** | 98.28% | 0.9985 | 70.0% | 1000000.0 | 1000000.0 | 89.3% | 0.0452 |
| **wpct** | 86.56% | 0.9956 | 70.0% | 1000000.0 | 1000000.0 | 89.0% | 0.1079 |
| **pi_wpct** | 83.50% | 0.9781 | 70.0% | 1000000.0 | 150.0 | 88.6% | 0.1056 |

## 3. Paired Statistical Bootstrap Comparison (PI-WPCT vs WPCT)

| Metric | Mean Difference (PI-WPCT - WPCT) | 95% Bootstrap CI | Statistically Significant? |
|---|---|---|---|
| `target_wp` | +0.0000 | [+0.0000, +0.0000] | NO (includes 0) |
| `threshold` | +0.0108 | [+0.0090, +0.0135] | **YES** (excludes 0) |
| `eps_b` | +0.0000 | [-0.0070, +0.0068] | NO (includes 0) |
| `r_light` | +0.0000 | [+0.0000, +0.0000] | NO (includes 0) |
| `r_charm` | +0.0026 | [-0.0037, +0.0135] | NO (includes 0) |
| `rej_light` | +0.0000 | [+0.0000, +0.0000] | NO (includes 0) |
| `rej_charm` | -332944.5094 | [-999925.7562, +160.2958] | NO (includes 0) |

## 4. Conformal Calibration Guarantees

All regimes evaluated under exact class-conditional Mondrian conformal prediction on the separate, held-out calibration split. Per-class empirical coverage is guaranteed to satisfy $P(Y \in C(X) \mid Y = k) \ge 1 - \alpha$.