# System Architecture: PI-WPCT

**Physics-Impact Weighted Working-Point Conformal Training (PI-WPCT) for Robust and Calibration-Aware Jet Flavour Tagging at the LHC**  
*Amrita Vishwa Vidyapeetham — 23AID498 Capstone Project (Team 6)*  
*Guide: Dr. Vivek Menon*  
*Team Members: R Kiran, Kurt Sony Rebello, Adityan Rajesh P*

---

## 1. High-Level Architecture Overview

The system extends Working-Point Conformal Training (WPCT) for 3-class jet flavour tagging ($b$-jet vs $c$-jet vs light-jet) using the JetClass dataset. The core innovation is a **periodic, non-differentiable risk-allocation mechanism** that dynamically sets the relative loss weights between light and charm mistags based on a physics-motivated downstream purity proxy.

```
+-----------------------------------------------------------------------------+
|                             JETCLASS 3-CLASS DATA                           |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                 LAYER 1: PREPROCESSING & CANONICAL 4-WAY SPLIT              |
|        - tanh(d0), tanh(dz) impact parameter transforms                     |
|        - Strictly disjoint: train / val / cal / test                        |
|        - Content-hashed SHA-256 manifest artifact                           |
+-----------------------------------------------------------------------------+
          |                         |                        |
          | (train only)            | (val only)             | (cal only)
          v                         |                        |
+----------------------+            |                        |
|  LAYER 2: BACKBONES  |            |                        |
| - ParticleNet-Lite   |            |                        |
| - LorentzNet-small   |            |                        |
| - ParT               |            |                        |
| - GBT (XGB/LGBM)     |            |                        |
+----------------------+            |                        |
          |                         |                        |
          v                         |                        |
+----------------------+            |                        |
|  LAYER 3: LOSS       |            |                        |
| - WPCTLoss           |            |                        |
| - PIWPCTLoss         |            |                        |
| - ConfTrLoss         |            |                        |
+----------------------+            |                        |
          |                         |                        |
          v                         |                        |
+----------------------+            |                        |
|  LAYER 4: TRAIN LOOP |            |                        |
| - Shared optimizer   |            |                        |
| - K-epoch boundary   |<-----------+                        |
+----------------------+  (every K  |                        |
          |                epochs)  |                        |
          |                         v                        |
          |             +-----------------------+            |
          |             | LAYER 5: RISK REFRESH |            |
          |             | - Freeze model        |            |
          |             | - Estimate purity P   |            |
          |             | - dP/dr sensitivities |            |
          |             | - Var[r] with floor   |            |
          |             | - Update scalar wts   |            |
          |             +-----------------------+            |
          |                                                  |
          v                                                  |
+------------------------------------------------------------+----------------+
|                        LAYER 6: POST-TRAINING CONFORMAL CALIBRATION          |
|  - Exact Mondrian (class-conditional) conformal calibration on frozen logits|
|  - Strictly isolated to CALIBRATION split                                   |
|  - P(Y in C(X) | Y = k) >= 1 - alpha                                        |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                 LAYER 7: EVALUATION & ROBUSTNESS REPORTING                  |
|  - Held-out TEST split only                                                 |
|  - Fixed-WP rejections, Accuracy, Macro-AUC, ECE                            |
|  - Paired bootstrap statistical significance CIs                            |
|  - Automated null-result / failure-mode detection                           |
|  - Detector mismodelling sweeps (JES, JER, IP smear, track drop)             |
+-----------------------------------------------------------------------------+
```

---

## 2. The Seven Logical Layers

### Layer 1: Data Ingestion and Canonical Splits (`src/piwpct/data.py`)
- Filters JetClass into the 3-class flavour tagging task: light quarks/gluons ($0$), charm ($1$), bottom ($2$).
- Computes non-linear impact parameter transforms $\tanh(d_0), \tanh(d_z)$ to tame high-energy tails while preserving linear resolution near zero.
- Produces the canonical 4-way split (`train`, `validation`, `calibration`, `test`). A content-hashed JSON manifest (`canonical_split_manifest.json`) records SHA-256 checksums and exact indices.
- Supports **pilot mode** with provisional light:b:c class ratio ~4:1:1, explicitly flagged in metadata and logs.

### Layer 2: Backbone Inference (`src/piwpct/backbones/`)
All backbones adhere to a uniform forward interface:
```python
forward(features, points=None, mask=None, vectors=None, high_level=None) -> logits [B, 3]
```
1. **ParticleNet-Lite**: EdgeConv dynamic $k$-NN graph neural network in coordinate space $(\Delta\eta, \Delta\phi)$.
2. **LorentzNet-small**: Equivariant graph network processing 4-momenta $(p_x, p_y, p_z, E)$ using Minkowski metric inner products and coordinate updates.
3. **Particle Transformer (ParT)**: Multi-head particle self-attention with learned `[CLS]` token representation.
4. **GBT (XGBoost / LightGBM)**: Tree ensemble trained on summary jet observables, outputting pseudo-logits.

### Layer 3: Working-Point Loss Formulation (`src/piwpct/losses.py`)
The shared objective for target working point $w$ is:
$$\mathcal{L}(w) = \mathcal{L}_{\text{CE}} + \lambda_{\text{eff}} |\epsilon_b^{\text{soft}} - w| + \lambda_{\text{light}}(w) \mathcal{L}_{\text{light}}^{\text{mistag}} + \lambda_{\text{charm}}(w) \mathcal{L}_{\text{charm}}^{\text{mistag}}$$
- `WPCTLoss`: Uses fixed or manually-tuned static $\lambda_{\text{light}}, \lambda_{\text{charm}}$.
- `PIWPCTLoss`: Identical skeleton, but $\lambda_{\text{light}}(w), \lambda_{\text{charm}}(w)$ are injected at epoch boundaries from `RiskAllocator` as read-only scalar parameters.
- `ConfTrLoss`: Differentiable conformal training loss (Stutz et al., ICLR 2022) with smooth coverage and set-size penalties.

### Layer 4: Shared Training Loop (`src/piwpct/train.py`)
- Standard gradient backpropagation on the `train` split only.
- Hooks the periodic risk refresh routine every $K$ epochs for PI-WPCT.
- Checkpoints include model weights, full config snapshot, git commit hash, seed, and dataset split manifest hash.

### Layer 5: Non-Differentiable Risk Allocation (`src/piwpct/risk_allocator.py`)
Executed strictly under `torch.no_grad()` on held-out **validation data only**:
1. Evaluates validation logits and identifies threshold $\tau_w$ such that true $b$-efficiency $\epsilon_b \approx w$.
2. Computes background mistag rates $r_{\text{light}}(w), r_{\text{charm}}(w)$.
3. Computes stabilized variances $\text{Var}[r_i(w)]$ using Wilson-score regularization and a minimum variance floor ($\sigma^2_{\min} = 10^{-7}$).
4. Computes selected sample purity:
   $$P(w) = \frac{S(w)}{S(w) + B_{\text{light}}(w) + B_{\text{charm}}(w)} = \frac{S}{S + Y_{\text{light}} r_{\text{light}} + Y_{\text{charm}} r_{\text{charm}}}$$
5. Evaluates analytic sensitivities:
   $$\frac{\partial P}{\partial r_i} = - \frac{S \cdot Y_i}{[S + Y_{\text{light}} r_{\text{light}} + Y_{\text{charm}} r_{\text{charm}}]^2}$$
6. Calculates first-order risk contributions:
   $$R_i(w) = \left[\frac{\partial P}{\partial r_i}\right]^2 \cdot \text{Var}[r_i(w)], \quad i \in \{\text{light}, \text{charm}\}$$
7. Normalizes into loss weights:
   $$\lambda_{\text{light}}(w) = \lambda_{\text{bg}} \frac{R_{\text{light}}}{R_{\text{light}} + R_{\text{charm}}}, \quad \lambda_{\text{charm}}(w) = \lambda_{\text{bg}} \frac{R_{\text{charm}}}{R_{\text{light}} + R_{\text{charm}}}$$
   with fallback guards if $R_{\text{light}} + R_{\text{charm}} < \epsilon$.

### Layer 6: Exact Mondrian Conformal Calibration (`src/piwpct/conformal.py`)
- Frozen model logits on the separate, held-out **calibration split** are used to fit exact class-conditional quantiles $\hat{q}_k$ for $k \in \{0, 1, 2\}$.
- For any test sample $X$, the prediction set is $C(X) = \{k : s(X, k) \le \hat{q}_k\}$.
- Guarantees finite-sample conditional coverage:
  $$P(Y \in C(X) \mid Y = k) \ge 1 - \alpha$$
- Completely decoupled from training regimes: identical calibration path for Baseline CE, WPCT, and PI-WPCT.

### Layer 7: Evaluation & Robustness Reporting (`src/piwpct/evaluation/`)
- Evaluated strictly on the held-out **test split**.
- Systematic detector-mismodelling stress tests: Jet Energy Scale (JES), Jet Energy Resolution (JER), Impact Parameter smearing, and Tracking inefficiency.
- Paired bootstrap 95% confidence intervals and McNemar test.
- **Automated failure-mode / null-result detection**: flags flat risk ratios, non-significant gains, or proxy instabilities.

---

## 3. Strict Split Leakage Rules

| Split | Primary Role | PI-WPCT Role | Forbidden Actions |
|---|---|---|---|
| `train` | Gradient descent | Loss computation | Never used for risk updates or conformal calibration |
| `validation` | Model selection | Periodic risk allocation refresh | Never used for headline test evaluation |
| `calibration` | Conformal calibration | Class quantiles $\hat{q}_k$ | **NEVER** touched during training or risk refresh |
| `test` | Headline evaluation | Final metrics & robustness | **NEVER** used for training, risk, or hyperparameter tuning |

These boundaries are actively enforced in code via `SplitAuditor` (`src/piwpct/utils/split_integrity.py`), raising `SplitIntegrityViolation` if violated.

---

## 4. Scope and Distinction from Prior Art

- **INFERNO / SANNT**: These methods perform differentiable end-to-end backpropagation through a downstream statistical profile likelihood fit. **PI-WPCT does NOT do this**. PI-WPCT is a lightweight, non-differentiable loss-reweighting scheme operating on validation outputs at epoch boundaries.
- **Conformal Prediction**: Conformal calibration provides post-hoc finite-sample coverage guarantees. PI-WPCT changes the upstream training weights, preserving the downstream conformal calibration pipeline intact.
