# System Architecture: PI-WPCT

**Physics-Impact Weighted Working-Point Conformal Training (PI-WPCT) for Robust and Calibration-Aware Jet Flavour Tagging at the LHC**  
*Amrita Vishwa Vidyapeetham — 23AID498 Capstone Project (Team 6)*  
*Guide: Dr. Vivek Menon*  
*Team Members: R Kiran, Kurt Sony Rebello, Adityan Rajesh P*

---

## 1. High-Level Architecture Overview

The project's scientific objective is three-class jet-flavour tagging ($b$-jet vs $c$-jet vs light-jet). The inspected JetClass ROOT files expose generated-process labels (QCD, Hbb, Hcc, etc.), not per-jet flavour. The ingestion pipeline keeps those targets distinct: process classification is available as a separate task, while flavour-tagging conversion requires verified flavour annotations and fails if they are absent. The core PI-WPCT mechanism remains a **periodic, non-differentiable risk-allocation method** for light and charm mistags on a valid flavour target.

```
+-----------------------------------------------------------------------------+
|        EXPLICIT SYNTHETIC PILOT OR EXPLICIT REAL ROOT/TAR TARGET             |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|        LAYER 1: ROOT/TAR CONVERSION OR EXPLICIT PILOT & 4-WAY SPLIT          |
|        - Named 17-feature schema; separate points, four-vectors, mask       |
|        - Jagged mmap shards and portable source-member manifest             |
|        - Train / validation / calibration / test; source-file cal split     |
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

### Layer 1: Data Ingestion and Canonical Splits (`src/piwpct/root_data.py`, `src/piwpct/data.py`)
- `scripts/convert_root.py` reads TAR archives and ROOT tree `tree` in bounded chunks, one ROOT member at a time. Raw input is read-only; staged members are cleaned after conversion.
- `src/piwpct/schema.py` is authoritative for the ordered 17 candidate model features, raw jagged fields, points, and four-vectors. The candidate transforms were compared against the [official Particle Transformer JetClass feature configuration](https://github.com/jet-universe/particle_transformer/blob/main/data/JetClass/JetClass_full.yaml).
- Prepared data is stored as flat jagged `particles.npy` and `offsets.npy`, plus per-jet metadata, labels, and stable file/entry sample IDs. The DataLoader memory-maps shards, pads/truncates per batch, and derives masks from offsets rather than feature values.
- Official train/validation/test archive partitions are preserved. Whole ROOT source members are reserved deterministically from training for calibration; entries from one source member never cross the train/calibration boundary.
- Pilot mode remains a separate, explicitly selected synthetic dataset with provisional light:charm:b composition. Pilot labels do not represent real detector or truth annotations.

### Target modes and scientific scope

- `process_classification` uses explicit JetClass process labels. It is useful for a separately selected process-classification task and pipeline smoke testing. It is not jet-flavour tagging.
- `flavour_tagging` requires verified per-jet light/charm/b annotations. The inspected ROOT schema does not provide those branches, so this mode raises an informative unsupported-target error.
- No code maps `label_QCD` to light flavour or treats Hbb/Hcc process labels as per-jet truth. WPCT and PI-WPCT are blocked for process-classification targets because their working point and risks depend on b efficiency and light/charm mistag semantics.

### Layer 2: Backbone Inference (`src/piwpct/backbones/`)
All backbones adhere to a uniform forward interface (with $K$ target classes):
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
- WPCT and PI-WPCT apply only to verified flavour targets; generic cross-entropy supports an explicitly selected process-classification task.

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
- Frozen model logits on the separate, held-out **calibration split** are used to fit exact class-conditional quantiles $\hat{q}_k$ for the configured target classes.
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
