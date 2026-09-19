# Reproducibility Guide: PI-WPCT

This guide provides step-by-step instructions to reproduce all experiments, data splits, pre-training diagnostics, regime benchmarks, and ablation reports for the PI-WPCT project.

---

## 1. Environment Setup

Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/697kiran/pi-wpct-jet-tagging.git
cd pi-wpct-jet-tagging

# Create and activate a virtual environment (Python 3.8 - 3.11 recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package dependencies
pip install -e .[dev]
```

Verify the installation and run the test suite:
```bash
pytest tests/ -v
```
All 24 tests must pass, confirming split integrity, mathematical sensitivities, and Mondrian conformal coverage.

---

## 2. Step 1: Generate the Canonical 4-Way Data Split

The entire experimental framework relies on a single, versioned, content-hashed canonical 4-way split artifact (`train`, `validation`, `calibration`, `test`).

To generate the canonical pilot dataset (12,000 jets, provisional light:b:c ratio ~4:1:1):
```bash
python scripts/build_splits.py --pilot --output-dir data/splits --seed 42
```

This generates:
- `data/splits/train.npz`
- `data/splits/validation.npz`
- `data/splits/calibration.npz`
- `data/splits/test.npz`
- `data/splits/canonical_split_manifest.json`

The manifest stores the SHA-256 hash of each split. Every subsequent script checks this manifest to guarantee zero split drift.

---

## 3. Step 2: Run Mandatory Pre-Training Diagnostic (§6 & §9.2)

Before executing large-scale training, run the pre-training risk diagnostic to evaluate $R_{\text{light}}(w)$, $R_{\text{charm}}(w)$, and their ratio across candidate working points ($50\%, 70\%, 77\%, 85\%$):

```bash
python scripts/run_pretraining_diagnostic.py --config configs/experiment.yaml
```

Outputs:
- Terminal diagnostic table of mistag rates, risks, and lambda allocations.
- Allocation curve plot: `plots/pretraining_risk_diagnostic.png`.
- Automated test of the flat-risk null-result criterion (§9.3).

---

## 4. Step 3: Run a Single Regime Training and Evaluation

To train and evaluate an individual regime (e.g. `pi_wpct` on `pn_lite` at target WP $70\%$):

```bash
python scripts/run_regime.py --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
```

Supported regimes:
- `baseline_ce`: Standard Cross-Entropy with raw softmax threshold.
- `ce_temp_scaling`: Baseline CE with validation-fitted post-hoc temperature scaling.
- `ce_mondrian_cp`: Baseline CE with exact post-hoc Mondrian conformal calibration.
- `conftr`: Differentiable Conformal Training (Stutz et al., ICLR 2022).
- `wpct`: Working-Point Conformal Training with static background loss weights.
- `pi_wpct`: WPCT with periodic validation physics-impact risk allocation.

Outputs created under `outputs/pi_wpct_pn_lite_wp70_seed42/`:
- `checkpoint.pt`: Checkpoint with model weights, full config, git commit hash, seed, and split hash.
- `trajectories.jsonl`: JSON Lines log of validation risk refresh events.
- `lambda_trajectories.png`: Dynamic weight and purity evolution curves.
- `conformal_coverage.png`: Class-conditional Mondrian coverage plot.
- `robustness_curves.png`: Performance degradation curves under detector mismodellings.

---

## 5. Step 4: Run the Full Ablation Matrix

To run the complete benchmark comparison between regimes (e.g. `baseline_ce`, `wpct`, and `pi_wpct`) with paired bootstrap statistical uncertainty analysis:

```bash
python scripts/run_full_ablation_matrix.py --backbone pn_lite --regimes baseline_ce wpct pi_wpct --epochs 3
```

Outputs generated under `outputs/reports/`:
- `ablation_summary_report.md`: Markdown summary table with paired bootstrap CIs and automated failure/null-result flags.
- `ablation_summary_report.json`: Machine-readable structured metric data.

---

## 6. Seed and Parameter Configuration

All experimental hyperparameters are predeclared in versioned YAML configs:
- `configs/experiment.yaml`: Master seeds (`42, 101, 2024`), target working points (`0.50, 0.70, 0.77, 0.85`), evaluation settings.
- `configs/regimes/pi_wpct.yaml`: Refresh interval $K$, variance stabilization rules (`wilson_floor`), variance floor ($\sigma^2_{\min} = 10^{-7}$).
- `configs/data.yaml`: Split fractions (`0.60, 0.15, 0.10, 0.15`), physics yield priors.
- `configs/backbones/`: Network architectures for GBT, ParticleNet-Lite, LorentzNet-small, and ParT.
