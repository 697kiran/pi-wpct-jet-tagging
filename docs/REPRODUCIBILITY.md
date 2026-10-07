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

Verify the installation and run the test suite. Tests use tiny generated fixtures and do not read the supplied raw dataset:
```bash
pytest tests/ -v
```
The suite covers pilot/data-mode separation, feature and target contracts, generated ROOT/TAR conversion, memory-mapped loading, calibration consistency, and backbone forwards.

---

## 2. Step 1: Generate the Canonical 4-Way Data Split

The entire experimental framework relies on a single, versioned, content-hashed canonical 4-way split artifact (`train`, `validation`, `calibration`, `test`).

To generate the **synthetic pilot dataset** (12,000 generated examples, provisional light:b:c ratio ~4:1:1):
```bash
python scripts/build_splits.py --pilot --output-dir data/splits --seed 42
```

This generates:
- `data/splits/train.npz`
- `data/splits/validation.npz`
- `data/splits/calibration.npz`
- `data/splits/test.npz`
- `data/splits/canonical_split_manifest.json`

These examples are synthetic and are not read from the JetClass ROOT archives. `scripts/build_splits.py` requires the explicit `--pilot` flag and only creates pilot splits. Real ROOT conversion is provided separately by `scripts/convert_root.py`.

The manifest stores the SHA-256 hash of each split. Every subsequent script checks this manifest to guarantee zero split drift.

---

## 3. Step 2: Run Mandatory Pre-Training Diagnostic (§6 & §9.2)

Before executing large-scale training, run the pre-training risk diagnostic to evaluate $R_{\text{light}}(w)$, $R_{\text{charm}}(w)$, and their ratio across candidate working points ($50\%, 70\%, 77\%, 85\%$):

```bash
python scripts/run_pretraining_diagnostic.py --pilot --config configs/experiment.yaml
```

The `--pilot` flag is required to select synthetic data. If its manifest is missing, this explicit mode creates the synthetic pilot split. Real-data mode (`--real-data`) only accepts an existing prepared manifest marked `is_pilot: false` and never falls back to synthetic data. PI-WPCT diagnostics require verified light/charm/b labels; JetClass process labels cannot be used for that diagnostic.

Outputs:
- Terminal diagnostic table of mistag rates, risks, and lambda allocations.
- Allocation curve plot: `plots/pretraining_risk_diagnostic.png`.
- Automated test of the flat-risk null-result criterion (§9.3).

---

## 4. Step 3: Run a Single Regime Training and Evaluation

To train and evaluate an individual regime (e.g. `pi_wpct` on `pn_lite` at target WP $70\%$):

```bash
python scripts/run_regime.py --pilot --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
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
python scripts/run_full_ablation_matrix.py --pilot --backbone pn_lite --regimes baseline_ce wpct pi_wpct --epochs 3
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

---

## 7. Real ROOT Data: Safe Conversion and Training

Keep `RAW_DATA_ROOT` read-only. Choose a new prepared-data destination outside that root. Begin with a small, explicit conversion:

```bash
python scripts/convert_root.py --real-data \
  --raw-data-root /data/JetClass \
  --prepared-data-root /data/prepared/process-smoke \
  --target-mode process_classification \
  --max-files-per-partition 2 --max-jets-per-file 8 \
  --chunk-size 256 --particle-cap 128
```

The conversion command reports aggregate member, jet, class, malformed-row, truncation, size, and duration statistics. Prepared files are per-ROOT-member jagged NPY shards (`particles`, `offsets`, `jets`, `labels`, and `sample_ids`) with a portable manifest. The loader memory maps shards and constructs the padded batch tensors and constituent mask at collate time.

The training and calibration smoke on that bounded manifest is opt-in:

```bash
python scripts/run_regime.py --real-data \
  --prepared-data-root /data/prepared/process-smoke \
  --output-root /data/outputs/process-smoke \
  --target-mode process_classification --allow-incomplete-data \
  --regime baseline_ce --backbone pn_lite --epochs 1 \
  --batch-size 8 --workers 0
```

`--allow-incomplete-data` is only for an explicitly selected bounded smoke manifest. A full manifest does not need it. Full conversion has separate safeguards: it requires a prior sample manifest, an explicit output destination, `--full --confirm-full-conversion`, and at least 25% free-space headroom above the linear sample estimate. Do not use the full flag until the estimate and destination have been reviewed.

### Target semantics

`process_classification` uses ordered generated-process labels such as QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, and Tbl. It is an independent classification task. It does not imply light/charm/b jet flavour, and the code blocks WPCT/PI-WPCT for that target. `flavour_tagging` requires verified per-jet flavour branches; conversion fails clearly if they are absent. Do not map QCD to light or infer per-jet flavour from Hbb/Hcc process names.

The authoritative 17-channel candidate feature ordering and transformations are in `src/piwpct/schema.py`; four-vectors and the mask remain separate model inputs. The transforms were checked against the [official Particle Transformer JetClass feature configuration](https://github.com/jet-universe/particle_transformer/blob/main/data/JetClass/JetClass_full.yaml). Calibration and test logits receive the same fitted temperature before conformal scores are computed.

Portable roots resolve as CLI argument, config value, environment variable, then repository-relative default. Environment names are `RAW_DATA_ROOT`, `PREPARED_DATA_ROOT`, and `OUTPUT_ROOT`.

---

## 8. Astra

Use `hpc/astra/run_stage.py` through the documented **Jobs > Shell Script** interface. The wrapper supports capped conversion, smoke training, full conversion, and full training. Select one GPU card in Astra's UI for the initial training run. The supplied guide does not specify PBS directive syntax, CPU allocation syntax, or a wall-time value; select supported resources in the site UI rather than adding guessed directives. See `hpc/astra/README.md` for the tested Torch/CUDA installation order and wrapper commands.
