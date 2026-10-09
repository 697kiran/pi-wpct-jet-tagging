# Experiment output layout

## Path contract and ownership

`output_root` is the **base output root**, not a specific run directory. It resolves as CLI `--output-root`, `configs/data.yaml` `data.output_root`, `OUTPUT_ROOT`, then repository-relative `outputs`. Relative repository paths resolve from the repository root, independent of the process or scheduler working directory. The Astra wrapper resolves its configured output root to an absolute path and the stage runner forwards that same base path. Neither wrapper creates a run directory.

`train_pipeline` is the sole owner of training-run allocation. It atomically reserves a new directory beneath `<output_root>/runs/` before training begins. Single-regime execution uses one run directory. A pretraining diagnostic gets its own directory. An ablation creates one suite directory, then one child run per currently executed regime under the suite's `runs/` subdirectory; the suite's aggregate report goes under `reports/`.

```text
outputs/
└── runs/
    ├── training-pi-wpct-pn-lite-wp70-seed42-<UTC>-<uuid>/
    │   ├── metadata.json
    │   ├── checkpoints/checkpoint.pt       # GBT: checkpoints/gbt_model.pkl
    │   ├── trajectories/trajectories.jsonl
    │   ├── metrics/
    │   │   ├── epochs.jsonl
    │   │   ├── final_metrics.json
    │   │   └── conformal_calibration.json
    │   ├── plots/
    │   └── logs/application.log
    └── ablation-<UTC>-<uuid>/
        ├── metadata.json
        ├── runs/<child-run-id>/...          # isolated training child runs
        └── reports/ablation_summary_report.{md,json}
```

Run directories are created with exclusive `mkdir`; concurrent collisions cannot reuse the same directory. Generated IDs combine run type and available scientific parameters with a UTC timestamp and UUID suffix. `--run-id` accepts a path-free ID of up to 96 letters, digits, dots, underscores, or hyphens. Reusing an existing ID raises an error; there is no resume or append mode. Choose a new ID for every execution. Run IDs apply to the single run, diagnostic, or ablation suite; ablation child runs receive their own generated IDs.

## Artifacts and metadata

- PyTorch checkpoints are saved to `checkpoints/checkpoint.pt` via a temporary file and atomic replace. They contain model weights, regime/backbone, working point, seed, epoch count, best epoch/loss, config, git commit, manifest hash, target mode, class names, temperature, trajectories, and run ID. The current pipeline saves the best model state selected by its existing validation-loss comparison. It does not save optimizer state and does not implement resume.
- GBT fits save to `checkpoints/gbt_model.pkl` using the existing GBT serialization format.
- PI-WPCT validation-risk refresh records go to the exclusive, run-local `trajectories/trajectories.jsonl`. Each record includes the run ID, UTC timestamp, and epoch (including the initial refresh at epoch 0). Each write is flushed. A preexisting trajectory file causes an error instead of append.
- Per-epoch train/validation losses go to flushed `metrics/epochs.jsonl`. Single-regime final test and robustness metrics plus conformal quantiles/calibration counts go to `metrics/`. Ablation child metrics/calibration go in each child run; aggregate Markdown/JSON reports go in the suite's `reports/` folder. Diagnostic risk arrays and figures are stored in the diagnostic run.
- Figures go under `plots/`. Training application logs go to `logs/application.log` and remain on the console. Astra's scheduler `.o`/`.e` logs remain managed by Astra and are not redirected or replaced.
- `metadata.json` records the run ID and type, UTC creation/update times, status, actual training selections, effective config, dataset mode/target and manifest identifier/hash, output paths, project version when installed, git commit, and a boolean indicating uncommitted Git changes when available. It records no credentials, private environment variables, or dataset-content hashes.

Example metadata fields:

```json
{
  "schema": "piwpct.run_metadata.v1",
  "run_id": "training-baseline-ce-pn-lite-wp70-seed42-<UTC>-<uuid>",
  "status": "completed",
  "regime": "baseline_ce",
  "backbone": "pn_lite",
  "target_wp": 0.7,
  "seed": 42,
  "dataset_mode": "real",
  "target_mode": "flavour_tagging",
  "manifest_name": "canonical_split_manifest.json",
  "manifest_sha256": "<hash of manifest file only>",
  "effective_training_config": {"epochs": 10},
  "git_commit": "<commit or unknown>",
  "uncommitted_changes": true
}
```

Run metadata starts as `running`; training records `training_complete`, and the entry-point script marks it `completed` after its evaluation/report artifacts are written. Training exceptions mark the run `failed`. An interrupted process can leave `running` metadata; inspect it and start a fresh run rather than reusing it.

## Entry points

```bash
python scripts/run_regime.py --pilot --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
python scripts/run_regime.py --pilot --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --run-id pilot-pi-wpct-repeat-01
python scripts/run_pretraining_diagnostic.py --pilot
python scripts/run_full_ablation_matrix.py --pilot --backbone pn_lite --regimes baseline_ce wpct pi_wpct --epochs 3
```

Without `--run-id`, repeated commands produce separate directories. With an explicit ID, a second command using the same base root and ID fails before training. Astra's `run_on_astra/job_config.json` accepts an optional `run_id` for training and forwards it through `submit_stage.py` and `hpc/astra/run_stage.py` to `run_regime.py`.

`--dry-run` on the Astra wrapper only prints the command. It does not run the stage runner and does not create a run directory, metadata, or artifacts. To locate a result after a future job, use the printed run ID and inspect `<output_root>/runs/<run-id>/metadata.json`; for ablations, follow `child_run_directories` in the suite metadata.

The deprecated diagnostic `--output-plot` option may change the plot filename, but the plot stays under that diagnostic's `plots/` directory. Existing tracked results under `outputs/pi_wpct_pn_lite_wp70_seed42/`, `outputs/reports/`, and `plots/` remain untouched and readable. The diagnostic `--checkpoint` accepts existing legacy checkpoint paths as well as paths in the new `checkpoints/` subdirectories.

## Scientific review note

Checkpoint-selection behavior is unchanged: the best checkpoint is selected by the lowest `val_loss`. In PI-WPCT, dynamic loss weights are updated across training, so validation losses observed under different weight states may not be directly comparable. This is a scientific objective-selection question; Phase 5 records the selected epoch/loss but does not change the criterion.
