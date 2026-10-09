# Astra environment preparation (not yet executed)

The guide's persistent workspace is `/home/<username>/`, not `/PBSHOME/stage/<username>/`. Keep the existing repository layout under `/home/<username>/pi-wpct-jet-tagging/`. Create `env/` on Astra; do not transfer a Windows virtual environment. The exact Astra Python, driver, package-index access, and CUDA compatibility must be checked after login. No commands in this document have been run on Astra.

## Intended installation sequence

Replace `<username>` with the actual account name before running these commands. After confirming that the chosen Astra Python supports the required packages and selected PyTorch wheel:

```bash
cd /home/<username>/pi-wpct-jet-tagging
python3 --version
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` contains `-e .` and delegates non-PyTorch dependency versions to `pyproject.toml`. This installs the `src/piwpct` package in editable mode, making it importable from a scheduler-generated working directory. It includes NumPy, SciPy, scikit-learn, Matplotlib, PyYAML, uproot, and awkward. The last two are required for ROOT conversion and its corresponding tests. It does not install PyTorch, GBT backends, the optional API, or the Next.js UI.

If needed, install optional Python extras **before** the CUDA build: `python -m pip install -e ".[gbt]"` for XGBoost and LightGBM, or `python -m pip install -e ".[api]"` for the prototype FastAPI service. These extras add their specialized libraries; the API and GBT model code also use the separately installed Torch package. Neither extra is needed for the ordinary ParticleNet-Lite training installation. The generic `.[training]` extra is for local development and must not be used as the Astra CUDA installation step. Do not use `--no-deps` with the ordinary installation: it would omit required non-PyTorch packages.

The guide's *reference* environment uses PyTorch 2.5.1 with CUDA 11.8 wheels. Only after verifying that this combination fits the actual Astra Python and driver, install the project-required PyTorch package separately:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

The guide also lists `torchvision==0.20.1` and `torchaudio==2.5.1`, but this repository imports neither. Install them only if a later workload actually needs them, using versions compatible with the selected Torch build. Installing ordinary requirements first and avoiding `.[training]` on Astra prevents a later generic Torch resolution from replacing the selected CUDA build.

After installation, verify without starting a training run:

```bash
python -c "import piwpct, torch, numpy, scipy, sklearn, yaml, uproot, awkward, matplotlib; from piwpct.backbones import build_backbone; print('imports OK', torch.__version__, torch.version.cuda)"
python -m pip check
python -m pip freeze
```

`pip check` reports declared dependency conflicts; the import check confirms that training and conversion modules are available. Record the resolved package versions from `pip freeze` with the Astra Python version once the environment has been validated; the current lower bounds are not a cluster-specific lockfile. GPU visibility must be checked **later inside a scheduled GPU job**, for example with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`. A setup or login terminal is not a reliable GPU allocation test. Record the actual `torch.__version__` and `torch.version.cuda` values.

## Scheduler wrapper and deployment config

[`run_on_astra/submit_stage.py`](../../run_on_astra/submit_stage.py) is the Python file to select through Astra's **Jobs > Shell Script** flow after the deployment config and environment are ready. Astra's GUI selects queue/GPU resources. The repository does not add PBS directives or assume undocumented CPU, memory, or wall-time settings.

The GUI can select the Python file without supplying command-line arguments. The wrapper locates the repository from its own file and reads `run_on_astra/job_config.json` beside itself. Copy `run_on_astra/job_config.example.json` to that name and complete it deliberately. The example is disabled, incomplete, and cannot launch work. Set `enabled` to `true` only after reviewing the paths, data mode, stage, and resource requirements. To preview locally, run `python run_on_astra/submit_stage.py --dry-run` from any directory after completing a config; preview prints the absolute command but does not check that future Astra data paths exist or start a child process. Importing the file also does not run a stage.

Config fields:

- `enabled`: required literal `true` before preview or execution.
- `project_root`: optional absolute path or project-relative override; null derives the repository root from the wrapper location.
- `python`: optional interpreter override; null selects `<project_root>/env/bin/python`.
- `stage`: required: `convert-smoke`, `convert-full`, `train-smoke`, or `train-full`.
- `data_mode`: required `real` or `pilot`. Pilot is supported only for `train-smoke` with an explicit pilot manifest. ROOT conversion and full training require `real`.
- `target_mode`: required for real data: `process_classification` or `flavour_tagging`; the selected manifest must match. WPCT/PI-WPCT cannot run against process-classification labels.
- `raw_data_root`: required only for conversion; point it at the read-only archive directory.
- `prepared_data_root`: required for conversion destinations and real-data training; it is not needed for pilot training.
- `output_root`: required for training and is the persistent **base output root**. The scientific entry point creates one unique `runs/<run-id>/` directory below it for checkpoints, metrics, figures, reports, and application logs. Conversion writes prepared shards to `prepared_data_root` and does not need a training output root.
- `run_id`: optional explicit ID for a training run. It is forwarded unchanged to the scientific entry point; reusing an existing ID fails before training.
- `split_manifest`: for real training, optional because it defaults to `<prepared_data_root>/canonical_split_manifest.json`; an explicit path must match `target_mode`. Pilot smoke training requires a manifest marked as pilot.
- `target_config`: optional class-to-branch mapping for conversion.
- `size_estimate_manifest` and `confirm_full_conversion`: required for `convert-full`; review the size estimate and set confirmation true deliberately.
- `working_point`, `seed`: training selections forwarded to `run_regime.py`.
- `regime`, `backbone`, `epochs`: full-training selections. Smoke training is fixed to one epoch of `baseline_ce`.
- `workers`, `batch_size`, `particle_cap`, `pin_memory`: training loader options.
- `conversion`: conversion tuning. `max_files_per_partition` (at least two) and `max_jets_per_file` apply only to `convert-smoke`; `chunk_size`, `particle_cap`, `calibration_fraction`, and top-level `seed` apply to both conversion stages. Full conversion never receives sample caps.

All configured relative paths resolve from the project root, never the scheduler's generated working directory. The wrapper passes the absolute base output root and optional run ID through `hpc/astra/run_stage.py`; only `train_pipeline` creates the run directory. Standard output and standard error remain attached to the scheduler job stream. Application logs are written inside each run; scheduler `.o`/`.e` files remain managed by Astra. `--dry-run` previews the command without creating a run directory. See `docs/EXPERIMENT_OUTPUTS.md` for the full layout. The account-specific `job_config.json` is ignored by Git; keep only sanitized templates in version control.

The stage runner exposes four supported stages:

- `convert-smoke`: capped TAR/ROOT conversion; defaults to 2 files per official partition and 8 jets per member.
- `train-smoke`: one epoch on a bounded manifest; explicitly permits incomplete smoke data.
- `convert-full`: requires a prior sample manifest, an available-space estimate, and explicit full-conversion confirmation.
- `train-full`: uses the selected manifest and regime. A flavour-specific WPCT run requires verified per-jet flavour labels; runtime manifest checks prevent process labels from being treated as flavour labels.

A process-classification run is a separately selected JetClass task. It does not produce light/charm/b flavour-tagging results. The full-conversion stage is not launched by this project setup or its tests.

`train-smoke`/`train-full` may use an allocated GPU, while conversion does not require one. The DataLoader `workers` setting does not claim scheduler CPU resources. No stage was executed on Astra. Live paths, interpreter availability, filesystem permissions, job working directory, GPU allocation, queue configuration, and actual CUDA visibility can only be confirmed after login and a permitted job submission. Do not enable conversion until archive contents, label branches, a fresh prepared destination, and (for full conversion) the sample size estimate and free space are verified. Do not enable training until the manifest mode and target semantics are verified; full training also requires a complete manifest. Run isolation was implemented in Phase 5; see [experiment output layout](../../docs/EXPERIMENT_OUTPUTS.md).
