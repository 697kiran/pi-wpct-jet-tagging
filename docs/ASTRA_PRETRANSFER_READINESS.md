# Final local Astra pre-transfer readiness audit

**Audit date:** 2026-10-09
**Repository:** local working tree
**Scope:** local source, configuration, packaging, data metadata, wrapper, outputs, and transfer-package checks. Astra was not accessed.

## Executive finding

The repository is structurally prepared for a **source-only transfer package**, including uncommitted and untracked Phase 1-5 work. This does not establish Astra runtime compatibility. The configured job example remains disabled. No experiment, real ROOT conversion, transfer, dependency installation, GPU check, or scheduler action was performed.

Real-data scientific readiness is blocked: the local JetClass archives describe generated-process labels, the bounded prepared process dataset is incomplete, the tracked synthetic pilot manifest references missing split files, and no verified per-jet light/charm/b target dataset is available.

## Repository architecture and proposed project placement

The scientific Python package is under `src/piwpct/`, with backbones, evaluation, and utility modules. Execution scripts are in `scripts/`; YAML configuration is in `configs/`; Astra wrapper and stage runner are in `run_on_astra/` and `hpc/astra/`. Tests and detailed documentation are kept in `tests/` and `docs/`.

Illustrative future project layout:

```text
/home/<username>/pi-wpct-jet-tagging/
├── configs/
├── data/
│   ├── raw/                 # created/staged deliberately; not in source package
│   ├── prepared/            # created/staged deliberately; not in source package
│   └── splits/              # no runnable pilot arrays currently present
├── docs/
├── hpc/astra/
├── run_on_astra/
│   ├── submit_stage.py
│   └── job_config.example.json  # disabled template only
├── scripts/
├── src/piwpct/
├── tests/
├── env/                     # create on Astra; never transfer a Windows environment
├── outputs/                 # persistent output root, configured when ready
└── requirements.txt
```

The source archive does not preserve empty raw/prepared directories. Create those directories on the destination only when staging data. Do not copy scheduler-generated staging folders as project roots.

Optional/non-scientific components omitted from the source package: `api/`, `ui/`, `Flow/`, historical `outputs/`, and `plots/`. The API is an optional prototype; the UI and Flow images are not required by `piwpct` package discovery or the scientific entry points. Historical outputs remain in the local repository and are not migrated.

## Source-only package inventory

Run `python scripts/package_astra_source.py` to rebuild the source package from the current working tree. The utility writes the ZIP, the source inventory (`dist/astra/source_manifest.json`), and the ZIP checksum (`dist/astra/pi-wpct-jet-tagging-source.zip.sha256`) beneath `dist/astra/`. The existing `.gitignore` excludes `dist/`. The allowlisted inputs never scan `dist/`, so generated outputs cannot enter their own ZIP. The accompanying source manifest records each relative path, file size, and SHA-256; it also records the ZIP size and checksum. These sidecars and ZIP are local handoff artifacts and are not included inside the ZIP.

The package includes:

- Root project metadata and instructions: `.gitignore`, `LICENSE`, `README.md`, `DESIGN.md`, `USER_MANUAL.md`, `pyproject.toml`, `requirements.txt`.
- All `src/piwpct/**` Python package files.
- All `scripts/**` Python entry points and local readiness/conversion utilities.
- All `configs/**` YAML files.
- `hpc/astra/README.md`, `hpc/astra/run_stage.py`.
- `run_on_astra/submit_stage.py` and the disabled `job_config.example.json` only.
- All `tests/**` test sources.
- All `docs/**` Markdown/JSON documentation, including this report and the sanitized transfer-plan template.
- `data/README.md` only.

The archive excludes `.git`, caches, virtual environments, `run_on_astra/job_config.json`, `data/transfer_manifest.local.json`, raw archives, prepared NPY/NPZ arrays, local ROOT staging, checkpoints, generated logs, `outputs/`, `plots/`, `api/`, `ui/`, and `Flow/`. Git currently has 163 tracked paths; the working tree includes required untracked additions. The source archive is assembled from explicit allowlisted folders and files, so ignored and untracked status does not cause required source to be dropped.

Raw-data directories are external inputs and must be created/staged deliberately. Conversion requires the configured raw directory to exist. The conversion stage requires the prepared destination itself not to exist and its parent directory to exist; create that parent deliberately before conversion. Training run/output directories are created by `create_run_directory()` beneath the configured output root. The package builder creates `dist/astra/` when run.

## Packaging and dependency contract

`pyproject.toml` uses setuptools package discovery from `src/`; `src/piwpct/__init__.py` exists and reports version `0.1.0`. `requirements.txt` delegates installation to `-e .`. Core project dependencies include NumPy, SciPy, scikit-learn, Matplotlib, PyYAML, uproot, and awkward. PyTorch is optional in package metadata and is deliberately not installed by `requirements.txt`; GBT, API, and development dependencies are optional extras.

Documented future sequence: transfer source; create the Astra virtual environment; install ordinary dependencies; install a separately selected CUDA-compatible PyTorch wheel; verify imports and `pip check`; verify GPU only within an allocated GPU job. No package was installed during this audit. Dependency lower bounds are not a lockfile, Python support is declared broadly (`>=3.8`), and compatibility with Astra's actual interpreter, package index, driver, and CUDA build remains unverified. The Astra guide's example Torch/CUDA versions are references, not a locally verified runtime result.

## Astra guide comparison

The attached guide's relevant instructions are: keep project files under `/home/<username>/`; do not use `/PBSHOME/stage/<username>/` as the persistent project/data root; use an Astra-created environment and its `env/bin/python`; use absolute wrapper paths; select queue/GPU resources in the GUI; verify CUDA from within a scheduled GPU job; and keep large-file transfer coordinated through ICTS. It describes direct upload for files smaller than 5 GB and ICTS external-drive transfer for files larger than 5 GB. It does not define a byte convention at exactly 5 GB, so the checker flags exactly 5,000,000,000 bytes for manual confirmation.

The guide's `Project1/content/files/` is an example layout only. This project keeps its `src/piwpct/` architecture. The guide documents an A100/gpuq workflow; no such host or allocation was observed. It does not establish this repository's runtime compatibility.

## Wrapper and stage-runner integration

Verified local flow:

```text
GUI selects run_on_astra/submit_stage.py
  -> wrapper selects <project_root>/env/bin/python by default
  -> hpc/astra/run_stage.py receives absolute project/data/output paths
  -> convert-smoke / convert-full: scripts/convert_root.py
  -> train-smoke / train-full: scripts/run_regime.py
  -> package modules under src/piwpct/
```

The implemented stage names are `convert-smoke`, `convert-full`, `train-smoke`, and `train-full`. The wrapper uses repository-root-relative path resolution, preserves target/data-mode checks, gates full conversion with explicit size-estimate/confirmation settings, propagates child failure, and does not create an enabled account-specific config. The project interpreter default follows the guide. Local tests verify configuration failure paths, command forwarding, and dry-run behavior. Dry-run does not execute the child runner or create scientific output. Actual GUI selection, working directory on the cluster, and scheduler behavior remain untested.

## Data and transfer readiness

The read-only preflight found:

| Resource | Observed local status | Interpretation |
|---|---|---|
| Raw JetClass process archives | 12 files in the audited local inventory, 189,839,872,000 bytes (189.840 GB / 176.802 GiB); file stats and TAR header checksums valid | Present locally, not transferred. Full archive hashes were not computed. Each file exceeds the guide's 5 GB boundary. |
| Prepared process smoke | External manifest and 31 files, 201,715 bytes, 72 jets | `complete: false`, target `process_classification`; structurally useful bounded smoke input only. |
| Tracked synthetic pilot | `data/splits/canonical_split_manifest.json` | Missing four referenced NPZ files; not runnable as checked out. |
| Verified flavour data | Not found | No verified per-jet light/charm/b labels; process classes are not substitutes. |
| Astra destinations, quota, permissions | Not checked | Requires authorized Astra verification. |

The raw archives must not be described as prepared data. The process smoke labels do not establish flavour labels. The local-only transfer inventory has private absolute paths and is Git-ignored; it is not included. The sanitized template contains no actual local source path or account name. The prepared smoke data is optional and separate from the raw archives; do not transfer both large raw and full prepared data redundantly without a deliberate storage plan.

## Output and logging contract

The wrapper resolves one **base output root** and forwards it unchanged. Training allocates `<output-root>/runs/<run-id>`; the stage runner does not add another `runs/` layer. A run contains metadata, checkpoint, trajectory JSONL, epoch/final metrics, plots, and application log. Explicit run ID collisions fail. Legacy tracked outputs are left intact and excluded from the transfer package. Application console output remains available to scheduler capture; run-specific application logs remain separate from scheduler `.o`/`.e` files.

## Local verification

- Focused integration suite: **96 passed, 0 failed, 0 skipped** (all repository tests except `tests/test_root_pipeline.py`).
- `tests/test_root_pipeline.py` was not executed because it invokes ROOT/TAR conversion logic; Phase 6 prohibits ROOT conversion. This omission leaves actual conversion integration unverified here.
- The readiness checker was run on the local raw archive directory, external prepared smoke manifest, and tracked pilot manifest. Results are recorded above.
- The source package is re-created by the standard-library packaging utility, which validates the archive member list, each packaged-file checksum, the ZIP integrity, and the ZIP SHA-256.
- Astra guide text was locally extracted from its 23-page PDF. `py_compile`, JSON/YAML parsing, package structure inspection, and `git diff --check` are part of final validation.
- No training, actual ROOT conversion, GPU operation, installation, scheduler submission, login, or transfer was performed.

### Phase 6.1 source-package freeze

The final durable package contains 82 files and is stored at `dist/astra/pi-wpct-jet-tagging-source.zip`. Its byte size, SHA-256, and per-source-file inventory are recorded in the adjacent ignored `.sha256` and `source_manifest.json` files. These checksums cover the source ZIP and packaged source files only; the 12 JetClass archive contents have not been fully hashed.

The ZIP was extracted to a temporary local directory and verified against every manifest path, size, and digest. Required Python/YAML/configuration files and case-sensitive script names were present; excluded data and account-specific files were absent. From an unrelated current directory, the extracted project-root path resolver selected the extracted tree. The extracted wrapper's `--help` succeeded, invocation without a live config failed closed with status 2, and a temporary smoke config produced a dry-run command preview without creating `outputs/`. The temporary extraction and test config were removed. No scientific command ran.

## Independent readiness verdicts

| Area | Verdict | Evidence / limit |
|---|---|---|
| Source and directory readiness | **READY LOCALLY** | Required package, configs, scripts, wrapper, tests and docs are in the allowlisted source archive, including untracked files. |
| Transfer-plan readiness | **PARTIALLY READY** | Source inventory and local raw archive inventory exist. Transfer is pending; exact host destination/storage and ICTS intake remain unknown. |
| Dataset readiness | **BLOCKED** | Process smoke is incomplete, pilot arrays are missing, and verified flavour labels are absent. |
| Astra runtime readiness | **REQUIRES ASTRA VERIFICATION** | Python, CUDA/PyTorch, storage, permissions, GUI, scheduler and GPU were not checked on Astra. |
| Full real-data experiment readiness | **BLOCKED** | No complete prepared flavour dataset with verified target semantics. |

## Post-login checklist (not performed)

- [ ] Confirm actual persistent home/project path and avoid `/PBSHOME/stage/<username>/` for project data.
- [ ] Confirm directory permissions, quota, free space, and conversion staging capacity.
- [ ] Confirm available Python version against package requirements.
- [ ] Create the project virtual environment and install approved ordinary dependencies.
- [ ] Install the selected CUDA-compatible PyTorch build separately; verify package imports and `pip check`.
- [ ] Verify CUDA/PyTorch and visible GPU inside a GPU-allocated job only.
- [ ] Confirm GUI wrapper selection, queue/resource settings, scheduler working directory, and failure propagation.
- [ ] Transfer source and any deliberately selected datasets; verify file integrity after transfer.
- [ ] Check prepared dataset completeness and target metadata; verify real flavour labels before flavour training.
- [ ] Run a controlled, authorized smoke test before considering a full experiment.
