# Astra execution

The supplied Astra guide documents job submission through **Jobs > Shell Script**. It sets the `gpuq` queue, enables GPU use, and requests one GPU card for an ordinary single-GPU program. The guide retains the GUI memory default. It does not document PBS directive syntax, CPU resource syntax, or a wall-time setting, so this repository does not guess those scheduler directives.

Use [`run_stage.py`](run_stage.py) as the Python scheduler wrapper in the Astra UI. Set `RAW_DATA_ROOT`, `PREPARED_DATA_ROOT`, and `OUTPUT_ROOT` in the job environment or pass their CLI overrides. Use a project virtual-environment interpreter built from the non-PyTorch project dependencies, then install the guide's tested PyTorch CUDA 11.8 build separately:

```bash
python -m pip install -e .
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

The wrapper exposes four stages:

- `convert-smoke`: capped TAR/ROOT conversion; defaults to 2 files per official partition and 8 jets per member.
- `train-smoke`: one epoch on a bounded manifest; explicitly permits incomplete smoke data.
- `convert-full`: requires a prior sample manifest, an available-space estimate, and explicit full-conversion confirmation.
- `train-full`: uses the selected prepared manifest and regime. A flavour-specific WPCT run still requires verified per-jet flavour labels.

Example GUI-submitted wrapper commands:

```bash
python hpc/astra/run_stage.py convert-smoke \
  --project-root "$PROJECT_ROOT" --python "$PROJECT_ROOT/env/bin/python" \
  --target-mode process_classification

python hpc/astra/run_stage.py train-smoke \
  --project-root "$PROJECT_ROOT" --python "$PROJECT_ROOT/env/bin/python" \
  --target-mode process_classification
```

The process-classification examples are a separately selected JetClass task. They do not produce light/charm/b flavour-tagging results. The full-conversion stage is not launched by this project setup or its tests.

The Astra UI is where GPU count, queue, and site-supported CPU/memory/wall-time values must be selected. Start with one A100 for training. The number of DataLoader workers is independently configurable with `--workers`; it does not claim or request scheduler CPU allocation.
