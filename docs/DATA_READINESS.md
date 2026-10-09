# Data readiness and Astra transfer plan

## Data flows and path rules

Real process-classification flow: external `JetClass_*.tar` archives → `scripts/convert_root.py` streams ROOT members one at a time → NPY shards plus a `piwpct.prepared_jagged.v1` manifest → `PreparedShardedJetDataset` memory-maps the NPY arrays → real-data training. Conversion writes to a fresh prepared destination, stages a ROOT member temporarily beside that destination, retains the raw TAR files, and does not currently resume a partial conversion. Full conversion performs a sample-size/free-space check first; the estimate is a projection, not a quota guarantee.

Synthetic pilot flow is separate: explicit `--pilot` creates four NPZ split files and a pilot manifest. The tracked `data/splits/canonical_split_manifest.json` is an older synthetic pilot manifest whose `data/splits/train.npz`, `validation.npz`, `calibration.npz`, and `test.npz` references do not exist in this checkout. It is not deployable or usable for training. Rebuild a pilot deliberately when needed; do not transfer this stale manifest as runnable data.

CLI paths override YAML, then environment variables, then repository-relative defaults (`data/raw`, `data/prepared`, `outputs`). Relative repository paths resolve from the repository root. Prepared shard references resolve from the directory containing the manifest, so transferring a prepared dataset requires preserving its manifest and `shards/` tree together. The user-facing Astra wrapper requires explicit paths and uses the project interpreter; see [Astra setup](../hpc/astra/README.md).

## Dataset and target separation

| Data kind | Target meaning | Current status | Allowed interpretation |
|---|---|---|---|
| Synthetic pilot | Simulated light/charm/b labels | Generator exists; tracked old manifest references missing NPZ files | Software diagnostics only |
| JetClass process classification | Original generated-process classes such as QCD, Hbb, Hcc, Hgg | 12 local TAR archives and a 72-jet bounded prepared smoke set | Separate process-classification objective; not flavour evidence |
| Verified flavour tagging | Per-jet light/charm/b annotation from a verified source | Not available in inspected local files/configuration | Required before the PI-WPCT flavour experiment |

JetClass process classes (including `Hbb` and `Hcc`) must not be relabeled as individual jet flavours. The current prepared smoke manifest says `process_classification`, `complete: false`, `is_pilot: false`, and `conversion_scope: bounded-smoke-sample`; it is structurally useful for bounded loader checks, not full training.

## Prepared manifest and shard contract

The real prepared contract is `format: piwpct.prepared_jagged.v1`, `format_version: 1`, explicit `complete`, explicit `target_definition.mode`, split names `train`, `validation`, `calibration`, `test`, and per-shard relative `path`, `n_jets`, and provenance. Each shard directory contains `particles.npy` (`[total_constituents, 16]`, float32), `offsets.npy` (`[n_jets + 1]`, int64), `jets.npy` (`[n_jets, 10]`, float32), `labels.npy` (`[n_jets]`, int64), and `sample_ids.npy` (`[n_jets, 2]`, uint64). The loader memory-maps arrays and checks lengths and offsets when opening shards. Structural readiness checks verify metadata and path existence without loading array contents. Hashes in the existing contract are optional; the preflight does not hash data. For complete integrity verification, calculate SHA-256 once after staging and compare source/destination manifests; do not infer archive integrity from a TAR header.

## Local read-only checks

From the repository root:

```powershell
python scripts/check_data_readiness.py --raw-root <LOCAL_RAW_ARCHIVE_DIRECTORY>
python scripts/check_data_readiness.py --manifest <PREPARED_SMOKE_MANIFEST_PATH> --stage train-smoke --data-mode real --target-mode process_classification
python scripts/check_data_readiness.py --manifest data\splits\canonical_split_manifest.json --stage train-full --data-mode pilot
```

The checker only stats archive files and reads their first 512 bytes for a TAR header checksum. For prepared manifests it reads JSON and checks contained references and required files; it does not open NPY arrays, extract TAR contents, hash datasets, or alter files. Statuses include `PRESENT_AND_STRUCTURALLY_VALID`, `PRESENT_BUT_INCOMPLETE`, `MISSING`, `TARGET_NOT_VERIFIED`, and `REQUIRES_ASTRA_CHECK`. A successful local check cannot confirm Astra paths, permissions, storage capacity, Python/CUDA availability, scheduler setup, or transfer integrity. `train-full` exits unsuccessfully if its manifest is missing, incomplete, or target-mismatched.

## Astra layout and transfer mapping

The guide recommends persistent project files under `/home/<username>` and explicitly warns against using `/PBSHOME/stage/<username>` as the project/data root. The planned root is `/home/<username>/pi-wpct-jet-tagging/`; the username and Astra storage capacity remain to be confirmed after authorized access. The guide's threshold is stated in GB (not GiB): files smaller than 5 GB use Astra's upload interface, and files larger than 5 GB use ICTS-assisted external-drive transfer. It does not specify a byte convention for the threshold; exactly 5,000,000,000 bytes is conservatively flagged for confirmation. Individual large files require the exact source path and absolute Astra destination to be supplied through a faculty member to ICTS System Care. No transfer is initiated here.

The operator-specific inventory is `data/transfer_manifest.local.json` (ignored by Git) and contains actual source paths and sizes. The local inventory audited for this report records 12 selected raw archives totaling 189,839,872,000 bytes (189.840 decimal GB; 176.802 GiB), plus an extra same-name/same-size validation-archive candidate found elsewhere that has not been content-verified and is excluded from transfer. The prepared smoke tree is 201,715 bytes across 31 files. Destination mapping:

| Source group | Future Astra destination | Handling |
|---|---|---|
| Repository source tree (excluding `.git`, datasets, outputs, caches) | `/home/<username>/pi-wpct-jet-tagging/` | Upload source files; stale pilot manifest is not runnable |
| Local `JetClass_Pythia_*.tar` archive directory (12 files; each >5 GB) | `/home/<username>/pi-wpct-jet-tagging/data/raw/jetclass-process/` | ICTS external-drive procedure per guide; move one raw copy only |
| Local prepared process smoke tree (72 jets) | `/home/<username>/pi-wpct-jet-tagging/data/prepared/process/smoke-72/` | Optional small direct upload (<5 GB total); preserve manifest/shards together |
| Verified per-jet flavour source | Not available | No transfer mapping until source and semantics are verified |

Choose either raw archives plus conversion on Astra or the bounded prepared smoke dataset for a smoke check. Avoid transferring redundant full raw and full prepared copies unless capacity and provenance justify it. Do not place content in scheduler staging folders. The exact destination filesystem, quota, free space, and ICTS intake details are Astra-side checks still pending.

## Storage and staging

Measured raw archive total is in the local inventory (bytes are authoritative; displayed GiB is base-2). The previous bounded smoke conversion reported about 2.59 kB per jet; linear extrapolation to the 125 million jets represented by the train/validation/test archive names is roughly 301 GiB and is only an estimate. Actual output depends on schema and conversion parameters. During full conversion, budget additional temporary space: a ROOT member is staged while its shard is written to a fresh `.building-*` directory, then the output is atomically renamed. Raw archives remain in place. Conversion is streamed by ROOT member and chunk, but partial output is not resumable. Astra quota, staging capacity, and available space cannot be established locally; run the built-in estimate/check only after Astra storage is confirmed.

## Astra-only checks after access

Confirm the actual persistent home path, project interpreter and dependencies, directory permissions, archive transfer checksums, free space/quota for raw + prepared + temporary staging, and scheduler/GPU environment. Keep `run_on_astra/job_config.json` disabled until each configured path and mode is real and verified. Full training requires `complete: true`; PI-WPCT requires a verified `flavour_tagging` manifest. Output/checkpoint/trajectory run isolation is implemented in Phase 5 and documented in [experiment outputs](EXPERIMENT_OUTPUTS.md).
