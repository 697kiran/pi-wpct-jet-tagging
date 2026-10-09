# Data directory contract

Large inputs and generated data are external to Git. Keep raw JetClass archives in their existing local location; configure `RAW_DATA_ROOT` or pass `--raw-data-root` when converting. The repository's default `data/raw` path is only a fallback, not a claim that raw data is present.

Recommended project-relative destinations when data is deliberately staged:

```text
data/
├── raw/
│   ├── jetclass-process/       # JetClass TAR files; process labels only
│   └── jetclass-flavour/       # reserved; requires a verified per-jet flavour source
├── prepared/
│   ├── process/                # piwpct.prepared_jagged.v1 process-classification output
│   └── flavour/                 # only after verified light/charm/b labels are supplied
└── splits/
    └── canonical_split_manifest.json  # legacy synthetic pilot metadata; incomplete here
```

Empty raw/prepared folders are not committed. Do not place the tracked legacy pilot manifest in a real-data role: its four referenced NPZ files are absent. New pilot manifests are separate, explicit synthetic artifacts. See [data readiness and transfer](../docs/DATA_READINESS.md).
