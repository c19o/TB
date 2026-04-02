# Cloud 1W Profile

Date: 2026-04-02

Maintained authority:
- `deploy_1w.sh`
- `deploy_tf.py --tf 1w`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`

Profile summary:
- Warm-start parent: none
- Execution mode: `cpu_first_trimmed`
- Cross policy: forbidden
- Same-machine required: no

Machine policy:
- Hard floor: `16` CPU cores, `64 GB` RAM, `1` GPU
- Preferred: `32` CPU cores, `128 GB` RAM, `1` GPU optional

Maintained caveats:
- `1w` is the trimmed weekly path.
- Cross generation is intentionally skipped.
- GPU is optional and not the primary speed lever.
- `1w` is the warm-start root for `1d`.
