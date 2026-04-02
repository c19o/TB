# Cloud 1H Profile

Date: 2026-04-02

Maintained authority:
- `deploy_1h.sh`
- `deploy_tf.py --tf 1h`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`

Profile summary:
- Warm-start parent: `4h`
- Execution mode: `gpu_required`
- Cross policy: required
- Same-machine required: no

Machine policy:
- Hard floor: `64` CPU cores, `512 GB` RAM, `1` GPU
- Preferred: `128` CPU cores, `768 GB` RAM, `2` CUDA 12 GPUs

Maintained caveats:
- `1h` is where GPU stops being optional for maintained production speed.
- CPU-only should be treated as rescue mode, not the preferred path.
- Crosses are required.
- Warm-start from `4h`.
