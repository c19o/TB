# Cloud 1D Profile

Date: 2026-04-02

Maintained authority:
- `deploy_1d.sh`
- `deploy_tf.py --tf 1d`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`

Profile summary:
- Warm-start parent: `1w`
- Execution mode: `cpu_first`
- Cross policy: required
- Same-machine required: no

Machine policy:
- Hard floor: `32` CPU cores, `128 GB` RAM, `1` GPU
- Preferred: `128` CPU cores, `256 GB` RAM, `1` GPU optional

Maintained caveats:
- `1d` is still CPU-first in the current maintained path.
- Crosses are required.
- GPU is not the main limiter today.
- Use the `1w` artifacts as the warm-start parent.
