# Cloud 4H Profile

Date: 2026-04-02

Maintained authority:
- `deploy_4h.sh`
- `deploy_tf.py --tf 4h`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`

Profile summary:
- Warm-start parent: `1d`
- Execution mode: `hybrid_transition`
- Cross policy: required
- Same-machine required: no

Machine policy:
- Hard floor: `48` CPU cores, `256 GB` RAM, `1` GPU
- Preferred: `128` CPU cores, `512 GB` RAM, `1` CUDA 12 GPU

Maintained caveats:
- `4h` is the transition lane between CPU-heavy and GPU-heavy operation.
- Crosses are required.
- CPU-supported remains valid, but GPU meaningfully helps.
- `V2_RIGHT_CHUNK=500` is the maintained default.
