# Cloud 15M Profile

Date: 2026-04-02

Maintained authority:
- `deploy_15m.sh`
- `deploy_tf.py --tf 15m`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`

Profile summary:
- Warm-start parent: `1h`
- Execution mode: `gpu_required_same_machine`
- Cross policy: required
- Same-machine required: yes

Machine policy:
- Hard floor: `96` CPU cores, `768 GB` RAM, `1` GPU
- Preferred: `128` CPU cores, `1024 GB` RAM, `2` CUDA 12 GPUs

Maintained caveats:
- `15m` is the heaviest lane.
- Same-machine resume is mandatory in maintained mode.
- GPU is required.
- `V2_RIGHT_CHUNK=500` and `V2_BATCH_MAX=500` are maintained defaults; lower them only under measured memory pressure.
