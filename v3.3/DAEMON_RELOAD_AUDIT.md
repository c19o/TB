# Daemon RELOAD Audit

Last updated: 2026-04-01

## Current Judgment

The daemon issue appears substantially fixed in code.

What that means:
- for `1w`, it does not currently look like a real blocker
- for `1d` and below, it is still proof-pending rather than clearly broken

## Code-Level State

- `cross_supervisor.py` and `gpu_daemon.py` agree on the `RELOAD -> READY -> RESULT` contract
- `n_left_cols` handling is aligned
- the CSR rebuild path in `v2_cross_generator.py` is on the corrected contract
- the key daemon files compile cleanly

## Remaining Gap

The missing piece is not more theory. It is one real rerun on the supported machine showing:

1. `RELOAD` issued
2. every daemon returning `READY`
3. batch execution returning `RESULT`
4. no unexpected fallback to legacy path

## Operational Meaning

- `1w`: proceed
- `1d+`: require fresh rerun evidence before calling the daemon lane fully closed
