# V3.3 Session Resume — 2026-03-25

## STATUS: 5 TFs TRAINING ON VAST.AI

| TF | Instance | CPU | Port | Host | $/hr | Status | ETA |
|----|----------|-----|------|------|------|--------|-----|
| 1w | 33496064 | EPYC 9654 384c 3.7GHz | 16064 | ssh1.vast.ai | $0.80 | CPCV training | Soon |
| 1d | 33497394 | EPYC 7773X 128c 3.5GHz | 17394 | ssh2.vast.ai | $0.32 | Cross gen | ~3 hrs |
| 4h | 33497395 | EPYC 7B13 128c 3.5GHz | 17394 | ssh6.vast.ai | $0.23 | Running | ~3 hrs |
| 1h | 33496904 | EPYC 9K84 384c 3.7GHz | 16904 | ssh1.vast.ai | $0.80 | 3.6M crosses | ~6 hrs |
| 15m | 33497779 | Ryzen 9950X 32c 8.8GHz | 17778 | ssh9.vast.ai | $0.77 | Just launched | ~10 hrs |

## SSH COMMAND
```bash
SSH_OPTS="-o StrictHostKeyChecking=no -o IdentityFile=/c/Users/C/.ssh/vast_key -o IdentitiesOnly=yes"
ssh $SSH_OPTS -p <PORT> root@<HOST> "tail -20 /workspace/<TF>_pipeline.log"
```

## WHAT WAS DONE THIS SESSION
1. Fixed 7 streamers (tweet color, gematria, sports lookahead, DB paths, space weather, easy streamers)
2. Added 235 new esoteric features (vortex math, sacred geometry, planetary expansion, numerology, lunar/EM, gematria ciphers, holidays)
3. Fixed 12 calculation bugs (eclipse formula, BaZi, Tzolkin, VOC moon, oppose LUT, etc.)
4. Fixed training params (class_weight='balanced', max_bin=63, max_conflict_rate=0.0, path_smooth, extra_trees)
5. Added SHAP cross validation as pipeline Step 11
6. Fixed cloud deployment (cuda13 image, numba_cuda disable, script paths, all 15 DBs)
7. 15m runs SPARSE (no dense conversion needed — same accuracy, just slower loading)

## OVERNIGHT RULES
- Download artifacts as each TF finishes
- Never kill machine until files verified
- On error: reuse machine, don't destroy
- Consult Perplexity on any failure
- Update protocol after every fix

## NEXT STEPS (after training)
- Download + verify all 5 TF artifacts
- Kill machines after verification
- Scale-in/Kelly/dynamic exits in exhaustive_optimizer
- LSTM projection + attention upgrade
- 15m distributed training setup (for future retraining)
- Git commit on v3.3 branch
- Paper trading
