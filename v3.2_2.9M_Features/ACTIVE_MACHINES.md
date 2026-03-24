# Active Machines — Live Tracker
# LAST UPDATED: 2026-03-24 ~09:30 CST (15:30 UTC)

## RUNNING MACHINES ($4.17/hr total)

| ID | Label | TF | Task | Cores | RAM | $/hr | SSH | Status |
|---|---|---|---|---|---|---|---|---|
| 33446502 | texas-15m | 15m | Training (sparse, single-thread) | 256c | 1TB | $1.36 | ssh -p 16502 root@ssh6.vast.ai | Running ~3hrs, validation run |
| 33456978 | belgium-15m | — | KILLED (SSH kept dropping) | — | — | — | — | DEAD |
| 33457281 | norway-optuna | 1h | 1h Optuna RUNNING | 384c | 1TB | $0.82 | ssh -p 17280 root@ssh1.vast.ai | Launched, dense conversion |
| 33457282 | uk-optuna | — | KILLED (SSH unreachable) | — | — | — | — | DEAD |
| 33457954 | belgium2-15m | — | KILLED (unstable SSH, 3 upload failures) | — | — | — | — | DEAD |
| 33457957 | aus-optuna | 1d→4h | 1d Optuna RUNNING | 256c | 1TB | $1.21 | ssh -p 17956 root@ssh2.vast.ai | Then 4h after 1d |

## COMPLETED & DOWNLOADED

| TF | Dir | Model | LSTM | Meta | PBO | NPZ | Optuna |
|---|---|---|---|---|---|---|---|
| 1w | v32_cloud_results/1w_norway/ | ✅ 142KB | ✅ 2.8MB 55.3% | ✅ AUC=0.670 | REJECT 0.50 | ❌ rebuild 30s | ❌ deferred (818 rows) |
| 1d | v32_cloud_results/1d_final/ | ✅ 598KB | ✅ 7MB | ✅ | **DEPLOY 0.00** | ✅ 292MB | ⏳ running on UK |
| 4h | v32_cloud_results/4h_final/ | ✅ 893KB | ✅ 17MB 53.5% | ✅ AUC=0.616 | REJECT 0.40 | ✅ 125MB | ⏳ queued after 1d on UK |
| 1h | v32_cloud_results/1h_final/ | ✅ 2.7MB | ✅ 27MB 53.9% | ✅ AUC=0.648 | **DEPLOY 0.14** | ✅ 1.6GB | ⏳ running on Norway |
| 15m | — | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## PIPELINE PER MACHINE

### Texas (15m sparse validation)
- Cross gen: DONE (2,282,328 crosses)
- Training: IN PROGRESS (~22 min into CPCV, 15 folds, single-threaded)
- Might be valid per Perplexity (int32 fixed in v4.6) but slow
- ETA: 8-12 more hours
- ACTION: Let run, compare with Belgium when both done

### Belgium (15m dense 210K rows)
- Upload: RETRYING (failed twice, SSH drops)
- Plan: feature rebuild → cross gen → subsample 210K rows → dense train → meta → LSTM → PBO
- ETA from upload: ~3-4 hours total
- ACTION: Get upload working, then autonomous pipeline

### Norway (1h Optuna)
- Upload: IN PROGRESS (1.6GB tar)
- Plan: run cloud_run_optuna.py --tf 1h → Optuna search → optimizer
- Dense: 300GB on 503GB RAM (tight but worked before)
- ETA from upload: ~2-3 hours
- ACTION: Launch after upload, download results, kill machine

### UK (1d Optuna → 4h Optuna)
- Upload: IN PROGRESS (430MB tar)
- Plan: run cloud_run_optuna.py --tf 1d → download → rebuild tar for 4h → run 4h Optuna
- Dense: 66GB for 1d, 56GB for 4h — easy fit in 1TB
- ETA: ~1.5 hrs (1d) + ~1.5 hrs (4h) = ~3 hrs total
- ACTION: Launch 1d after upload, queue 4h after 1d done

## COST TRACKING
- Previous spending: ~$15
- Current burn: $4.17/hr
- Estimated remaining runtime: ~4 hrs
- Estimated remaining cost: ~$17
- Total estimated: ~$32 (at budget limit)

## QUEUE (in order)
1. [NOW] Get Belgium/Norway/UK uploads working
2. [NOW] Launch 1d Optuna on UK
3. [NOW] Launch 1h Optuna on Norway
4. [NOW] Launch 15m on Belgium (after upload)
5. [AFTER 1d] Run 4h Optuna on UK (same machine, rebuild tar)
6. [AFTER ALL] Download everything, kill all machines
7. [AFTER ALL] Verify 15m Texas vs Belgium results
8. [LATER] Run 1w Optuna locally if desired (small, free)
9. [LATER] Run 15m Optuna (needs training to finish first)
