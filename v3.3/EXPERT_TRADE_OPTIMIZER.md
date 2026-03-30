# EXPERT: GPU-Accelerated Trade Strategy Optimizer

## Context
GPU-accelerated exhaustive trade strategy optimizer with 7 parameters, maximizing Sortino ratio, using CuPy on RTX 5090 (21,760 CUDA cores, 32 GB GDDR7, 1.79 TB/s bandwidth).

---

## 1. GPU-Accelerated Grid Search for Trading Strategy Optimization

### Core Architecture
The optimal design is a **hybrid CuPy + Numba CUDA** engine:

- **CuPy** handles: parameter grid generation, market data arrays, masks, and final metric reductions (Sortino scoring). CuPy's reductions use optimized CUB/Thrust backends, which are highly efficient for large parallel reductions.
- **Numba CUDA kernels** handle: stateful bar-by-bar trade simulation (trailing stops, position transitions, inventory tracking). NVIDIA's own trading acceleration guidance confirms Numba gives more control when computation is dominated by sequential-in-time logic with branching.

### Why Hybrid Matters
Pure CuPy underdelivers for trading backtests because trading engines hide a serial state machine inside each candidate evaluation. Vectorized array math is great for metric computation, but the bar-by-bar state update is inherently sequential within a single candidate. An explicit CUDA kernel lets one thread own one candidate path, update state through time, and write back only summary statistics.

### RTX 5090 Fit
- ~27% ahead of RTX 4090 in CUDA benchmarks
- 32 GB VRAM is the critical constraint: determines whether all candidates evaluate in one pass or need chunked batches
- 1.79 TB/s bandwidth suits the reduction-heavy Sortino workload

### Performance Best Practices (CuPy 2025)
- **Kernel launch latency**: ~5-10 microseconds per small kernel. Fuse elementwise operations and batch work. Fewer large launches beat many tiny kernels.
- **Memory layout**: Store candidate-by-time arrays contiguously along the reduction axis. Avoid needless transposes in the hot path.
- **CUB-backed reductions**: Performance depends on contiguous axes. Benchmark both C-order and F-order layouts.
- **Use `argmax` not full sort**: If you only need the best parameter set, `argmax` on Sortino scores avoids the overhead of full GPU sorting, which is a known bottleneck at scale.

---

## 2. Bayesian Optimization vs Exhaustive Grid Search

### Decision Framework

| Approach | Best When | Main Upside | Main Downside |
|---|---|---|---|
| **Coarse exhaustive sweep** | Fast GPU engine, want full landscape | Finds ridges, plateaus, interactions, robustness zones directly | Resolution limited by combinatorial explosion |
| **Bayesian optimization** | Each eval is very expensive, tight budget | Reaches promising areas in far fewer evaluations | Hard to parallelize in large batches; misled by noisy objectives |
| **Hybrid (recommended)** | Fast GPU + want sample efficiency | Coarse sweep for topology, Bayes for local refinement | More engineering complexity |

### Why GPU Tilts Toward Exhaustive
- Grid search is **embarrassingly parallel** -- maximizes GPU occupancy
- Bayesian optimization is **sequential by nature** (each point depends on prior results)
- Parallel Bayesian methods (q-EI batches) exist but still don't achieve full occupancy
- On RTX 5090, GPU occupancy matters more than evaluation count reduction

### Sortino-Specific Caution
- Sortino ratio is **especially unstable** when downside deviation is estimated from limited trades
- Bayesian optimization's single best point may be a **sharp local spike**, not a robust region
- Exhaustive search lets you inspect whether the best result sits on a **broad plateau** (trustworthy) or a **needle peak** (overfit)
- For live trading, broad plateaus with slightly lower headline Sortino are usually more reliable

### Recommended Strategy
1. **Stage 1**: Coarse GPU-exhaustive sweep across all 7 parameters
2. **Stage 2**: Score robustness neighborhoods -- median Sortino across nearby parameter cells, sensitivity to one-step perturbations
3. **Stage 3**: Only add Bayesian optimization as second-stage refiner inside the best 1-3 regions if finer resolution is needed
4. **Stage 4**: Final selection by out-of-sample/walk-forward robustness, never in-sample peak Sortino alone

---

## 3. Sobol Sequences vs Latin Hypercube Sampling

### Verdict: Use Scrambled Sobol

For a 7D bounded parameter search on GPU, **Sobol sequences are superior to LHS**:

| Property | Sobol | LHS |
|---|---|---|
| Uniformity | Better low-discrepancy coverage | Good marginal coverage only |
| Determinism | Fully deterministic, reproducible | Depends on random seed |
| Extensibility | Can continue sequence / fast-forward | Must regenerate entire design |
| Generation cost | Lower | Higher |
| GPU batch-friendly | Yes (power-of-2 batches) | Less natural |

### Why Extensibility Matters
GPU optimization uses adaptive batches (65K points, then 262K more around promising regions). Sobol supports this naturally via sequence continuation or fast-forward. LHS requires regenerating the entire design when adding points, which is impractical for iterative refinement.

### Practical GPU Implementation
- **Generate Sobol on CPU** with `scipy.stats.qmc.Sobol` using `random_base2` for balanced draws
- **Bulk transfer** to GPU (sample generation cost is tiny vs evaluation cost)
- For enormous continuous batches: use NVIDIA cuRAND/cuRANDDx scrambled Sobol generators directly on GPU
- CuPy does not provide native GPU Sobol generation

### Recommended Workflow
1. **Global stage**: Sobol over full 7D box for broad coverage (power-of-2 batch sizes)
2. **Evaluate on GPU** in VRAM-sized chunks
3. **Keep top 1-5%** by Sortino, shrink bounds around robust clusters (not single best points)
4. **Refinement stage**: Narrowed Sobol again, or local exhaustive grid sweep around top clusters if parameter interactions look sharp
5. **Continue the Sobol sequence** instead of restarting random seeds

### When Sobol Acts as Pre-Filter
For truly exhaustive search, Sobol is useful as a front-end filter before the expensive dense sweep. It identifies which subregions deserve exhaustive enumeration -- much better than gridding the whole 7D space uniformly.

---

## 4. CuPy Kernel Fusion for Sortino Computation

### Sortino Ratio on GPU
The Sortino ratio formula:
```
Sortino = (mean_return - target) / downside_deviation
```
Where downside deviation uses only returns below the target.

### Optimal GPU Pipeline
A branch-light, fusion-friendly pipeline:

```
1. Compute per-period returns            (elementwise)
2. diff = returns - target               (elementwise, FUSE with step 1)
3. negative_part = min(diff, 0)          (elementwise, FUSE)
4. squared = negative_part^2             (elementwise, FUSE)
5. sum_squared = reduce_sum(squared)     (reduction along time axis)
6. mean_return = reduce_mean(returns)    (reduction along time axis)
7. sortino = (mean_return - target) / sqrt(sum_squared / N)  (elementwise)
```

### Fusion Strategy
- **Steps 1-4 should be fused** into a single `ElementwiseKernel` to avoid intermediate memory allocations and extra memory bandwidth
- **Steps 5-6** use CuPy `ReductionKernel` or built-in `sum`/`mean` with CUB backends
- **Step 7** is a final elementwise operation across all candidates

### CuPy Kernel Types
- **`ElementwiseKernel`**: Fuse multiple elementwise operations into one kernel launch. Eliminates intermediate tensor materializations. Critical for steps 1-4.
- **`ReductionKernel`**: Custom reduction with fused pre-reduction transforms. Can combine steps 3-5 into a single kernel (compute negative part, square, and sum in one pass).
- **`RawKernel`**: Maximum control. Write CUDA C directly. Use when CuPy abstractions add overhead or when combining elementwise + reduction logic.

### Online Accumulation (Critical Optimization)
**Do not materialize huge intermediate tensors.** Instead, accumulate numerator (sum of returns) and downside denominator (sum of squared negative deviations) **online inside the kernel**. This:
- Reduces VRAM from O(candidates x bars) to O(candidates)
- Cuts memory bandwidth pressure, which is often more important than raw FLOPS
- Enables evaluating more candidates per VRAM-limited batch

### Tensor Layout
- **2D tensor**: axis 0 = candidate index, axis 1 = time/bar index
- Store contiguously along the reduction axis (time) for optimal CUB reduction performance
- Each candidate's Sortino can be reduced in parallel across all candidates simultaneously

### Top-K Selection
- For best candidate: use `cp.argmax` on Sortino scores
- For top-K candidates for walk-forward validation: use partial selection / `cp.argpartition` rather than full sort
- Full GPU sorting is overkill and becomes a bottleneck at scale

---

## 5. Complete Optimizer Architecture

### Data Flow
```
[CPU] Generate Sobol points (scipy) --> scale to 7 parameter ranges
  |
  v
[GPU] Transfer parameter grid + OHLCV data (one-time copy)
  |
  v
[GPU] For each VRAM-sized chunk of candidates:
  |   - Numba CUDA kernel: bar-by-bar trade simulation
  |   - Writes per-candidate return streams (or online Sortino accumulators)
  |
  v
[GPU] CuPy fused kernel: Sortino computation (if not done online)
  |
  v
[GPU] cp.argmax or cp.argpartition for top-K selection
  |
  v
[CPU] Transfer back ONLY winning parameter rows + diagnostics
```

### VRAM Budget (RTX 5090, 32 GB)
- OHLCV data: ~50-200 MB (depends on bar count)
- Parameter grid chunk: sized to leave room for intermediate arrays
- If online accumulation: ~16 bytes per candidate (2 float64 accumulators)
- If materializing returns: candidates x bars x 8 bytes -- this is the bottleneck
- **Rule of thumb**: With 10K bars, materializing returns allows ~400K candidates per batch. Online accumulation allows millions.

### Chunk Sizing Strategy
```python
available_vram = 30 * 1024**3  # 30 GB usable of 32 GB
bytes_per_candidate = n_bars * 8  # if materializing returns
# OR
bytes_per_candidate = 16  # if online accumulation

chunk_size = available_vram // bytes_per_candidate
```

### 7-Parameter Indexing
For exhaustive grid, use CuPy broadcasting or flat index with modular arithmetic:
```python
# Flat index to 7D parameter tuple
idx = candidate_id
p0 = idx % n0; idx //= n0
p1 = idx % n1; idx //= n1
# ... etc for all 7 parameters
```

For Sobol, the 7D unit cube values are directly scaled to parameter ranges:
```python
# Scale Sobol [0,1]^7 to actual parameter bounds
params = lower_bounds + sobol_points * (upper_bounds - lower_bounds)
```

---

## 6. Summary of Recommendations

| Decision | Recommendation | Rationale |
|---|---|---|
| **Primary search method** | Coarse exhaustive via Sobol sampling | Maximizes GPU occupancy, reveals full landscape |
| **Refinement method** | Narrowed Sobol or local grid, optional Bayesian | Only where finer resolution needed |
| **Sampling scheme** | Scrambled Sobol (power-of-2 batches) | Deterministic, extensible, low-discrepancy |
| **GPU framework** | CuPy + Numba CUDA hybrid | CuPy for reductions/metrics, Numba for stateful simulation |
| **Sortino computation** | Fused ElementwiseKernel + ReductionKernel | Minimize kernel launches and intermediate memory |
| **Memory strategy** | Online accumulation inside kernel | O(candidates) not O(candidates x bars) VRAM |
| **Selection method** | `argmax` or `argpartition`, never full sort | Avoid sorting bottleneck |
| **Robustness check** | Score neighborhood stability, not just peak | Broad plateaus > needle peaks for live trading |
| **Final validation** | Out-of-sample / walk-forward | Never trust in-sample peak Sortino |

---

## Sources
- NVIDIA Developer Blog: GPU-Accelerate Algorithmic Trading Simulations (Numba)
- CuPy v14 Performance Best Practices Documentation
- NVIDIA cuRANDDx: Scrambled Sobol Generator Documentation
- SciPy v1.17: scipy.stats.qmc.Sobol Documentation
- Analytica: Monte Carlo vs Latin Hypercube vs Sobol Sampling Comparison
- Tom's Hardware: RTX 5090 CUDA Performance Benchmarks
- QuantInsti: RAPIDS Libraries for Trading (VRAM Constraints)
- EconStor: Bayesian Approach to Backtest Overfitting
- Grid Search vs Random Search in Trading (adventuresofgreg.blog)
- PMC: Performance Evaluation of GPU-Based Parallel Sorting Algorithms
