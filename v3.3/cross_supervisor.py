"""
cross_supervisor.py — Pipe-based GPU daemon supervisor / scheduler.

Stage 4 of the V4 Cross Gen Architecture.

Provides:
  prestage_gpu_daemons()    — fork one gpu_daemon per GPU
  run_cross_step()          — dispatch batches, collect .idx files via wait()
  shutdown_daemons()        — graceful teardown
  build_csr_from_idx_files() — fresh-process CSR assembly (scipy allowed there)

Integration: v2_cross_generator.py generate_all_crosses() calls:
  - prestage_gpu_daemons() ONCE at start
  - run_cross_step() for EACH of the 12 cross types
  - build_csr_from_idx_files() after each step
  - shutdown_daemons() at the end

KEY INVARIANTS (from architecture doc):
  - ZERO scipy in GPU daemons
  - ALL features preserved (matrix thesis — no filtering, no subsampling)
  - Pipe IPC, not Queue (CUDA fork deadlock avoidance)
  - Fresh subprocess for CSR build (zero pymalloc inheritance)
  - int64 indptr in final CSR (NNZ > 2^31 support)
"""

import gc
import json
import multiprocessing
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait as mp_wait
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Logging (matches v2_cross_generator pattern)
# ---------------------------------------------------------------------------

def _print(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [supervisor] {msg}", flush=True)


# ---------------------------------------------------------------------------
# IPC Protocol — plain tuples for pickle speed
# ---------------------------------------------------------------------------
# supervisor -> daemon:
#   ('RELOAD', left_npy_path, right_npy_path, n_left_cols)
#   ('BATCH',  batch_id, pairs_ndarray, out_path, pair_id_offset)
#   None                          # poison pill
#
# daemon -> supervisor:
#   ('READY', gpu_id)
#   ('RESULT', batch_id, idx_path, total_nnz, status, error_msg)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DaemonHandle
# ---------------------------------------------------------------------------

@dataclass
class DaemonHandle:
    """Supervisor-side handle to a GPU daemon process."""
    pipe: Connection
    process: multiprocessing.Process
    gpu_id: int
    status: str = 'idle'  # idle | computing | error | dead


# ---------------------------------------------------------------------------
# PairRegistry — global ID assignment across all cross types
# ---------------------------------------------------------------------------

class PairRegistry:
    """Maps global pair IDs -> (cross_type, left, right, feature_name).

    Global IDs are assigned sequentially across all cross types.
    Passed to CSR builder so it knows column mapping.
    """

    def __init__(self):
        self._next_id: int = 0
        self._names: list = []

    def register_pairs(
        self,
        valid_pairs: np.ndarray,
        left_names: list,
        right_names: list,
        prefix: str,
    ) -> np.ndarray:
        """Register pairs, assign global IDs. Returns int32 array of IDs."""
        n = len(valid_pairs)
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int32)
        for i in range(n):
            li, ri = int(valid_pairs[i, 0]), int(valid_pairs[i, 1])
            self._names.append(f"{prefix}{left_names[li]}_x_{right_names[ri]}")
        self._next_id += n
        return ids

    @property
    def total_features(self) -> int:
        return self._next_id

    def get_pair_id_to_col(self) -> np.ndarray:
        """arr[global_id] = output column index (identity for sequential)."""
        return np.arange(self._next_id, dtype=np.int32)

    def get_names_ordered(self) -> list:
        return list(self._names)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'total': self._next_id, 'names': self._names}, f)

    @classmethod
    def load(cls, path: str) -> 'PairRegistry':
        with open(path) as f:
            data = json.load(f)
        reg = cls()
        reg._next_id = data['total']
        reg._names = data['names']
        return reg


# ---------------------------------------------------------------------------
# Checkpoint / Resume State
# ---------------------------------------------------------------------------

@dataclass
class SupervisorState:
    """Tracks cross generation progress for resume capability."""
    completed_steps: list = field(default_factory=list)
    total_features: int = 0
    idx_files: list = field(default_factory=list)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'completed_steps': self.completed_steps,
                'total_features': self.total_features,
                'idx_files': self.idx_files,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SupervisorState':
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return cls()

    def mark_step_done(self, prefix: str, new_idx_files: list,
                       feature_count: int):
        self.completed_steps.append(prefix)
        self.idx_files.extend(new_idx_files)
        self.total_features += feature_count

    def is_step_done(self, prefix: str) -> bool:
        return prefix in self.completed_steps


# ---------------------------------------------------------------------------
# Stage 1: Daemon Pre-Start
# ---------------------------------------------------------------------------

def prestage_gpu_daemons(
    n_gpus: int,
    vram_limit_pct: float = 0.85,
    timeout: float = 30.0,
) -> list:
    """Fork one gpu_daemon per GPU. Each blocks on pipe.recv() until work.

    Must be called BEFORE binarize_contexts() returns, so CUDA init
    overlaps with CPU-bound feature preparation.

    Args:
        n_gpus: Number of GPUs to use (auto-detected if 0).
        vram_limit_pct: CuPy memory pool ceiling as fraction of total VRAM.
        timeout: Seconds to wait for each daemon READY signal.

    Returns:
        List of DaemonHandle, one per GPU. Caller owns lifecycle.
    """
    if n_gpus <= 0:
        n_gpus = _detect_gpu_count()

    handles = []
    ctx = multiprocessing.get_context('spawn')  # spawn avoids fork+CUDA issues

    for gpu_id in range(n_gpus):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        p = ctx.Process(
            target=_gpu_daemon_main,
            args=(child_conn, gpu_id, vram_limit_pct),
            daemon=True,
            name=f'gpu_daemon_{gpu_id}',
        )
        p.start()
        child_conn.close()  # parent doesn't need child end
        handles.append(DaemonHandle(
            pipe=parent_conn, process=p, gpu_id=gpu_id
        ))

    # Wait for all READY signals
    for h in handles:
        if h.pipe.poll(timeout):
            msg = h.pipe.recv()
            if isinstance(msg, tuple) and msg[0] == 'READY':
                _print(f"GPU daemon {h.gpu_id} ready (pid={h.process.pid})")
                h.status = 'idle'
            else:
                _print(f"GPU daemon {h.gpu_id} unexpected init msg: {msg}")
                h.status = 'error'
        else:
            _print(f"GPU daemon {h.gpu_id} TIMEOUT on init — marking dead")
            h.status = 'dead'

    alive = sum(1 for h in handles if h.status == 'idle')
    _print(f"Pre-staged {alive}/{n_gpus} GPU daemons")
    if alive == 0:
        raise RuntimeError("All GPU daemons failed to start")

    return handles


def _detect_gpu_count() -> int:
    """Detect available NVIDIA GPUs via CUDA_VISIBLE_DEVICES or nvidia-smi."""
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cvd:
        return len([x for x in cvd.split(',') if x.strip()])
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            timeout=10,
        )
        return len(out.decode().strip().split('\n'))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# GPU Daemon Main (runs in spawned child process)
# ---------------------------------------------------------------------------

def _gpu_daemon_main(conn: Connection, gpu_id: int, vram_limit_pct: float):
    """Entry point for GPU daemon process.

    IMPORT ORDER IS CRITICAL:
    1. Set CUDA_VISIBLE_DEVICES BEFORE importing cupy
    2. Import ONLY numpy + cupy (NEVER scipy)
    3. Initialize CuPy pool with hard limit
    4. Compile CUDA kernel
    5. Enter work loop

    INVARIANT: scipy is NEVER imported in this process.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')

    import numpy as _np  # fresh import in child

    try:
        import cupy as cp
    except ImportError:
        conn.send(('READY', gpu_id))
        _cpu_fallback_loop(conn, gpu_id)
        return

    # Hard VRAM ceiling
    cp.cuda.Device(0).use()
    try:
        cp.cuda.set_pinned_memory_allocator(None)
    except Exception:
        pass
    mempool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    total_vram = cp.cuda.Device(0).mem_info[1]
    mempool.set_limit(size=int(vram_limit_pct * total_vram))

    # Compile CUDA kernel once
    kernel = _compile_sparse_and_kernel(cp)

    conn.send(('READY', gpu_id))

    # State: current CSC on GPU (uploaded per RELOAD)
    indptr_gpu = None
    indices_gpu = None
    col_nnz_cpu = None
    n_rows = 0
    batches_done = 0

    stream_compute = cp.cuda.Stream(non_blocking=True)
    stream_transfer = cp.cuda.Stream(non_blocking=True)

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break

        if msg is None:  # poison pill
            break

        tag = msg[0]

        if tag == 'RELOAD':
            _, left_npy_path, right_npy_path, n_left_cols = msg
            try:
                indptr_gpu, indices_gpu, col_nnz_cpu, n_rows = _reload_csc_to_gpu(
                    left_npy_path, right_npy_path, n_left_cols, cp, _np
                )
                conn.send(('READY', gpu_id))
            except Exception as e:
                conn.send(('RESULT', -1, '', 0, 'error', f'RELOAD failed: {e}'))
            continue

        if tag == 'BATCH':
            _, batch_id, pairs, out_path, pair_id_offset = msg
            try:
                result = _process_batch(
                    batch_id, pairs, out_path, pair_id_offset,
                    indptr_gpu, indices_gpu, col_nnz_cpu, n_rows,
                    kernel, stream_compute, stream_transfer, cp, _np,
                )
                conn.send(result)
            except Exception as e:
                conn.send(('RESULT', batch_id, '', 0, 'error', str(e)))

            batches_done += 1
            # Free pool blocks every 10 batches (not every batch — perf)
            if batches_done % 10 == 0:
                cp.cuda.Stream.null.synchronize()
                mempool.free_all_blocks()
                pinned_pool.free_all_blocks()

    # Cleanup
    try:
        cp.cuda.Stream.null.synchronize()
        mempool.free_all_blocks()
    except Exception:
        pass


def _compile_sparse_and_kernel(cp):
    """Compile the sparse AND intersection CUDA kernel."""
    kernel_code = r'''
    extern "C" __global__
    void sparse_and_batch(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const int* __restrict__ col_pairs,
        int*       result_indices,
        int*       result_counts,
        const int  max_out_per_pair
    ) {
        int pair = blockIdx.x;
        int col_a = col_pairs[pair * 2];
        int col_b = col_pairs[pair * 2 + 1];
        int a = indptr[col_a], a_end = indptr[col_a + 1];
        int b = indptr[col_b], b_end = indptr[col_b + 1];
        int* out = result_indices + pair * max_out_per_pair;
        int cnt = 0;
        while (a < a_end && b < b_end && cnt < max_out_per_pair) {
            int ra = indices[a], rb = indices[b];
            if (ra == rb) { out[cnt++] = ra; a++; b++; }
            else if (ra < rb) { a++; }
            else { b++; }
        }
        result_counts[pair] = cnt;
    }
    '''
    return cp.RawKernel(kernel_code, 'sparse_and_batch')


def _reload_csc_to_gpu(left_npy_path, right_npy_path, n_left_cols, cp, np):
    """Upload combined [left|right] CSC to GPU. No scipy.

    Binary features -> only need indptr + indices, no data array.
    Returns (indptr_gpu, indices_gpu, col_nnz_cpu, n_rows).
    """
    left = np.load(left_npy_path, mmap_mode='r')
    right = np.load(right_npy_path, mmap_mode='r')
    combined = np.hstack([np.asarray(left), np.asarray(right)])
    del left, right

    n_rows, n_cols = combined.shape
    col_nnz = np.count_nonzero(combined, axis=0).astype(np.int64)
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(col_nnz, out=indptr[1:])
    total_nnz = int(indptr[-1])

    indices = np.empty(total_nnz, dtype=np.int32)
    for j in range(n_cols):
        rows = np.where(combined[:, j] != 0)[0].astype(np.int32)
        start = int(indptr[j])
        indices[start:start + len(rows)] = rows

    del combined

    # Upload to GPU (int32 for indptr is fine — per-column NNZ fits int32)
    indptr_gpu = cp.asarray(indptr.astype(np.int32))
    indices_gpu = cp.asarray(indices)

    return indptr_gpu, indices_gpu, col_nnz, n_rows


def _process_batch(batch_id, pairs, out_path, pair_id_offset,
                   indptr_gpu, indices_gpu, col_nnz_cpu, n_rows,
                   kernel, stream_compute, stream_transfer, cp, np):
    """Process one batch of cross pairs on GPU. Write .idx file.

    Triple-buffered: compute stream + transfer stream + CPU post-process.
    Returns ('RESULT', batch_id, idx_path, total_nnz, status, error_msg).
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return ('RESULT', batch_id, '', 0, 'ok', '')

    # Tight max_out estimate per pair
    left_nnz = col_nnz_cpu[pairs[:, 0]]
    right_nnz = col_nnz_cpu[pairs[:, 1]]
    pair_max = np.minimum(left_nnz, right_nnz).astype(np.int64)
    max_out = min(int(pair_max.max()), n_rows) if n_pairs > 0 else 0

    if max_out == 0:
        return ('RESULT', batch_id, '', 0, 'ok', '')

    # VRAM-adaptive sub-batching
    result_bytes = n_pairs * max_out * 4
    mem_free = cp.cuda.Device(0).mem_info[0]
    if result_bytes > mem_free * 0.6:
        sub_batch = max(100, int(n_pairs * (mem_free * 0.4) / result_bytes))
    else:
        sub_batch = n_pairs

    total_nnz = 0

    with open(out_path, 'wb') as f:
        # Header: magic(4) + version(2) + n_records(4) + n_rows(4) = 14 bytes
        f.write(b'IDX1')
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', 0))       # n_records placeholder
        f.write(struct.pack('<I', n_rows))

        n_records = 0

        for sb_start in range(0, n_pairs, sub_batch):
            sb_end = min(sb_start + sub_batch, n_pairs)
            sb_pairs = pairs[sb_start:sb_end]
            n_sb = sb_end - sb_start
            sb_max = int(pair_max[sb_start:sb_end].max())
            if sb_max == 0:
                continue

            # Launch kernel on compute stream
            with stream_compute:
                pairs_gpu = cp.asarray(sb_pairs.ravel().astype(np.int32))
                result_idx = cp.zeros(n_sb * sb_max, dtype=cp.int32)
                result_cnt = cp.zeros(n_sb, dtype=cp.int32)
                kernel(
                    (n_sb,), (1,),
                    (indptr_gpu, indices_gpu, pairs_gpu,
                     result_idx, result_cnt, np.int32(sb_max)),
                )

            # Sync compute, then bulk D2H on transfer stream
            stream_compute.synchronize()
            with stream_transfer:
                counts_cpu = cp.asnumpy(result_cnt)
                results_cpu = cp.asnumpy(result_idx)  # ONE bulk transfer
            stream_transfer.synchronize()

            # CPU post-process: build write buffer (ONE syscall per sub-batch)
            write_buf = bytearray()
            for i in range(n_sb):
                cnt = int(counts_cpu[i])
                if cnt > 0:
                    global_pair_id = pair_id_offset + sb_start + i
                    offset = i * sb_max
                    rows = results_cpu[offset:offset + cnt]
                    write_buf += struct.pack('<i', global_pair_id)
                    write_buf += struct.pack('<i', cnt)
                    write_buf += rows.astype(np.int32).tobytes()
                    total_nnz += cnt
                    n_records += 1

            if write_buf:
                f.write(write_buf)

            del pairs_gpu, result_idx, result_cnt

        # Patch header with actual record count
        f.seek(6)  # after magic(4) + version(2)
        f.write(struct.pack('<I', n_records))

    if total_nnz == 0:
        try:
            os.remove(out_path)
        except OSError:
            pass
        out_path = ''

    return ('RESULT', batch_id, out_path, total_nnz, 'ok', '')


def _cpu_fallback_loop(conn, gpu_id):
    """Stub loop when CuPy is unavailable."""
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        tag = msg[0]
        if tag == 'RELOAD':
            conn.send(('READY', gpu_id))
        elif tag == 'BATCH':
            conn.send(('RESULT', msg[1], '', 0, 'error',
                        'CuPy not available on this daemon'))


# ---------------------------------------------------------------------------
# Stage 4: Supervisor — run_cross_step
# ---------------------------------------------------------------------------

_BATCH_SIZE = 5000  # pairs per batch (architecture spec)


def run_cross_step(
    handles,
    left_mat,
    right_mat,
    valid_pairs,
    all_names,
    N,
    out_dir,
    prefix,
    pair_registry=None,
    left_names=None,
    right_names=None,
    batch_size=_BATCH_SIZE,
):
    """Execute one cross step across persistent GPU daemons.

    Steps:
      1. Save left/right matrices as .npy for daemon RELOAD
      2. Send RELOAD to all daemons, wait for READY
      3. Register pairs in PairRegistry for global ID tracking
      4. Build BatchMsg list, split into batches of batch_size
      5. Round-robin dispatch, collect via wait() multiplexing
      6. Return (.idx paths, total feature count)

    Args:
        handles: Live DaemonHandle list from prestage_gpu_daemons().
        left_mat: (N, n_left) binary numpy array.
        right_mat: (N, n_right) binary numpy array.
        valid_pairs: (n_pairs, 2) int32 — (left_col, right_col) indices.
        all_names: Combined feature names list.
        N: Number of data rows.
        out_dir: Directory for .idx files.
        prefix: Cross type prefix (e.g. 'dx_').
        pair_registry: Optional PairRegistry for global ID tracking.
        left_names: Left feature names (for pair_registry).
        right_names: Right feature names (for pair_registry).
        batch_size: Pairs per batch (default 5000).

    Returns:
        (idx_file_paths: list[str], total_feature_count: int)
    """
    n_pairs = len(valid_pairs)
    if n_pairs == 0:
        _print(f"[{prefix}] No valid pairs — skipping")
        return [], 0

    alive_handles = [h for h in handles if h.status not in ('dead', 'error')]
    if not alive_handles:
        raise RuntimeError(f"[{prefix}] All daemons dead — cannot proceed")

    # Create idx output directory
    idx_dir = os.path.join(out_dir, '_idx', prefix.rstrip('_'))
    os.makedirs(idx_dir, exist_ok=True)

    # Save left/right matrices as .npy for daemon RELOAD
    left_npy = os.path.join(out_dir, f'_left_{prefix.rstrip("_")}.npy')
    right_npy = os.path.join(out_dir, f'_right_{prefix.rstrip("_")}.npy')
    np.save(left_npy, np.ascontiguousarray(left_mat))
    np.save(right_npy, np.ascontiguousarray(right_mat))

    n_left_cols = left_mat.shape[1]

    # Register pairs and get global IDs
    if pair_registry is not None and left_names is not None and right_names is not None:
        global_ids = pair_registry.register_pairs(
            valid_pairs, left_names, right_names, prefix
        )
    else:
        global_ids = np.arange(n_pairs, dtype=np.int32)

    # Remap valid_pairs: right columns offset by n_left for combined CSC
    remapped_pairs = valid_pairs.copy().astype(np.int32)
    remapped_pairs[:, 1] += n_left_cols

    # ── RELOAD all daemons with new matrices ──
    _print(f"[{prefix}] RELOAD -> {len(alive_handles)} daemons "
           f"(left={left_mat.shape[1]}, right={right_mat.shape[1]}, "
           f"pairs={n_pairs:,})")

    for h in alive_handles:
        try:
            h.pipe.send(('RELOAD', left_npy, right_npy, n_left_cols))
        except (BrokenPipeError, OSError):
            h.status = 'dead'

    # Wait for READY from each daemon (timeout 60s per RELOAD)
    for h in alive_handles:
        if h.status == 'dead':
            continue
        try:
            if h.pipe.poll(60.0):
                resp = h.pipe.recv()
                if isinstance(resp, tuple) and resp[0] == 'READY':
                    h.status = 'idle'
                else:
                    _print(f"[{prefix}] daemon {h.gpu_id} RELOAD error: {resp}")
                    h.status = 'error'
            else:
                _print(f"[{prefix}] daemon {h.gpu_id} RELOAD timeout")
                h.status = 'dead'
        except (EOFError, OSError):
            h.status = 'dead'

    alive_handles = [h for h in handles if h.status == 'idle']
    if not alive_handles:
        raise RuntimeError(f"[{prefix}] All daemons dead after RELOAD")

    # ── Build batch messages ──
    all_batches = []
    for i in range(0, n_pairs, batch_size):
        end = min(i + batch_size, n_pairs)
        batch_pairs = remapped_pairs[i:end]
        bid = len(all_batches)
        idx_path = os.path.join(idx_dir, f'batch_{bid:05d}.idx')
        pair_id_offset = int(global_ids[i])
        all_batches.append(
            ('BATCH', bid, batch_pairs, idx_path, pair_id_offset)
        )

    n_batches = len(all_batches)
    _print(f"[{prefix}] Dispatching {n_batches} batches to "
           f"{len(alive_handles)} daemons")

    # ── Dispatch and collect via wait() multiplexing ──
    idx_paths, total_nnz = _dispatch_and_collect(
        alive_handles, all_batches, prefix
    )

    # Cleanup temp .npy files
    for p in [left_npy, right_npy]:
        try:
            os.remove(p)
        except OSError:
            pass

    _print(f"[{prefix}] Done — {len(idx_paths)} .idx files, "
           f"{n_pairs:,} features, {total_nnz:,} nnz")

    return idx_paths, n_pairs


def _dispatch_and_collect(handles, all_batches, prefix):
    """Dispatch batches round-robin, collect via wait() multiplexing.

    Work-stealing: sends next batch to whichever daemon finishes first.
    Handles daemon crashes: log error, skip batch, continue.

    Returns (idx_paths, total_nnz).
    """
    pending = list(range(len(all_batches)))
    in_flight = {}  # gpu_id -> batch_index
    pipe_to_handle = {}  # pipe object -> DaemonHandle
    idx_paths = []
    total_nnz = 0
    total_batches = len(all_batches)
    completed = 0

    def _update_pipe_map():
        pipe_to_handle.clear()
        for h in handles:
            if h.status == 'computing':
                pipe_to_handle[h.pipe] = h

    # Initial dispatch: one batch per daemon
    for h in handles:
        if not pending:
            break
        batch_idx = pending.pop(0)
        try:
            h.pipe.send(all_batches[batch_idx])
            in_flight[h.gpu_id] = batch_idx
            h.status = 'computing'
        except (BrokenPipeError, OSError):
            h.status = 'dead'
            pending.insert(0, batch_idx)

    _update_pipe_map()

    # Collect loop using multiprocessing.connection.wait()
    while in_flight:
        computing_pipes = [h.pipe for h in handles if h.status == 'computing']
        if not computing_pipes:
            break

        try:
            ready_pipes = mp_wait(computing_pipes, timeout=60.0)
        except (OSError, ValueError):
            for h in handles:
                if h.status == 'computing':
                    h.status = 'dead'
            break

        if not ready_pipes:
            # Timeout — health check: is the daemon process alive?
            for h in handles:
                if h.status == 'computing' and not h.process.is_alive():
                    _print(f"[{prefix}] daemon {h.gpu_id} died — "
                           f"skipping batch {in_flight.get(h.gpu_id, '?')}")
                    in_flight.pop(h.gpu_id, None)
                    h.status = 'dead'
            continue

        for pipe in ready_pipes:
            daemon = pipe_to_handle.get(pipe)
            if daemon is None:
                # Lookup fallback
                for h in handles:
                    if h.pipe is pipe:
                        daemon = h
                        break
            if daemon is None:
                continue

            try:
                result = daemon.pipe.recv()
            except (EOFError, OSError):
                daemon.status = 'dead'
                in_flight.pop(daemon.gpu_id, None)
                continue

            batch_idx = in_flight.pop(daemon.gpu_id, None)
            _, result_bid, result_path, result_nnz, status, err_msg = result

            if status == 'error':
                _print(f"[{prefix}] batch {result_bid} ERROR on GPU "
                       f"{daemon.gpu_id}: {err_msg}")
                daemon.status = 'idle'  # still try to reuse
            else:
                if result_path:
                    idx_paths.append(result_path)
                total_nnz += result_nnz
                completed += 1

            # Progress log every 10 batches
            if completed > 0 and (completed % 10 == 0 or completed == total_batches):
                _print(f"[{prefix}] progress: {completed}/{total_batches} "
                       f"batches, {total_nnz:,} nnz")

            # Dispatch next batch to this daemon (work-stealing)
            if pending and daemon.status != 'dead':
                next_idx = pending.pop(0)
                try:
                    daemon.pipe.send(all_batches[next_idx])
                    in_flight[daemon.gpu_id] = next_idx
                    daemon.status = 'computing'
                except (BrokenPipeError, OSError):
                    daemon.status = 'dead'
                    pending.insert(0, next_idx)
            else:
                if daemon.status != 'dead':
                    daemon.status = 'idle'

        _update_pipe_map()

    if pending:
        _print(f"[{prefix}] WARNING: {len(pending)} batches lost "
               f"(all daemons dead)")

    return idx_paths, total_nnz


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown_daemons(handles, timeout=60):
    """Send poison pill to each daemon, join with timeout, kill stragglers."""
    for h in handles:
        try:
            h.pipe.send(None)  # poison pill
        except (BrokenPipeError, OSError):
            pass

    for h in handles:
        h.process.join(timeout=timeout)
        if h.process.is_alive():
            _print(f"Killing straggler daemon {h.gpu_id} (pid={h.process.pid})")
            h.process.kill()
            h.process.join(timeout=5)
        try:
            h.pipe.close()
        except OSError:
            pass
        h.status = 'dead'

    _print(f"All {len(handles)} daemons shut down")


# ---------------------------------------------------------------------------
# Stage 3: CSR Assembly from .idx files (fresh subprocess)
# ---------------------------------------------------------------------------

# Embedded assembly script — runs in a fresh Python interpreter with zero
# inherited pymalloc fragmentation. scipy IS allowed here.
_ASSEMBLY_SCRIPT = r'''
"""CSR assembly from .idx files — fresh subprocess, zero pymalloc inheritance."""
import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np


def build_csr(idx_dir, n_rows, n_cols, pair_id_to_col_path, names_path,
              output_npz, tmp_dir):
    from scipy import sparse

    t0 = time.time()
    pair_id_to_col = np.load(pair_id_to_col_path)

    idx_files = sorted(glob.glob(os.path.join(idx_dir, '**', '*.idx'),
                                  recursive=True))
    if not idx_files:
        print(f"[csr_assembler] No .idx files in {idx_dir}", flush=True)
        json.dump({'npz_path': '', 'n_cols': 0, 'total_nnz': 0, 'elapsed_s': 0},
                  open(output_npz + '.result.json', 'w'))
        return

    print(f"[csr_assembler] Scanning {len(idx_files)} .idx files...", flush=True)

    # PASS 1: Pre-scan for total NNZ (headers only)
    total_nnz = 0
    for fpath in idx_files:
        with open(fpath, 'rb') as fh:
            magic = fh.read(4)
            if magic != b'IDX1':
                print(f"[csr_assembler] WARNING: bad magic in {fpath}", flush=True)
                continue
            _ver = struct.unpack('<H', fh.read(2))[0]
            n_records = struct.unpack('<I', fh.read(4))[0]
            for _ in range(n_records):
                _pair_id = struct.unpack('<i', fh.read(4))[0]
                nnz = struct.unpack('<i', fh.read(4))[0]
                total_nnz += nnz
                fh.seek(nnz * 4, 1)

    if total_nnz == 0:
        print("[csr_assembler] No nonzeros found", flush=True)
        json.dump({'npz_path': '', 'n_cols': 0, 'total_nnz': 0,
                   'elapsed_s': time.time() - t0},
                  open(output_npz + '.result.json', 'w'))
        return

    print(f"[csr_assembler] Total NNZ: {total_nnz:,}", flush=True)

    # Allocate memmap arrays for two-pass assembly
    os.makedirs(tmp_dir, exist_ok=True)
    col_mm_path = os.path.join(tmp_dir, '_asm_col.dat')
    row_mm_path = os.path.join(tmp_dir, '_asm_row.dat')
    col_mm = np.memmap(col_mm_path, dtype=np.int32, mode='w+', shape=(total_nnz,))
    row_mm = np.memmap(row_mm_path, dtype=np.int32, mode='w+', shape=(total_nnz,))

    # PASS 2: Fill memmaps
    write_pos = 0
    for fpath in idx_files:
        with open(fpath, 'rb') as fh:
            magic = fh.read(4)
            if magic != b'IDX1':
                continue
            fh.seek(12)  # skip full header (magic + ver + n_records + n_rows)
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                pair_id = struct.unpack('<i', hdr[:4])[0]
                nnz = struct.unpack('<i', hdr[4:8])[0]
                rows = np.frombuffer(fh.read(nnz * 4), dtype=np.int32)
                if pair_id < 0 or pair_id >= len(pair_id_to_col):
                    continue
                col_id = int(pair_id_to_col[pair_id])
                end = write_pos + nnz
                col_mm[write_pos:end] = col_id
                row_mm[write_pos:end] = rows
                write_pos += nnz

    col_mm.flush()
    row_mm.flush()
    actual_nnz = write_pos

    print(f"[csr_assembler] Building CSC ({n_rows} x {n_cols}, "
          f"nnz={actual_nnz:,})...", flush=True)

    # Build indptr via bincount -> cumsum
    col_counts = np.bincount(col_mm[:actual_nnz], minlength=n_cols).astype(np.int64)
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(col_counts, out=indptr[1:])

    # Sort by column via argsort (stable for deterministic output)
    order = np.argsort(col_mm[:actual_nnz], kind='stable')
    indices_out = row_mm[:actual_nnz][order].astype(np.int32)
    del order

    # Data array: all ones (binary crosses)
    data = np.ones(actual_nnz, dtype=np.float32)

    # Construct CSC (zero-copy from sorted arrays)
    csc = sparse.csc_matrix((data, indices_out, indptr),
                             shape=(n_rows, n_cols), copy=False)

    # Convert to CSR
    print("[csr_assembler] CSC -> CSR...", flush=True)
    csr = csc.tocsr()
    del csc

    # Enforce int64 indptr for NNZ > 2^31
    if csr.nnz > 2**30:
        csr.indptr = csr.indptr.astype(np.int64)

    # Save (uncompressed for speed)
    print(f"[csr_assembler] Saving {output_npz}...", flush=True)
    sparse.save_npz(output_npz, csr, compressed=False)

    elapsed = time.time() - t0
    result = {
        'npz_path': output_npz,
        'n_cols': n_cols,
        'total_nnz': int(csr.nnz),
        'elapsed_s': round(elapsed, 1),
    }
    json.dump(result, open(output_npz + '.result.json', 'w'))

    print(f"[csr_assembler] Done in {elapsed:.1f}s — {n_cols:,} cols, "
          f"{csr.nnz:,} nnz", flush=True)

    # Cleanup memmaps
    del col_mm, row_mm, indices_out, data, csr
    for p in [col_mm_path, row_mm_path]:
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx-dir', required=True)
    parser.add_argument('--n-rows', type=int, required=True)
    parser.add_argument('--n-cols', type=int, required=True)
    parser.add_argument('--pair-id-to-col', required=True)
    parser.add_argument('--names-path', required=True)
    parser.add_argument('--output-npz', required=True)
    parser.add_argument('--tmp-dir', required=True)
    args = parser.parse_args()
    build_csr(args.idx_dir, args.n_rows, args.n_cols,
              args.pair_id_to_col, args.names_path,
              args.output_npz, args.tmp_dir)
'''


def build_csr_from_idx_files(
    idx_files,
    all_names,
    N,
    out_dir,
    pair_registry=None,
    tmp_dir=None,
    timeout=3600,
):
    """Build CSR from .idx files in a FRESH subprocess (scipy allowed there).

    Uses subprocess.Popen (NOT multiprocessing.Process) for cleanest
    process isolation — child gets fresh Python interpreter with zero
    inherited heap fragmentation.

    Args:
        idx_files: List of .idx file paths (used to determine idx_dir).
        all_names: Ordered feature names.
        N: Number of data rows.
        out_dir: Output directory for final NPZ.
        pair_registry: PairRegistry with global ID mappings.
        tmp_dir: Temp directory for memmap (NVMe preferred).
        timeout: Max seconds to wait for assembly.

    Returns:
        (merged_names: list[str], csr_npz_path: str, total_cols: int)
    """
    if not idx_files:
        _print("build_csr_from_idx_files: no .idx files — returning empty")
        return [], '', 0

    # Determine idx root directory
    idx_dir = os.path.join(out_dir, '_idx')
    if not os.path.isdir(idx_dir):
        # Fallback: use parent of first .idx file
        idx_dir = os.path.dirname(idx_files[0])

    if pair_registry is not None:
        n_cols = pair_registry.total_features
        names = pair_registry.get_names_ordered()
        pair_id_to_col = pair_registry.get_pair_id_to_col()
    else:
        n_cols = len(all_names)
        names = list(all_names)
        pair_id_to_col = np.arange(n_cols, dtype=np.int32)

    if n_cols == 0:
        return [], '', 0

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix='csr_asm_')

    # Serialize inputs for child process
    os.makedirs(tmp_dir, exist_ok=True)
    pid_to_col_path = os.path.join(tmp_dir, '_pair_id_to_col.npy')
    names_path = os.path.join(tmp_dir, '_names.json')
    output_npz = os.path.join(out_dir, 'v2_crosses_merged.npz')
    script_path = os.path.join(tmp_dir, '_csr_assembler.py')

    np.save(pid_to_col_path, pair_id_to_col)
    with open(names_path, 'w') as f:
        json.dump(names, f)
    with open(script_path, 'w') as f:
        f.write(_ASSEMBLY_SCRIPT)

    _print(f"CSR assembly subprocess: {n_cols:,} cols, N={N}, idx_dir={idx_dir}")

    cmd = [
        sys.executable, script_path,
        '--idx-dir', idx_dir,
        '--n-rows', str(N),
        '--n-cols', str(n_cols),
        '--pair-id-to-col', pid_to_col_path,
        '--names-path', names_path,
        '--output-npz', output_npz,
        '--tmp-dir', tmp_dir,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream child output to parent stdout
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"CSR assembly timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(
            f"CSR assembly failed with exit code {proc.returncode}")

    # Read result JSON written by child
    result_path = output_npz + '.result.json'
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        try:
            os.remove(result_path)
        except OSError:
            pass

        if not result.get('npz_path'):
            return [], '', 0

        _print(f"CSR assembly complete: {result['n_cols']:,} cols, "
               f"{result['total_nnz']:,} nnz, {result['elapsed_s']:.1f}s")
        return names, result['npz_path'], result['n_cols']

    # Fallback: check if NPZ was written
    if os.path.exists(output_npz):
        return names, output_npz, n_cols

    return [], '', 0
