#!/usr/bin/env python
"""
gpu_daemon.py — Zero-scipy GPU daemon for cross feature generation (Stage 1+2)
===============================================================================
Each daemon owns ONE GPU. It:
  1. Initializes CUDA context + compiles the sparse AND kernel
  2. Receives RELOAD messages to upload new CSC matrices per cross step
  3. Receives BATCH messages to compute intersections → writes raw .idx files
  4. Receives None (poison pill) to exit

INVARIANT: ZERO scipy imports. Only numpy + cupy + raw binary I/O.

IPC: multiprocessing.Pipe (duplex=True). NOT Queue (deadlock risk with CUDA fork).

.idx file format (per record):
    global_pair_id (int32) | nnz (int32) | row_indices (int32[nnz])
Header (12 bytes):
    magic (4 bytes 'IDX1') | version (uint16) | n_records (uint32) | n_rows (uint32)
"""

import os
import time
import gc
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing import Process
from multiprocessing.connection import Connection
from dataclasses import dataclass, field


# ============================================================
# DATA CLASSES — IPC message protocol
# ============================================================

@dataclass
class DaemonHandle:
    """Parent-side handle to a GPU daemon process."""
    pipe: Connection      # parent end of Pipe(duplex=True)
    process: Process
    gpu_id: int
    status: str = 'idle'  # idle | computing | error | dead


@dataclass
class ReloadMsg:
    """Upload new CSC matrices for a new cross step."""
    tag: str = 'RELOAD'
    left_npy_path: str = ''     # path to left binary matrix (.npy)
    right_npy_path: str = ''    # path to right binary matrix (.npy)
    n_left_cols: int = 0
    n_rows: int = 0


@dataclass
class BatchMsg:
    """Process a batch of cross pairs."""
    tag: str = 'BATCH'
    batch_id: int = 0
    pairs: object = None        # np.ndarray (n_pairs, 2) int32 — already remapped
    out_path: str = ''          # where to write .idx file
    pair_id_offset: int = 0     # global pair ID offset for this batch


@dataclass
class ResultMsg:
    """Result from a completed batch."""
    tag: str = 'RESULT'
    batch_id: int = 0
    idx_path: str = ''          # path to written .idx file (or '' if empty)
    total_nnz: int = 0
    status: str = 'ok'          # 'ok' | 'error'
    error_msg: str = ''


@dataclass
class ReadyMsg:
    """Daemon is ready for work."""
    tag: str = 'READY'
    gpu_id: int = 0


# ============================================================
# LIFECYCLE: prestage + shutdown
# ============================================================

def prestage_gpu_daemons(n_gpus, available_gpu_ids=None, vram_limit_pct=0.85):
    """Fork one daemon per GPU. Each blocks on pipe.recv() until work arrives.

    Args:
        n_gpus: Number of GPUs to use
        available_gpu_ids: List of GPU device IDs (default: range(n_gpus))
        vram_limit_pct: CuPy memory pool ceiling as fraction of total VRAM

    Returns:
        List of DaemonHandle, one per GPU. Caller owns lifecycle.
    """
    if available_gpu_ids is None:
        available_gpu_ids = list(range(n_gpus))

    handles = []
    # spawn context creates a FRESH Python interpreter — no CUDA inheritance.
    # Each daemon sets its own CUDA_VISIBLE_DEVICES BEFORE importing CuPy.
    # Do NOT touch CUDA_VISIBLE_DEVICES in parent — it breaks spawn children.
    ctx = multiprocessing.get_context('spawn')
    for gpu_id in available_gpu_ids[:n_gpus]:
        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        p = ctx.Process(
            target=_gpu_daemon_main,
            args=(child_conn, gpu_id, vram_limit_pct),
        )
        p.start()
        child_conn.close()
        handles.append(DaemonHandle(pipe=parent_conn, process=p, gpu_id=gpu_id))

    # Wait for all daemons to signal READY
    for h in handles:
        try:
            msg = h.pipe.recv()
            # Accept both tuple ('READY', gpu_id) and dataclass ReadyMsg
            is_ready = (
                (isinstance(msg, tuple) and len(msg) >= 1 and msg[0] == 'READY') or
                (hasattr(msg, 'tag') and msg.tag == 'READY')
            )
            if is_ready:
                h.status = 'idle'
                _print(f"Daemon GPU-{h.gpu_id} ready")
            else:
                h.status = 'error'
                _print(f"Daemon GPU-{h.gpu_id} unexpected init msg: {msg}")
        except Exception as e:
            h.status = 'error'
            _print(f"Daemon GPU-{h.gpu_id} init failed: {e}")

    return handles


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
            h.process.kill()
        try:
            h.pipe.close()
        except OSError:
            pass
    _print(f"All {len(handles)} daemons shut down")


def _print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [gpu_daemon] {msg}", flush=True)


def _build_combined_csc_from_binary(left, right, n_left_cols, np):
    n_rows = int(left.shape[0])
    left_n_cols = int(left.shape[1])
    right_n_cols = int(right.shape[1])
    n_cols = left_n_cols + right_n_cols
    col_nnz = np.empty(n_cols, dtype=np.int64)

    for j in range(left_n_cols):
        col_nnz[j] = int(np.count_nonzero(left[:, j]))
    for j in range(right_n_cols):
        col_nnz[n_left_cols + j] = int(np.count_nonzero(right[:, j]))

    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(col_nnz, out=indptr[1:])
    total_nnz = int(indptr[-1])
    indices = np.empty(total_nnz, dtype=np.int32)

    for j in range(left_n_cols):
        start = int(indptr[j])
        rows = np.where(left[:, j] != 0)[0].astype(np.int32)
        indices[start:start + len(rows)] = rows
    for j in range(right_n_cols):
        out_col = n_left_cols + j
        start = int(indptr[out_col])
        rows = np.where(right[:, j] != 0)[0].astype(np.int32)
        indices[start:start + len(rows)] = rows

    return indptr, indices, col_nnz, n_rows


# ============================================================
# DAEMON ENTRY POINT (forked process)
# ============================================================

def _gpu_daemon_main(conn, gpu_id, vram_limit_pct):
    """Entry point for GPU daemon process.

    IMPORT ORDER IS CRITICAL:
    1. Set CUDA_VISIBLE_DEVICES BEFORE importing cupy
    2. Import ONLY numpy + cupy (NEVER scipy)
    3. Initialize CuPy pool with hard limit
    4. Compile CUDA kernel
    5. Enter work loop
    """
    import os, time, gc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')

    import numpy as np
    import cupy as cp

    def _log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [GPU-DAEMON-{gpu_id}] {msg}", flush=True)

    try:
        # Hard VRAM ceiling
        cp.cuda.Device(0).use()
        cp.cuda.set_pinned_memory_allocator(None)
        mempool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        total_vram = cp.cuda.Device(0).mem_info[1]
        mempool.set_limit(size=int(vram_limit_pct * total_vram))

        vram_gb = total_vram / (1024**3)
        _log(f"Initializing — VRAM: {vram_gb:.1f}GB, limit: {vram_limit_pct*100:.0f}%")

        # Compile CUDA sparse AND kernel
        kernel = cp.RawKernel(r"""
        extern "C" __global__
        void sparse_and_batch(
            const long long* __restrict__ indptr,
            const int* __restrict__ indices,
            const int* __restrict__ col_pairs,
            int*       result_indices,
            int*       result_counts,
            const int  max_out_per_pair
        ) {
            int pair = blockIdx.x;
            int col_a = col_pairs[pair * 2];
            int col_b = col_pairs[pair * 2 + 1];
            long long a = indptr[col_a], a_end = indptr[col_a + 1];
            long long b = indptr[col_b], b_end = indptr[col_b + 1];
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
        """, "sparse_and_batch")

        # State: current CSC on GPU
        indptr_gpu = None
        indices_gpu = None
        col_nnz_cpu = None
        n_rows = 0

        # Send tuple format (matches cross_supervisor protocol)
        conn.send(('READY', gpu_id))
        _log("Kernel compiled, entering work loop")

        # ── WORK LOOP ──
        batches_done = 0
        while True:
            msg = conn.recv()

            if msg is None:  # poison pill
                break

            # Normalize IPC format: accept both tuples (from cross_supervisor)
            # and dataclasses (from gpu_daemon's own protocol)
            if isinstance(msg, tuple):
                tag = msg[0]
            else:
                tag = getattr(msg, 'tag', None)

            if tag in ('RELOAD', 'RELOAD_SHM'):
                # Extract fields from either tuple or dataclass format
                _left_shm = None
                _right_shm = None
                if isinstance(msg, tuple) and tag == 'RELOAD':
                    _, left_npy_path, right_npy_path, n_left_cols = msg
                elif isinstance(msg, tuple) and tag == 'RELOAD_SHM':
                    _, left_meta, right_meta, n_left_cols = msg
                    _left_shm = shared_memory.SharedMemory(name=left_meta['name'])
                    _right_shm = shared_memory.SharedMemory(name=right_meta['name'])
                    left = np.ndarray(tuple(left_meta['shape']), dtype=np.dtype(left_meta['dtype']), buffer=_left_shm.buf)
                    right = np.ndarray(tuple(right_meta['shape']), dtype=np.dtype(right_meta['dtype']), buffer=_right_shm.buf)
                else:
                    left_npy_path = msg.left_npy_path
                    right_npy_path = msg.right_npy_path
                    n_left_cols = msg.n_left_cols

                # Free prior GPU arrays BEFORE loading new ones (memory leak fix)
                if indptr_gpu is not None:
                    del indptr_gpu, indices_gpu
                    cp.cuda.Stream.null.synchronize()
                    mempool.free_all_blocks()
                    pinned_pool.free_all_blocks()
                    indptr_gpu = None
                    indices_gpu = None

                # Upload new CSC to GPU (happens once per cross step)
                if tag == 'RELOAD':
                    _log(f"RELOAD: loading matrices from {left_npy_path}")
                    left = np.load(left_npy_path, mmap_mode='r')
                    right = np.load(right_npy_path, mmap_mode='r')
                else:
                    _log(f"RELOAD_SHM: attaching shared matrices {left_meta['name']} / {right_meta['name']}")
                indptr_np, indices_np, col_nnz_local, n_rows = _build_combined_csc_from_binary(
                    left, right, n_left_cols, np
                )
                n_cols = int(len(col_nnz_local))
                del left, right
                if _left_shm is not None:
                    _left_shm.close()
                if _right_shm is not None:
                    _right_shm.close()

                # Upload to GPU
                indptr_gpu = cp.asarray(indptr_np, dtype=cp.int64)
                indices_gpu = cp.asarray(indices_np)
                col_nnz_cpu = col_nnz_local
                del indptr_np, indices_np
                gc.collect()

                _log(f"CSC uploaded — {n_cols} cols, max_nnz={int(col_nnz_cpu.max())}, "
                     f"GPU mem: {cp.cuda.Device(0).mem_info[0]/1e9:.1f}GB free")
                # Send tuple format (matches cross_supervisor protocol)
                conn.send(('READY', gpu_id))
                continue

            if tag == 'BATCH':
                # Extract fields from either tuple or dataclass format
                if isinstance(msg, tuple):
                    _, batch_id, pairs, out_path, pair_id_offset = msg
                else:
                    batch_id = msg.batch_id
                    pairs = msg.pairs
                    out_path = msg.out_path
                    pair_id_offset = msg.pair_id_offset

                # Process intersection batch
                n_pairs = len(pairs)
                t0 = time.time()

                # Per-pair max_nnz for tight buffer sizing
                left_nnz = col_nnz_cpu[pairs[:, 0]]
                right_nnz = col_nnz_cpu[pairs[:, 1]]
                pair_max = np.minimum(left_nnz, right_nnz)
                max_out = int(pair_max.max()) if n_pairs > 0 else 0

                if max_out == 0:
                    conn.send(('RESULT', batch_id, '', 0, 'ok', ''))
                    continue
                max_out = min(max_out, n_rows)

                # VRAM-adaptive sub-batching
                result_bytes = n_pairs * max_out * 4
                mem_free = cp.cuda.Device(0).mem_info[0]
                if result_bytes > mem_free * 0.6:
                    sub_batch = max(100, int(n_pairs * (mem_free * 0.4) / result_bytes))
                else:
                    sub_batch = n_pairs

                total_nnz = 0
                idx_path = out_path
                idx_parent = os.path.dirname(idx_path)
                if idx_parent:
                    os.makedirs(idx_parent, exist_ok=True)

                from atomic_io import atomic_save
                with atomic_save(idx_path) as tmp_idx_path:
                    with open(tmp_idx_path, 'wb') as f:
                        # Write header
                        f.write(b'IDX1')
                        f.write(np.uint16(1).tobytes())
                        f.write(np.uint32(0).tobytes())  # n_records (patched at end)
                        f.write(np.uint32(n_rows).tobytes())

                        n_records = 0

                        for sb_start in range(0, n_pairs, sub_batch):
                            sb_end = min(sb_start + sub_batch, n_pairs)
                            sb_pairs = pairs[sb_start:sb_end]
                            n_sb = sb_end - sb_start
                            sb_max = int(pair_max[sb_start:sb_end].max())
                            if sb_max == 0:
                                continue
                            sb_max = min(sb_max, n_rows)

                            # Launch kernel
                            pairs_gpu = cp.asarray(sb_pairs.astype(np.int32).ravel())
                            result_idx = cp.zeros(n_sb * sb_max, dtype=cp.int32)
                            result_cnt = cp.zeros(n_sb, dtype=cp.int32)

                            kernel((n_sb,), (1,),
                                   (indptr_gpu, indices_gpu, pairs_gpu,
                                    result_idx, result_cnt, np.int32(sb_max)))
                            cp.cuda.Stream.null.synchronize()

                            # Bulk D2H transfer
                            counts_cpu = cp.asnumpy(result_cnt)
                            results_cpu = cp.asnumpy(result_idx)

                            # CPU post-process: build write buffer (batched f.write)
                            write_buf = bytearray()
                            for i in range(n_sb):
                                cnt = int(counts_cpu[i])
                                if cnt > 0:
                                    global_pair_id = pair_id_offset + sb_start + i
                                    offset = i * sb_max
                                    rows = results_cpu[offset:offset + cnt]
                                    write_buf += np.int32(global_pair_id).tobytes()
                                    write_buf += np.int32(cnt).tobytes()
                                    write_buf += rows.astype(np.int32).tobytes()
                                    total_nnz += cnt
                                    n_records += 1

                            if write_buf:
                                f.write(write_buf)

                            del pairs_gpu, result_idx, result_cnt

                        # Patch header with actual record count
                        f.seek(6)  # after magic(4) + version(2)
                        f.write(np.uint32(n_records).tobytes())

                if total_nnz == 0:
                    try:
                        os.remove(idx_path)
                    except OSError:
                        pass
                    idx_path = ''

                dt = time.time() - t0
                batches_done += 1

                # Send tuple format (matches cross_supervisor protocol)
                conn.send(('RESULT', batch_id, idx_path, total_nnz, 'ok', ''))

                # Free pool blocks every 10 batches (not every batch — perf)
                if batches_done % 10 == 0:
                    cp.cuda.Stream.null.synchronize()
                    mempool.free_all_blocks()
                    pinned_pool.free_all_blocks()

                # SAV-4/SAV-12 runtime IPC contract note:
                # KB_GAP: local KB searches returned no direct evidence for this
                # exact tuple/dataclass contract drift failure mode.
                # PERPLEXITY_SOURCE: Python multiprocessing docs (Connection.recv/
                # wait semantics), used to verify Pipe message handling assumptions:
                # https://docs.python.org/3/library/multiprocessing.html
                # Expected contract: BATCH may arrive as tuple
                # ('BATCH', batch_id, pairs, out_path, pair_id_offset) OR BatchMsg,
                # and daemon must always respond with RESULT without process death.
                # Broken behavior: logging used msg.batch_id (dataclass-only field),
                # which raises AttributeError for tuple IPC payloads and can kill the
                # daemon loop mid-run, forcing false fallback on later steps.
                # Fix scope: logging now uses normalized local batch_id from parsed
                # IPC payload; no feature-generation math or model logic changed.
                # QA still needed: multi-step RELOAD + BATCH soak to confirm no
                # step-3+ daemon deaths and no unintended legacy fallback.
                if batches_done % 3 == 0:
                    _log(f"Batch {batch_id}: {n_pairs} pairs → {total_nnz:,} nnz "
                         f"in {dt:.1f}s ({batches_done} done)")

    except Exception as e:
        import traceback
        _log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        try:
            conn.send(('RESULT', -1, '', 0, 'error', str(e)))
        except (BrokenPipeError, OSError):
            pass

    finally:
        try:
            if indptr_gpu is not None:
                del indptr_gpu, indices_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        _log(f"Daemon exiting ({batches_done if 'batches_done' in dir() else '?'} batches processed)")
