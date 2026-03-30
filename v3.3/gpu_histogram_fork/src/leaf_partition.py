"""
GPU leaf-to-row partition manager for histogram-based tree building.

LightGBM grows trees leaf-wise (best-first). Each split divides a leaf's
rows into two children. This module tracks which rows belong to which leaf
ON THE GPU, avoiding expensive CPU→GPU row index transfers per node.

For binary features (our sparse cross features): rows where feature[row]=1
go left, feature[row]=0 go right. The threshold is always 0.5.

Requires: CuPy with CUDA support.
"""

import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def is_available() -> bool:
    """Check if CUDA is available for GPU leaf partitioning."""
    if not CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CUDA kernel: partition rows of a parent leaf into left/right children
# ---------------------------------------------------------------------------
# For each row in the dataset, check if it belongs to the parent leaf.
# If so, look up the split feature in the CSR matrix:
#   - Binary search through the row's column indices for the split feature
#   - If found AND value > threshold → left child
#   - Otherwise → right child
#
# This kernel processes ALL rows in a single launch — rows not in the parent
# leaf are skipped (early exit). This avoids the cost of first extracting
# parent row indices to a separate buffer.
# ---------------------------------------------------------------------------
_PARTITION_KERNEL_SRC = r"""
extern "C" __global__
void partition_kernel(
    int*             leaf_assignment,  // int32 [n_rows] — read/write
    int              n_rows,
    int              parent_leaf,
    int              left_leaf,
    int              right_leaf,
    const long long* csr_indptr,       // int64 [n_rows + 1]
    const int*       csr_indices,      // int32 [nnz] — sorted per row
    int              split_feature,
    float            threshold          // 0.5 for binary features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Skip rows not in parent leaf
    if (leaf_assignment[row] != parent_leaf) return;

    // Binary search for split_feature in this row's CSR columns
    long long start = csr_indptr[row];
    long long end   = csr_indptr[row + 1];

    // Default: feature value is 0 (structural zero in CSR) → right child
    int goes_left = 0;

    // Binary search — CSR indices are sorted within each row
    long long lo = start;
    long long hi = end;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        int col = csr_indices[mid];
        if (col == split_feature) {
            // Found: implicit value is 1.0 (binary feature stored in CSR)
            // For binary: 1.0 > 0.5 → left
            goes_left = 1;
            break;
        } else if (col < split_feature) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    leaf_assignment[row] = goes_left ? left_leaf : right_leaf;
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: partition using explicit CSR data values (non-binary features)
# ---------------------------------------------------------------------------
# Same as above but reads the actual float value from csr_data and compares
# against threshold. Used when features are not strictly binary.
# ---------------------------------------------------------------------------
_PARTITION_VALUED_KERNEL_SRC = r"""
extern "C" __global__
void partition_valued_kernel(
    int*             leaf_assignment,  // int32 [n_rows] — read/write
    int              n_rows,
    int              parent_leaf,
    int              left_leaf,
    int              right_leaf,
    const long long* csr_indptr,       // int64 [n_rows + 1]
    const int*       csr_indices,      // int32 [nnz] — sorted per row
    const float*     csr_data,         // float32 [nnz]
    int              split_feature,
    float            threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    if (leaf_assignment[row] != parent_leaf) return;

    long long start = csr_indptr[row];
    long long end   = csr_indptr[row + 1];

    // Default: structural zero → value = 0.0
    float val = 0.0f;

    long long lo = start;
    long long hi = end;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        int col = csr_indices[mid];
        if (col == split_feature) {
            val = csr_data[mid];
            break;
        } else if (col < split_feature) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // LightGBM convention: value <= threshold → left, value > threshold → right
    // For binary with threshold=0.5: 0 <= 0.5 → left, 1 > 0.5 → right
    // BUT our design says: value=1 (feature ON) → left, value=0 → right
    // So we use: value > threshold → left
    leaf_assignment[row] = (val > threshold) ? left_leaf : right_leaf;
}
"""


class GPULeafPartition:
    """Manages leaf-to-row assignments on GPU for tree building.

    All state lives on GPU. The histogram builder calls get_leaf_rows()
    to obtain GPU arrays of row indices per leaf — no CPU↔GPU transfer
    needed between splits.

    Parameters
    ----------
    n_rows : int
        Total number of training rows.
    max_leaves : int
        Maximum leaves per tree (LightGBM num_leaves, default 63).
    device_id : int
        CUDA device to use.
    """

    def __init__(self, n_rows: int, max_leaves: int = 63, device_id: int = 0):
        if not is_available():
            raise RuntimeError("CUDA not available — cannot create GPULeafPartition")

        self.n_rows = n_rows
        self.max_leaves = max_leaves
        self.device_id = device_id
        self._next_leaf_id = 1  # leaf 0 is root, next split produces 1 and 2

        with cp.cuda.Device(device_id):
            # Which leaf each row belongs to — starts as all in leaf 0 (root)
            self.leaf_assignment = cp.zeros(n_rows, dtype=cp.int32)

            # Row count per leaf
            self.leaf_row_counts = cp.zeros(max_leaves, dtype=cp.int32)
            self.leaf_row_counts[0] = np.int32(n_rows)

        # Compile kernels once
        self._partition_kernel = cp.RawKernel(
            _PARTITION_KERNEL_SRC, "partition_kernel"
        )
        self._partition_valued_kernel = cp.RawKernel(
            _PARTITION_VALUED_KERNEL_SRC, "partition_valued_kernel"
        )

        # Cache for row indices per leaf — invalidated on each split
        self._row_index_cache: dict[int, cp.ndarray] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def split_leaf(
        self,
        leaf_id: int,
        feature_idx: int,
        threshold: float,
        gpu_csr_indptr: cp.ndarray,
        gpu_csr_indices: cp.ndarray,
        gpu_csr_data: cp.ndarray | None = None,
    ) -> tuple[int, int, int, int]:
        """Partition a leaf's rows into left/right children on GPU.

        For binary features (the common case for cross features):
            - rows where feature[row] = 1 (stored in CSR) → left child
            - rows where feature[row] = 0 (structural zero)  → right child
            - threshold should be 0.5

        Parameters
        ----------
        leaf_id : int
            The parent leaf to split.
        feature_idx : int
            Column index of the split feature in the CSR matrix.
        threshold : float
            Split threshold. For binary features, use 0.5.
        gpu_csr_indptr : cp.ndarray
            CSR indptr array, int64, already on GPU.
        gpu_csr_indices : cp.ndarray
            CSR column indices array, int32, already on GPU.
        gpu_csr_data : cp.ndarray or None
            CSR data values, float32, on GPU. If None, assumes binary
            features (all stored values = 1.0) and uses the fast
            binary-only kernel.

        Returns
        -------
        (left_leaf_id, right_leaf_id, left_count, right_count) : tuple[int, int, int, int]
        """
        if self._next_leaf_id + 1 >= self.max_leaves:
            raise RuntimeError(
                f"max_leaves={self.max_leaves} exceeded — cannot split leaf {leaf_id}"
            )

        left_leaf = self._next_leaf_id
        right_leaf = self._next_leaf_id + 1
        self._next_leaf_id += 2

        # Invalidate cached row indices for the parent (and children don't
        # exist yet, but clear the whole cache to be safe)
        self._row_index_cache.pop(leaf_id, None)

        with cp.cuda.Device(self.device_id):
            block_size = 256
            grid_size = (self.n_rows + block_size - 1) // block_size

            if gpu_csr_data is None:
                # Binary-only kernel: no data array needed
                self._partition_kernel(
                    (grid_size,),
                    (block_size,),
                    (
                        self.leaf_assignment,
                        np.int32(self.n_rows),
                        np.int32(leaf_id),
                        np.int32(left_leaf),
                        np.int32(right_leaf),
                        gpu_csr_indptr,
                        gpu_csr_indices,
                        np.int32(feature_idx),
                        np.float32(threshold),
                    ),
                )
            else:
                # Valued kernel: reads actual CSR data values
                self._partition_valued_kernel(
                    (grid_size,),
                    (block_size,),
                    (
                        self.leaf_assignment,
                        np.int32(self.n_rows),
                        np.int32(leaf_id),
                        np.int32(left_leaf),
                        np.int32(right_leaf),
                        gpu_csr_indptr,
                        gpu_csr_indices,
                        gpu_csr_data,
                        np.int32(feature_idx),
                        np.float32(threshold),
                    ),
                )

            cp.cuda.Device(self.device_id).synchronize()

            # Count rows in each child using GPU reduction
            left_count = int(cp.sum(self.leaf_assignment == left_leaf))
            right_count = int(cp.sum(self.leaf_assignment == right_leaf))

            # Update counts: parent goes to 0, children get their counts
            self.leaf_row_counts[leaf_id] = 0
            self.leaf_row_counts[left_leaf] = np.int32(left_count)
            self.leaf_row_counts[right_leaf] = np.int32(right_count)

        return left_leaf, right_leaf, left_count, right_count

    def get_leaf_rows(self, leaf_id: int) -> cp.ndarray:
        """Return GPU array of row indices belonging to this leaf.

        Uses CuPy nonzero() for extraction. Results are cached until
        the next split invalidates them.

        Parameters
        ----------
        leaf_id : int
            Leaf to query.

        Returns
        -------
        cp.ndarray
            int32 array of row indices on GPU.
        """
        if leaf_id in self._row_index_cache:
            return self._row_index_cache[leaf_id]

        with cp.cuda.Device(self.device_id):
            # nonzero returns a tuple of arrays; we want the first (only) axis
            (rows,) = cp.nonzero(self.leaf_assignment == leaf_id)
            rows = rows.astype(cp.int32)

        self._row_index_cache[leaf_id] = rows
        return rows

    def get_leaf_count(self, leaf_id: int) -> int:
        """Return row count for a leaf (from cached GPU counter, no kernel)."""
        return int(self.leaf_row_counts[leaf_id])

    def get_smaller_leaf(self, left_id: int, right_id: int) -> int:
        """Return the leaf ID with fewer rows (for histogram subtraction trick).

        LightGBM's histogram subtraction: build histogram for the smaller
        child, derive the larger child's histogram by subtracting from the
        parent. This halves the histogram build cost.

        Parameters
        ----------
        left_id : int
            Left child leaf ID.
        right_id : int
            Right child leaf ID.

        Returns
        -------
        int
            Leaf ID of the smaller child.
        """
        left_count = int(self.leaf_row_counts[left_id])
        right_count = int(self.leaf_row_counts[right_id])
        return left_id if left_count <= right_count else right_id

    def get_larger_leaf(self, left_id: int, right_id: int) -> int:
        """Return the leaf ID with more rows (complement of get_smaller_leaf)."""
        left_count = int(self.leaf_row_counts[left_id])
        right_count = int(self.leaf_row_counts[right_id])
        return right_id if left_count <= right_count else left_id

    def reset(self) -> None:
        """Reset to root state — all rows in leaf 0.

        Called at the start of each new tree in the boosting ensemble.
        """
        with cp.cuda.Device(self.device_id):
            self.leaf_assignment[:] = 0
            self.leaf_row_counts[:] = 0
            self.leaf_row_counts[0] = np.int32(self.n_rows)

        self._next_leaf_id = 1
        self._row_index_cache.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_active_leaves(self) -> list[tuple[int, int]]:
        """Return list of (leaf_id, row_count) for non-empty leaves."""
        counts_cpu = cp.asnumpy(self.leaf_row_counts)
        return [
            (int(i), int(c))
            for i, c in enumerate(counts_cpu)
            if c > 0
        ]

    def verify_partition(self) -> bool:
        """Verify that all rows are assigned and counts are consistent.

        Returns True if partition is valid, raises AssertionError otherwise.
        """
        total_from_counts = int(cp.sum(self.leaf_row_counts))
        assert total_from_counts == self.n_rows, (
            f"Row count mismatch: leaf_row_counts sum={total_from_counts}, "
            f"expected n_rows={self.n_rows}"
        )

        # Verify no row is assigned to an invalid leaf
        max_assigned = int(cp.max(self.leaf_assignment))
        assert max_assigned < self.max_leaves, (
            f"Row assigned to leaf {max_assigned} >= max_leaves={self.max_leaves}"
        )

        # Verify counts match actual assignments
        for leaf_id, expected_count in self.get_active_leaves():
            actual = int(cp.sum(self.leaf_assignment == leaf_id))
            assert actual == expected_count, (
                f"Leaf {leaf_id}: tracked count={expected_count}, "
                f"actual={actual}"
            )

        return True

    def __repr__(self) -> str:
        active = self.get_active_leaves()
        return (
            f"GPULeafPartition(n_rows={self.n_rows}, max_leaves={self.max_leaves}, "
            f"active_leaves={len(active)}, next_id={self._next_leaf_id})"
        )
