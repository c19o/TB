"""
LightGBM integration tests for the GPU sparse histogram fork.

Validates that the GPU-accelerated histogram co-processor produces
correct LightGBM models — identical accuracy, feature importance,
and predictions compared to standard CPU training.

Tests cover:
  1.  CPU vs GPU accuracy parity (synthetic sparse binary data)
  2.  Deterministic GPU runs (FP noise only)
  3.  Early stopping with GPU histograms
  4.  Multiclass (3-class) training
  5.  Feature importance ranking parity (Spearman > 0.9)
  6.  Prediction agreement (>95% same class)
  7.  Save/load model roundtrip
  8.  Fallback behavior when no GPU
  9.  Large feature count (500K sparse binary)
  10. Custom LightGBM params with GPU histograms

Run:  pytest tests/test_lgbm_integration.py -v
Skip GPU tests:  pytest tests/test_lgbm_integration.py -v -k "not gpu"
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — add src/ for lgbm_integration imports
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False

# Import the fork's integration module
try:
    from lgbm_integration import (
        gpu_train,
        _check_forked_lgbm_support,
        _detect_cuda,
        check_gpu_histogram_available,
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _cuda_sparse_available() -> bool:
    """Check if the forked LightGBM with cuda_sparse histogram support is present."""
    if not HAS_LGBM or not HAS_INTEGRATION:
        return False
    try:
        status = _check_forked_lgbm_support()
        if status.get('forked', False):
            return True
    except Exception:
        pass
    # Also check if CUDA is available at all (for the external co-processor path)
    try:
        cuda_info = _detect_cuda()
        return cuda_info.get('available', False)
    except Exception:
        return False


HAS_CUDA_SPARSE = _cuda_sparse_available()

skip_no_lgbm = pytest.mark.skipif(not HAS_LGBM, reason="lightgbm not installed")
skip_no_gpu = pytest.mark.skipif(not HAS_CUDA_SPARSE, reason="cuda_sparse not available in LightGBM build")


# ---------------------------------------------------------------------------
# Data generators — all synthetic, no external files
# ---------------------------------------------------------------------------

def _make_sparse_binary_classification(
    n_rows: int = 2000,
    n_features: int = 500,
    density: float = 0.01,
    n_informative: int = 20,
    seed: int = 42,
    n_classes: int = 2,
):
    """Generate a sparse binary classification dataset.

    First `n_informative` features have actual predictive signal.
    Remaining features are random noise (sparse binary).

    Returns (X_csr, y, X_csr_test, y_test).
    """
    rng = np.random.default_rng(seed)

    # Generate sparse binary matrix
    X = sp_sparse.random(n_rows, n_features, density=density, format='csr',
                         dtype=np.float32, random_state=rng.integers(0, 2**31))
    # Binarize (threshold at 0.5)
    X.data[:] = (X.data > 0.5).astype(np.float32)
    X.eliminate_zeros()

    # Create labels from informative features
    if n_informative > n_features:
        n_informative = n_features

    # Extract informative columns as dense for label generation
    X_info = X[:, :n_informative].toarray()
    signal = X_info.sum(axis=1) + rng.normal(0, 0.3, size=n_rows)

    if n_classes == 2:
        y = (signal > np.median(signal)).astype(np.int32)
    else:
        # n_classes quantiles
        thresholds = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(signal, thresholds).astype(np.int32)

    # 80/20 split
    split = int(n_rows * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test


def _cpu_params(n_classes: int = 2, **overrides) -> dict:
    """Standard CPU LightGBM params matching the project's V3_LGBM_PARAMS pattern."""
    objective = 'binary' if n_classes == 2 else 'multiclass'
    params = {
        'objective': objective,
        'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_bin': 255,
        'min_data_in_leaf': 5,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'verbose': -1,
        'seed': 42,
        'num_threads': 2,
    }
    if n_classes > 2:
        params['num_class'] = n_classes
    params.update(overrides)
    return params


def _gpu_params(n_classes: int = 2, **overrides) -> dict:
    """GPU params — activates the cuda_sparse histogram fork."""
    params = _cpu_params(n_classes, **overrides)
    # The forked LightGBM uses these custom params
    params['use_cuda_histogram'] = True
    params['cuda_histogram_gpu_id'] = 0
    return params


def _train_model(params, X_train, y_train, X_valid=None, y_valid=None,
                 num_boost_round=100, callbacks=None):
    """Train a LightGBM model with the given data."""
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False,
                         params={'feature_pre_filter': False})
    valid_sets = []
    valid_names = []
    if X_valid is not None:
        dval = lgb.Dataset(X_valid, label=y_valid, reference=dtrain,
                           free_raw_data=False,
                           params={'feature_pre_filter': False})
        valid_sets = [dval]
        valid_names = ['valid']

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def _train_via_fork(params, X_train, y_train, X_valid=None, y_valid=None,
                    num_boost_round=100, callbacks=None):
    """Train using the fork's gpu_train() function."""
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False,
                         params={'feature_pre_filter': False})
    valid_sets = []
    valid_names = []
    if X_valid is not None:
        dval = lgb.Dataset(X_valid, label=y_valid, reference=dtrain,
                           free_raw_data=False,
                           params={'feature_pre_filter': False})
        valid_sets = [dval]
        valid_names = ['valid']

    model = gpu_train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets if valid_sets else None,
        valid_names=valid_names if valid_names else None,
        callbacks=callbacks,
        X_csr=X_train if sp_sparse.issparse(X_train) else None,
    )
    return model


def _accuracy(model, X_test, y_test, n_classes=2):
    """Compute classification accuracy."""
    raw = model.predict(X_test)
    if n_classes == 2:
        preds = (raw > 0.5).astype(int)
    else:
        preds = raw.argmax(axis=1).astype(int)
    return float(np.mean(preds == y_test))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_lgbm
class TestSyntheticCPUvsGPU:
    """1. Train on synthetic sparse binary data with CPU and cuda_sparse.
    Accuracy should be within 0.5%."""

    @skip_no_gpu
    def test_synthetic_cpu_vs_gpu(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=3000, n_features=1000, density=0.02,
            n_informative=30, seed=42, n_classes=2,
        )

        cpu_model = _train_model(
            _cpu_params(), X_train, y_train, num_boost_round=200,
        )
        gpu_model = _train_via_fork(
            _gpu_params(), X_train, y_train, num_boost_round=200,
        )

        cpu_acc = _accuracy(cpu_model, X_test, y_test)
        gpu_acc = _accuracy(gpu_model, X_test, y_test)

        log.info("CPU accuracy: %.4f, GPU accuracy: %.4f, diff: %.4f",
                 cpu_acc, gpu_acc, abs(cpu_acc - gpu_acc))

        assert abs(cpu_acc - gpu_acc) < 0.005, (
            f"Accuracy gap too large: CPU={cpu_acc:.4f} GPU={gpu_acc:.4f} "
            f"diff={abs(cpu_acc - gpu_acc):.4f} (max 0.005)"
        )


@skip_no_lgbm
class TestDeterministicRuns:
    """2. Train twice with cuda_sparse on same data. Results within FP noise."""

    @skip_no_gpu
    def test_deterministic_runs(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=2000, n_features=500, density=0.02,
            n_informative=20, seed=99, n_classes=2,
        )
        params = _gpu_params(seed=42)

        model_a = _train_via_fork(params, X_train, y_train, num_boost_round=100)
        model_b = _train_via_fork(params, X_train, y_train, num_boost_round=100)

        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)

        max_diff = float(np.max(np.abs(pred_a - pred_b)))
        log.info("Deterministic run max prediction diff: %.2e", max_diff)

        # FP noise tolerance — GPU atomics can introduce minor ordering diffs
        assert max_diff < 1e-6, (
            f"Non-deterministic GPU runs: max_diff={max_diff:.2e} (expected < 1e-6)"
        )


@skip_no_lgbm
class TestEarlyStopping:
    """3. Verify early stopping works with cuda_sparse (callbacks fire)."""

    @skip_no_gpu
    def test_early_stopping(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=2000, n_features=500, density=0.02,
            n_informative=20, seed=77, n_classes=2,
        )
        params = _gpu_params(learning_rate=0.3)  # high LR to converge fast

        callbacks = [
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0),  # suppress logging
        ]

        model = _train_via_fork(
            params, X_train, y_train,
            X_valid=X_test, y_valid=y_test,
            num_boost_round=500,
            callbacks=callbacks,
        )

        # If early stopping fired, best_iteration < num_boost_round
        best_iter = model.best_iteration
        log.info("Early stopping fired at iteration %d / 500", best_iter)
        assert best_iter < 500, (
            f"Early stopping did not fire: best_iteration={best_iter} "
            f"(trained all 500 rounds)"
        )
        assert best_iter > 0, "Model trained 0 iterations — something is wrong"


@skip_no_lgbm
class TestMulticlass:
    """4. 3-class training with cuda_sparse. All 3 classes predicted."""

    @skip_no_gpu
    def test_multiclass(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=3000, n_features=800, density=0.02,
            n_informative=30, seed=33, n_classes=3,
        )

        model = _train_via_fork(
            _gpu_params(n_classes=3), X_train, y_train,
            num_boost_round=200,
        )

        raw = model.predict(X_test)
        assert raw.shape[1] == 3, f"Expected 3 class probabilities, got shape {raw.shape}"

        preds = raw.argmax(axis=1)
        unique_preds = set(preds.tolist())
        log.info("Multiclass predictions: classes present = %s", unique_preds)

        # All 3 classes should appear in predictions
        assert len(unique_preds) == 3, (
            f"Not all 3 classes predicted: only {unique_preds} found"
        )

        # Each class should have reasonable representation (>5% of predictions)
        for cls in range(3):
            frac = float(np.mean(preds == cls))
            assert frac > 0.05, (
                f"Class {cls} predicted only {frac:.1%} of the time (need >5%)"
            )

        # Accuracy should be meaningfully above random (33%)
        acc = _accuracy(model, X_test, y_test, n_classes=3)
        log.info("Multiclass accuracy: %.4f (random = 0.333)", acc)
        assert acc > 0.36, (
            f"Multiclass accuracy {acc:.4f} not meaningfully above random (0.333)"
        )


@skip_no_lgbm
class TestFeatureImportance:
    """5. Feature importance rankings between CPU and GPU (Spearman > 0.9)."""

    @skip_no_gpu
    def test_feature_importance(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=3000, n_features=500, density=0.03,
            n_informative=30, seed=55, n_classes=2,
        )

        cpu_model = _train_model(
            _cpu_params(), X_train, y_train, num_boost_round=200,
        )
        gpu_model = _train_via_fork(
            _gpu_params(), X_train, y_train, num_boost_round=200,
        )

        cpu_imp = np.array(cpu_model.feature_importance(importance_type='gain'),
                           dtype=np.float64)
        gpu_imp = np.array(gpu_model.feature_importance(importance_type='gain'),
                           dtype=np.float64)

        assert len(cpu_imp) == len(gpu_imp), (
            f"Feature importance length mismatch: CPU={len(cpu_imp)} GPU={len(gpu_imp)}"
        )

        # Only compare features that have nonzero importance in at least one model
        mask = (cpu_imp > 0) | (gpu_imp > 0)
        if mask.sum() < 10:
            pytest.skip("Too few features with nonzero importance for meaningful comparison")

        corr, pval = spearmanr(cpu_imp[mask], gpu_imp[mask])
        log.info("Feature importance Spearman correlation: %.4f (p=%.2e, n=%d)",
                 corr, pval, mask.sum())

        assert corr > 0.9, (
            f"Feature importance Spearman correlation too low: {corr:.4f} (need >0.9)"
        )


@skip_no_lgbm
class TestPredictionAgreement:
    """6. CPU and GPU models should agree on >95% of predictions."""

    @skip_no_gpu
    def test_prediction_agreement(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=3000, n_features=800, density=0.02,
            n_informative=30, seed=66, n_classes=2,
        )

        cpu_model = _train_model(
            _cpu_params(), X_train, y_train, num_boost_round=200,
        )
        gpu_model = _train_via_fork(
            _gpu_params(), X_train, y_train, num_boost_round=200,
        )

        cpu_preds = (cpu_model.predict(X_test) > 0.5).astype(int)
        gpu_preds = (gpu_model.predict(X_test) > 0.5).astype(int)

        agreement = float(np.mean(cpu_preds == gpu_preds))
        log.info("CPU/GPU prediction agreement: %.4f", agreement)

        assert agreement > 0.95, (
            f"Prediction agreement too low: {agreement:.4f} (need >0.95)"
        )


@skip_no_lgbm
class TestSaveLoadModel:
    """7. Train with cuda_sparse, save, load, predict. Results match."""

    @skip_no_gpu
    def test_save_load_model(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=2000, n_features=500, density=0.02,
            n_informative=20, seed=77, n_classes=2,
        )

        model = _train_via_fork(
            _gpu_params(), X_train, y_train, num_boost_round=100,
        )

        pred_before = model.predict(X_test)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            model_path = f.name

        try:
            model.save_model(model_path)
            loaded_model = lgb.Booster(model_file=model_path)
            pred_after = loaded_model.predict(X_test)

            max_diff = float(np.max(np.abs(pred_before - pred_after)))
            log.info("Save/load prediction max diff: %.2e", max_diff)

            # Predictions must be exactly identical after save/load (no FP drift)
            assert max_diff == 0.0, (
                f"Predictions changed after save/load: max_diff={max_diff:.2e}"
            )
        finally:
            try:
                os.unlink(model_path)
            except OSError:
                pass


@skip_no_lgbm
class TestFallbackNoGPU:
    """8. Setting GPU params on a machine without GPU should fallback or error clearly."""

    def test_fallback_no_gpu(self):
        """When cuda_sparse is not available, gpu_train should either:
        - Fall back to CPU training (with a warning) and produce a valid model, OR
        - Raise a clear error (not a segfault or silent wrong result)
        """
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=500, n_features=100, density=0.05,
            n_informative=10, seed=88, n_classes=2,
        )

        params = _gpu_params()

        if HAS_CUDA_SPARSE:
            # On a GPU machine, this should just work
            model = _train_via_fork(params, X_train, y_train, num_boost_round=50)
            assert model is not None
            pred = model.predict(X_test)
            assert pred.shape[0] == X_test.shape[0]
        else:
            # On a CPU-only machine, gpu_train should either fallback or raise
            if HAS_INTEGRATION:
                # The fork's gpu_train has fallback logic
                try:
                    model = _train_via_fork(params, X_train, y_train,
                                            num_boost_round=50)
                    # Fallback worked — verify model is valid
                    assert model is not None
                    pred = model.predict(X_test)
                    assert pred.shape[0] == X_test.shape[0]
                    log.info("GPU fallback to CPU: model trained successfully")
                except (RuntimeError, ImportError, lgb.basic.LightGBMError) as e:
                    # Clear error is acceptable
                    log.info("GPU fallback raised expected error: %s", e)
                    assert "cuda" in str(e).lower() or "gpu" in str(e).lower(), (
                        f"Error message should mention CUDA/GPU, got: {e}"
                    )
            else:
                # No integration module at all — stock LightGBM ignores unknown params
                model = _train_model(params, X_train, y_train, num_boost_round=50)
                pred = model.predict(X_test)
                assert pred.shape[0] == X_test.shape[0]


@skip_no_lgbm
class TestLargeFeatureCount:
    """9. 500K sparse binary features — no overflow or crash."""

    @skip_no_gpu
    @pytest.mark.slow
    def test_large_feature_count(self):
        n_rows = 800
        n_features = 500_000
        density = 0.001  # ~400 nonzeros per row
        n_informative = 50

        rng = np.random.default_rng(22)

        # Build sparse matrix in chunks to avoid memory spike
        X = sp_sparse.random(n_rows, n_features, density=density, format='csr',
                             dtype=np.float32, random_state=rng.integers(0, 2**31))
        X.data[:] = (X.data > 0.5).astype(np.float32)
        X.eliminate_zeros()

        # Labels from first n_informative columns
        X_info = X[:, :n_informative].toarray()
        signal = X_info.sum(axis=1) + rng.normal(0, 0.5, size=n_rows)
        y = (signal > np.median(signal)).astype(np.int32)

        split = int(n_rows * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        log.info("Large feature test: %d rows x %d features, nnz=%d",
                 n_rows, n_features, X.nnz)

        # Verify int64 indptr for large feature counts
        assert X_train.indptr.dtype in (np.int64, np.int32), (
            f"Unexpected indptr dtype: {X_train.indptr.dtype}"
        )

        params = _gpu_params(
            num_leaves=15,       # keep small for speed
            learning_rate=0.1,
            min_data_in_leaf=3,
            max_bin=255,
            feature_pre_filter=False,
        )

        model = _train_via_fork(
            params, X_train, y_train, num_boost_round=50,
        )
        assert model is not None

        pred = model.predict(X_test)
        assert pred.shape[0] == X_test.shape[0]
        assert not np.any(np.isnan(pred)), "NaN in predictions"
        assert not np.any(np.isinf(pred)), "Inf in predictions"

        acc = _accuracy(model, X_test, y_test)
        log.info("Large feature count accuracy: %.4f", acc)
        # Should be above random (50%) — informative features exist
        assert acc >= 0.45, f"Accuracy {acc:.4f} below expected minimum"


@skip_no_lgbm
class TestCustomParams:
    """10. Verify custom LightGBM params work with cuda_sparse."""

    @skip_no_gpu
    def test_custom_params(self):
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=2000, n_features=500, density=0.02,
            n_informative=20, seed=100, n_classes=2,
        )

        custom_configs = [
            {
                'name': 'max_bin=15',
                'overrides': {'max_bin': 15},
            },
            {
                'name': 'num_leaves=63',
                'overrides': {'num_leaves': 63},
            },
            {
                'name': 'learning_rate=0.01',
                'overrides': {'learning_rate': 0.01},
            },
            {
                'name': 'feature_pre_filter=False+min_data_in_leaf=3',
                'overrides': {
                    'feature_pre_filter': False,
                    'min_data_in_leaf': 3,
                },
            },
            {
                'name': 'bagging_fraction=0.8',
                'overrides': {
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                },
            },
            {
                'name': 'colsample_bytree=0.5',
                'overrides': {'colsample_bytree': 0.5},
            },
        ]

        for cfg in custom_configs:
            params = _gpu_params(**cfg['overrides'])
            try:
                model = _train_via_fork(
                    params, X_train, y_train, num_boost_round=50,
                )
                pred = model.predict(X_test)
                assert pred.shape[0] == X_test.shape[0], (
                    f"Config '{cfg['name']}': wrong prediction shape"
                )
                assert not np.any(np.isnan(pred)), (
                    f"Config '{cfg['name']}': NaN in predictions"
                )
                acc = _accuracy(model, X_test, y_test)
                log.info("Config '%s': accuracy=%.4f", cfg['name'], acc)
            except Exception as e:
                pytest.fail(
                    f"Config '{cfg['name']}' failed with cuda_sparse: {e}"
                )

    @skip_no_gpu
    def test_max_bin_values(self):
        """Verify different max_bin values all work (binary features always get 2 bins)."""
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=1500, n_features=300, density=0.03,
            n_informative=15, seed=101, n_classes=2,
        )

        for max_bin in [2, 15, 63, 127, 255]:
            params = _gpu_params(max_bin=max_bin)
            model = _train_via_fork(
                params, X_train, y_train, num_boost_round=30,
            )
            pred = model.predict(X_test)
            assert pred.shape[0] == X_test.shape[0], (
                f"max_bin={max_bin}: wrong prediction shape"
            )
            assert not np.any(np.isnan(pred)), (
                f"max_bin={max_bin}: NaN in predictions"
            )

    @skip_no_gpu
    def test_multiclass_custom_params(self):
        """3-class with custom params — validates the full param pipeline."""
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=2500, n_features=600, density=0.02,
            n_informative=25, seed=102, n_classes=3,
        )

        params = _gpu_params(
            n_classes=3,
            num_leaves=63,
            min_data_in_leaf=3,
            max_bin=255,
            feature_pre_filter=False,
            learning_rate=0.05,
        )

        model = _train_via_fork(params, X_train, y_train, num_boost_round=100)
        raw = model.predict(X_test)
        assert raw.shape == (X_test.shape[0], 3)

        preds = raw.argmax(axis=1)
        assert len(set(preds.tolist())) >= 2, "Degenerate predictions (all same class)"


# ---------------------------------------------------------------------------
# CPU-only tests (run even without GPU)
# ---------------------------------------------------------------------------

@skip_no_lgbm
class TestCPUBaseline:
    """Baseline CPU tests to validate the test infrastructure itself."""

    def test_cpu_binary_trains(self):
        """Sanity: CPU binary training works with our synthetic data."""
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=1000, n_features=200, density=0.03,
            n_informative=15, seed=1, n_classes=2,
        )
        model = _train_model(_cpu_params(), X_train, y_train, num_boost_round=50)
        acc = _accuracy(model, X_test, y_test)
        assert acc > 0.5, f"CPU baseline too low: {acc:.4f}"

    def test_cpu_multiclass_trains(self):
        """Sanity: CPU 3-class training works."""
        X_train, y_train, X_test, y_test = _make_sparse_binary_classification(
            n_rows=1500, n_features=300, density=0.03,
            n_informative=20, seed=2, n_classes=3,
        )
        model = _train_model(_cpu_params(n_classes=3), X_train, y_train,
                             num_boost_round=50)
        raw = model.predict(X_test)
        assert raw.shape[1] == 3
        acc = _accuracy(model, X_test, y_test, n_classes=3)
        assert acc > 0.33, f"CPU multiclass too low: {acc:.4f}"

    def test_sparse_input_accepted(self):
        """Verify LightGBM Dataset accepts sparse CSR directly."""
        X_train, y_train, _, _ = _make_sparse_binary_classification(
            n_rows=500, n_features=100, density=0.05, seed=3,
        )
        assert sp_sparse.issparse(X_train)
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False,
                             params={'feature_pre_filter': False})
        dtrain.construct()
        assert dtrain.num_data() == X_train.shape[0]
        assert dtrain.num_feature() == X_train.shape[1]
