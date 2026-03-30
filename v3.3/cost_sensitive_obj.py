"""
cost_sensitive_obj.py
---------------------
Custom cost-sensitive multiclass objective for LightGBM.
3-class BTC signal: LONG=0, SHORT=1, FLAT=2

Cost matrix C[true][pred]:
            pred=LONG  pred=SHORT  pred=FLAT
true=LONG       0          3          1       (wrong direction = 3x, missed opp = 1x)
true=SHORT      3          0          1       (wrong direction = 3x, missed opp = 1x)
true=FLAT       1          1          0       (false signal = 1x)

Regime-dependent scaling: multiply wrong-direction penalty by volatility regime weight.
High-vol/crash bars get 2x cost scaling (so wrong direction = 6x instead of 3x).

Usage with v2_multi_asset_trainer.py:
    from cost_sensitive_obj import make_cost_sensitive_obj, make_cost_sensitive_eval, make_sortino_eval

    # Build regime weights from HMM states or realized volatility
    regime_weights = compute_regime_weights(...)  # shape (N,), values 1.0-5.0

    # Create closures that capture regime info
    fobj  = make_cost_sensitive_obj(regime_weights)
    feval_cost = make_cost_sensitive_eval(regime_weights)
    feval_sortino = make_sortino_eval(forward_returns)

    # Modify params
    params['objective'] = 'custom'   # tells LightGBM to use fobj
    params['metric'] = 'None'        # disable built-in metric, use feval only
    params['num_class'] = 3          # MUST be set explicitly with custom obj

    model = lgb.train(
        params, dtrain,
        fobj=fobj,
        feval=[feval_cost, feval_sortino],
        ...
    )

    # CRITICAL: predict() returns raw logits with custom obj -- apply softmax manually
    raw = model.predict(X_test)
    probs = softmax_np(raw)
    pred_labels = np.argmax(probs, axis=1)

References:
    - LightGBM multiclass OVR layout: column-major (Fortran order)
    - Hessian: use softmax curvature surrogate (always positive) not exact (can be negative)
    - boost_from_average silently ignored with custom obj (LightGBM #7193)
    - arXiv 2509.04541: Finance-Grounded Optimization for Algorithmic Trading
"""

import numpy as np
import lightgbm as lgb

# ── Cost Matrix ──────────────────────────────────────────────────────────────
# C[true_class][predicted_class]  --  LONG=0, SHORT=1, FLAT=2
BASE_COST_MATRIX = np.array([
    [0.0, 3.0, 1.0],   # true=LONG:  correct=0, wrong_dir=3, missed=1
    [3.0, 0.0, 1.0],   # true=SHORT: correct=0, wrong_dir=3, missed=1
    [1.0, 1.0, 0.0],   # true=FLAT:  correct=0, false_sig=1, false_sig=1
], dtype=np.float64)

N_CLASSES = 3
HESSIAN_EPS = 1e-6     # floor for hessian to prevent division-by-zero in tree splits
SORTINO_CLIP = 10.0    # clip Sortino to prevent outlier-driven early stops


# ── Numerically Stable Softmax ───────────────────────────────────────────────

def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-shift for numerical stability. Shape (N, K) -> (N, K)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# ── Custom Objective Factory ─────────────────────────────────────────────────

def make_cost_sensitive_obj(regime_weights=None):
    """
    Factory that returns a cost-sensitive multiclass objective function.
    Uses closure to capture regime_weights (per-sample scaling for crash emphasis).

    Args:
        regime_weights: np.ndarray shape (N,) or None.
            Values >= 1.0. Higher = more penalty for errors on that bar.
            Example: crash bars = 3.0, trend bars = 1.5, range bars = 1.0
            If None, all bars weighted equally (pure asymmetric cost only).

    Returns:
        fobj function compatible with lgb.train(fobj=...)

    Math:
        Loss_i = SUM_k C[y_i, k] * p_{i,k} * regime_weight_i
        Gradient: g_{i,k} = p_{i,k} * (C_scaled[y_i, k] - E_i)
        Hessian:  h_{i,k} = max(p_{i,k} * (1 - p_{i,k}), eps)  [softmax surrogate]

        where E_i = SUM_k C_scaled[y_i, k] * p_{i,k} = expected cost at current probs
    """
    _regime_weights = regime_weights  # captured by closure

    def cost_sensitive_obj(preds: np.ndarray, train_data: lgb.Dataset):
        labels = train_data.get_label().astype(np.int32)
        n_samples = len(labels)

        # 1. Reshape logits from flat OVR -> (N, K)
        #    LightGBM packs multiclass preds column-major: [cls0_all, cls1_all, cls2_all]
        logits = preds.reshape(n_samples, N_CLASSES, order='F')

        # 2. Softmax probabilities
        probs = softmax_np(logits)

        # 3. Per-sample cost row: costs[i] = C[y_i, :]
        costs = BASE_COST_MATRIX[labels].copy()  # (N, K)

        # 4. Apply regime scaling (amplifies wrong-direction penalty in crash bars)
        if _regime_weights is not None:
            costs *= _regime_weights[:n_samples, np.newaxis]

        # 5. Expected cost: E_i = SUM_k C[y_i, k] * p_{i,k}
        expected_cost = (costs * probs).sum(axis=1, keepdims=True)  # (N, 1)

        # 6. Gradient: g_{i,k} = p_{i,k} * (C[y_i, k] - E_i)
        grad = probs * (costs - expected_cost)  # (N, K)

        # 7. Hessian: positive-definite softmax curvature surrogate
        #    Exact hessian can be negative -> training instability.
        #    Softmax curvature p*(1-p) is always in (0, 0.25] -> stable Newton steps.
        #    The gradient carries all cost-sensitive signal; hessian only scales step size.
        hess = np.maximum(probs * (1.0 - probs), HESSIAN_EPS)  # (N, K)

        # 8. Re-flatten to OVR layout (column-major)
        return grad.ravel(order='F'), hess.ravel(order='F')

    return cost_sensitive_obj


# ── Custom Eval: Mean Expected Cost ──────────────────────────────────────────

def make_cost_sensitive_eval(regime_weights=None):
    """
    Factory that returns a cost-sensitive eval metric for early stopping monitoring.
    Tracks mean expected cost (lower = better). Use alongside Sortino for stopping.

    Args:
        regime_weights: same as make_cost_sensitive_obj. If None, unweighted cost.

    Returns:
        feval function compatible with lgb.train(feval=...)
    """
    _regime_weights = regime_weights

    def cost_sensitive_eval(preds: np.ndarray, train_data: lgb.Dataset):
        labels = train_data.get_label().astype(np.int32)
        n_samples = len(labels)

        logits = preds.reshape(n_samples, N_CLASSES, order='F')
        probs = softmax_np(logits)
        costs = BASE_COST_MATRIX[labels].copy()

        if _regime_weights is not None:
            costs *= _regime_weights[:n_samples, np.newaxis]

        mean_cost = (costs * probs).sum(axis=1).mean()
        return 'mean_expected_cost', float(mean_cost), False  # lower is better

    return cost_sensitive_eval


# ── Custom Eval: Sortino Ratio ───────────────────────────────────────────────

def make_sortino_eval(forward_returns, target_return=0.0):
    """
    Factory that returns a Sortino ratio eval metric for early stopping.
    Higher = better. Early stopping should use this as the primary metric.

    Args:
        forward_returns: np.ndarray shape (N,) -- per-bar forward returns (e.g., next-bar % return).
            Must be aligned with the training data indices. For CPCV, pass the full array
            and the function uses train_data.get_label() indices implicitly.
        target_return: float, minimum acceptable return (default 0.0)

    Returns:
        feval function compatible with lgb.train(feval=...)
    """
    _returns = forward_returns.copy()
    _target = target_return

    def sortino_eval(preds: np.ndarray, train_data: lgb.Dataset):
        labels = train_data.get_label().astype(np.int32)
        n_samples = len(labels)

        logits = preds.reshape(n_samples, N_CLASSES, order='F')
        probs = softmax_np(logits)
        pred_labels = np.argmax(probs, axis=1)

        # Map signals to trade returns:
        #   LONG (0)  -> +forward_return (go long)
        #   SHORT (1) -> -forward_return (go short)
        #   FLAT (2)  -> 0.0 (no position)
        # Use only first n_samples of _returns (handles CPCV subsetting)
        returns_slice = _returns[:n_samples]
        trade_returns = np.where(
            pred_labels == 0, returns_slice,      # LONG
            np.where(pred_labels == 1, -returns_slice,  # SHORT
                     0.0)                          # FLAT
        )

        excess = trade_returns - _target
        downside = excess[excess < 0]

        if len(downside) < 2:
            downside_std = 1e-9
        else:
            downside_std = downside.std()

        sortino = excess.mean() / (downside_std + 1e-9)

        # Clip to prevent outlier-driven early stops in early iterations
        sortino = float(np.clip(sortino, -SORTINO_CLIP, SORTINO_CLIP))

        return 'sortino_ratio', sortino, True  # higher is better

    return sortino_eval


# ── Regime Weight Computation Helpers ────────────────────────────────────────

def compute_regime_weights_from_vol(returns: np.ndarray, window: int = 20,
                                     low_mult: float = 1.0, mid_mult: float = 1.5,
                                     high_mult: float = 3.0) -> np.ndarray:
    """
    Compute per-bar regime weights from realized volatility.
    High-vol bars get higher weight (penalizes errors during crashes more).

    Args:
        returns: np.ndarray of per-bar returns
        window: rolling window for volatility calculation
        low_mult: weight for low-vol bars (below 25th percentile)
        mid_mult: weight for mid-vol bars (25th-75th percentile)
        high_mult: weight for high-vol bars (above 75th percentile)

    Returns:
        np.ndarray shape (N,) of regime weights >= 1.0
    """
    n = len(returns)
    weights = np.ones(n, dtype=np.float64)

    if n < window:
        return weights

    # Rolling realized volatility
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        vol[i] = np.std(returns[i - window:i])

    # Fill NaN with median vol
    valid = ~np.isnan(vol)
    if valid.sum() > 0:
        vol[~valid] = np.nanmedian(vol)

    # Percentile-based regime assignment
    p25 = np.percentile(vol[valid], 25) if valid.sum() > 0 else 0.0
    p75 = np.percentile(vol[valid], 75) if valid.sum() > 0 else 1.0

    weights = np.where(vol < p25, low_mult,
              np.where(vol > p75, high_mult, mid_mult))

    return weights.astype(np.float64)


def compute_regime_weights_from_hmm(hmm_states: np.ndarray,
                                     state_weights: dict = None) -> np.ndarray:
    """
    Compute per-bar regime weights from HMM state labels.

    Args:
        hmm_states: np.ndarray of integer HMM state labels (e.g., 0=bull, 1=bear, 2=range)
        state_weights: dict mapping state -> weight. Default: {0: 1.0, 1: 3.0, 2: 1.0}
            (bear/crash state penalized 3x more)

    Returns:
        np.ndarray shape (N,) of regime weights >= 1.0
    """
    if state_weights is None:
        # Default: bear state (typically highest vol) gets 3x weight
        state_weights = {0: 1.0, 1: 3.0, 2: 1.0}

    weights = np.ones(len(hmm_states), dtype=np.float64)
    for state, w in state_weights.items():
        weights[hmm_states == state] = w

    return weights


# ── Integration Helper ───────────────────────────────────────────────────────

def get_custom_obj_params(base_params: dict) -> dict:
    """
    Modify LightGBM params dict for custom objective usage.
    Call this before lgb.train() when using cost_sensitive_obj.

    Args:
        base_params: existing V3_LGBM_PARAMS dict (will be copied, not mutated)

    Returns:
        Modified params dict ready for custom objective training
    """
    params = base_params.copy()

    # Custom objective mode
    params['objective'] = 'custom'  # tells LightGBM to use fobj parameter
    params['metric'] = 'None'       # disable built-in metrics, use feval only
    params['num_class'] = N_CLASSES  # MUST be set explicitly with custom obj

    # boost_from_average is silently ignored with custom obj (LightGBM #7193)
    # Remove it to avoid confusion
    params.pop('boost_from_average', None)

    # first_metric_only: early stopping uses FIRST feval only
    # Put sortino_eval first in feval list so it drives early stopping
    params['first_metric_only'] = True

    return params
