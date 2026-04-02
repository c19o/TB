#!/usr/bin/env python3
"""
cloud_run_tf.py  Run full pipeline for ONE timeframe on cloud.

1w Cloud Runtime Contract (explicit):
- Code executes from immutable `CODE_ROOT`.
- Shared DBs live under `SHARED_DB_ROOT`.
- Run-produced outputs live under `ARTIFACT_ROOT`.
- Heartbeat and run control files live under `RUN_ROOT`.
- cuDF available -> GPU path; cuDF missing -> CPU fallback via `ALLOW_CPU=1`.
- cuDF/cuML may be absent on CUDA 13-era hosts without forcing a whole-stack CPU fallback.

Usage: python -u cloud_run_tf.py --tf 1w

Tar should extract flat to /workspace/ with:
  - All .py, .json, .pkl files from v3.3
  - features_BTC_{tf}.parquet (or will rebuild if missing/incomplete)
  - btc_prices.db (root version with BTC/USDT, or multi_asset_prices.db to be fixed)
  - V1 DBs (needed only if feature rebuild required)
  - kp_history_gfz.txt (needed only if feature rebuild required)

Steps:
  0. Kill stale processes, install deps
  1. Fix btc_prices.db symbol format if needed
  2. Rebuild features if parquet missing or < 2000 cols
  3. Build crosses (v2_cross_generator.py --symbol BTC --save-sparse)
  4. Baseline train (ml_multi_tf.py --tf TF)
  5. Optuna hyperparameter search (saves optuna_configs_{tf}.json params only, no model)
  6. Retrain with winning params
  7-10. Optimizer, meta, LSTM, PBO, audit
  11. SHAP analysis + final artifact verification
"""
import os, sys, subprocess, time, json, glob, sqlite3, threading, importlib.util, shlex, shutil
from path_contract import CODE_ROOT, ARTIFACT_ROOT, RUN_ROOT, SHARED_DB_ROOT, V1_ROOT, artifact_path, db_path, run_path, ensure_runtime_dirs
try:
    from pipeline_contract import heartbeat_statuses as _contract_heartbeat_statuses
    from pipeline_contract import load_timeframe_contract as _load_timeframe_contract
    from pipeline_contract import phase_degradation_policy as _contract_phase_degradation_policy
    from pipeline_contract import phase_min_parallelism as _contract_phase_min_parallelism
except Exception:
    _contract_heartbeat_statuses = None
    _load_timeframe_contract = None
    _contract_phase_degradation_policy = None
    _contract_phase_min_parallelism = None
try:
    from deploy_profiles import env_defaults as _deploy_env_defaults
    from deploy_profiles import execution_mode as _deploy_execution_mode
except Exception:
    _deploy_env_defaults = None
    _deploy_execution_mode = None
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

os.environ['PYTHONUNBUFFERED'] = '1'
# ALLOW_CPU=1 is fallback-only: enable it automatically only when cuDF is unavailable.
# Do NOT force it on GPU-capable cloud boxes, because ml_multi_tf.py treats it as a CPU-only hint.
if 'ALLOW_CPU' not in os.environ and importlib.util.find_spec('cudf') is None:
    os.environ['ALLOW_CPU'] = '1'
# SAV-53: OMP_NUM_THREADS=4 MUST be set BEFORE any numpy/scipy/LightGBM import in subprocesses.
# Prevents thread exhaustion during cross generation on cloud machines (all 5 TFs).
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'FALSE'
# FIX #43: PyTorch expandable segments  reduces fragmentation-induced OOM on LSTM/meta steps
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# CuPy Blackwell (sm_120) compat: PTX JIT lets CUDA 13.0 driver compile for sm_120
os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')
os.environ.setdefault('SAVAGE22_ARTIFACT_DIR', os.environ.get('V30_DATA_DIR', ARTIFACT_ROOT))
os.environ.setdefault('V30_DATA_DIR', os.environ['SAVAGE22_ARTIFACT_DIR'])
os.environ.setdefault('SAVAGE22_DB_DIR', SHARED_DB_ROOT)
os.environ.setdefault('SAVAGE22_V1_DIR', V1_ROOT)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = CODE_ROOT
ensure_runtime_dirs()
os.chdir(PROJECT_DIR)

#  OPT: Transparent Huge Pages  reduces TLB misses for large sparse matrices 
try:
    with open('/sys/kernel/mm/transparent_hugepage/enabled', 'w') as f:
        f.write('always')
except (PermissionError, FileNotFoundError, OSError):
    pass  # container may not have permission

#  OPT: jemalloc  better malloc for sparse alloc/free patterns (reduces fragmentation) 
_jemalloc_paths = [
    '/usr/lib/x86_64-linux-gnu/libjemalloc.so.2',
    '/usr/lib/x86_64-linux-gnu/libjemalloc.so',
    '/usr/lib/libjemalloc.so.2',
    '/usr/lib/libjemalloc.so',
]
_jemalloc_found = None
for _jp in _jemalloc_paths:
    if os.path.exists(_jp):
        _jemalloc_found = _jp
        break
if _jemalloc_found:
    _existing_preload = os.environ.get('LD_PRELOAD', '')
    if 'jemalloc' not in _existing_preload:
        os.environ['LD_PRELOAD'] = f"{_jemalloc_found}:{_existing_preload}" if _existing_preload else _jemalloc_found
else:
    _jemalloc_found = None  # will log warning after log() is defined

#  Fix #5: jemalloc tuning  reduce fragmentation for sparse alloc/free patterns 
if _jemalloc_found:
    os.environ['MALLOC_CONF'] = 'background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000,narenas:32,tcache_max:4096,metadata_thp:auto'

#  Fix #6: NUMA balancing off  prevents page migration storms during cross gen 
try:
    with open('/proc/sys/kernel/numa_balancing', 'w') as f:
        f.write('0')
    print("  NUMA auto-balancing: disabled (prevents page migration storms)")
except (PermissionError, FileNotFoundError, OSError):
    pass

#  Fix #7: Dirty page limits + swappiness  keep CSR in RAM 
for path, val, desc in [
    ('/proc/sys/vm/swappiness', '10', 'swappiness=10 (keep CSR in RAM)'),
    ('/proc/sys/vm/dirty_background_bytes', str(int(1.5e9)), 'dirty_bg=1.5GB'),
    ('/proc/sys/vm/dirty_bytes', str(int(4e9)), 'dirty=4GB'),
]:
    try:
        with open(path, 'w') as f:
            f.write(val)
        print(f"  Kernel: {desc}")
    except (PermissionError, FileNotFoundError, OSError):
        pass

def _script(name):
    """Resolve script path  check CWD first, then script directory."""
    if os.path.exists(name):
        return name
    alt = os.path.join(_SCRIPT_DIR, name)
    if os.path.exists(alt):
        return alt
    return name  # let it fail with clear error

TF = sys.argv[sys.argv.index('--tf') + 1] if '--tf' in sys.argv else '1d'
ASSEMBLY_LINE = ('--assembly-line' in sys.argv) or (os.environ.get('SAVAGE22_AUTO_ASSEMBLY', '0') == '1')

def _parse_resume_from(argv):
    for i, token in enumerate(argv):
        if token == '--resume-from' and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith('--resume-from='):
            return token.split('=', 1)[1]
    return os.environ.get('SAVAGE22_RESUME_FROM', '').strip()

_MANAGED_RUN = any(os.environ.get(name) for name in ('SAVAGE22_RUN_DIR', 'SAVAGE22_ARTIFACT_DIR'))
if _MANAGED_RUN:
    if ARTIFACT_ROOT == CODE_ROOT or RUN_ROOT == CODE_ROOT or RUN_ROOT == ARTIFACT_ROOT:
        raise SystemExit(
            f"Managed run requires separate roots: CODE_ROOT={CODE_ROOT}, "
            f"ARTIFACT_ROOT={ARTIFACT_ROOT}, RUN_ROOT={RUN_ROOT}"
        )

def _load_first_json(paths):
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f), path
        except Exception:
            continue
    return None, None


_PIPELINE_CONTRACT_SOURCE = os.path.join(_SCRIPT_DIR, 'contracts', 'pipeline_contract.json')

def _default_tf_contract(tf_name):
    cross_required = tf_name != '1w'
    cross_artifacts = []
    if cross_required:
        cross_artifacts = [
            f'v2_crosses_BTC_{tf_name}.npz',
            f'v2_cross_names_BTC_{tf_name}.json',
            f'inference_{tf_name}_thresholds.json',
            f'inference_{tf_name}_cross_pairs.npz',
            f'inference_{tf_name}_ctx_names.json',
            f'inference_{tf_name}_base_cols.json',
            f'inference_{tf_name}_cross_names.json',
        ]
    train_artifacts = [
        f'model_{tf_name}.json',
        f'cpcv_oos_predictions_{tf_name}.pkl',
        f'platt_{tf_name}.pkl',
    ]
    optuna_artifacts = [
        f'optuna_configs_{tf_name}.json',
        f'lgbm_dataset_{tf_name}.bin',
    ]
    phases = {
        'step0_preflight': {
            'phase_seq': 0,
            'required_artifacts': [],
            'policy': 'required',
        },
        'step1_features': {
            'phase_seq': 1,
            'required_artifacts': [f'features_BTC_{tf_name}.parquet'],
            'policy': 'required',
        },
        'step2_crosses': {
            'phase_seq': 2,
            'required_artifacts': cross_artifacts,
            'policy': 'required' if cross_required else 'skip',
            'notes': '1w maintained runs skip crosses by contract' if not cross_required else '',
        },
        'step3_baseline': {
            'phase_seq': 3,
            'required_artifacts': train_artifacts[:2],
            'policy': 'required',
        },
        'step4_optuna': {
            'phase_seq': 4,
            'required_artifacts': optuna_artifacts,
            'policy': 'required',
        },
        'step5_retrain': {
            'phase_seq': 5,
            'required_artifacts': train_artifacts,
            'policy': 'required',
        },
        'step6_optimizer': {
            'phase_seq': 6,
            'required_artifacts': [f'optuna_configs_{tf_name}.json'],
            'policy': 'required',
        },
        'complete': {
            'phase_seq': 7,
            'required_artifacts': (
                [f'features_BTC_{tf_name}.parquet']
                + cross_artifacts
                + optuna_artifacts
                + train_artifacts
            ),
            'policy': 'required',
        },
    }
    return {
        'tf': tf_name,
        'heartbeat_statuses': ['running', 'validated', 'failed', 'complete'],
        'skip_crosses': not cross_required,
        'phases': phases,
    }

def _build_tf_contract(tf_name):
    if _load_timeframe_contract is not None:
        try:
            loaded = _load_timeframe_contract(tf_name)
            phases = {name: dict(cfg) for name, cfg in loaded.get('phases', {}).items()}
            return {
                'tf': tf_name,
                'source': _PIPELINE_CONTRACT_SOURCE,
                'heartbeat_statuses': _contract_heartbeat_statuses() if callable(_contract_heartbeat_statuses) else ['running', 'validated', 'failed', 'complete'],
                'skip_crosses': str(loaded.get('cross_policy', 'required')) != 'required',
                'phases': phases,
            }
        except Exception:
            pass
    return _default_tf_contract(tf_name)


def _tf_profile_env_defaults():
    if _deploy_env_defaults is None:
        return {}
    try:
        payload = _deploy_env_defaults(TF)
    except Exception:
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def _tf_execution_mode():
    if _deploy_execution_mode is None:
        return ''
    try:
        return str(_deploy_execution_mode(TF)).strip()
    except Exception:
        return ''


_TF_CONTRACT = _build_tf_contract(TF)

_PHASE_ALIASES = {
    'Deployment verification': 'step0_preflight',
    'Pre-flight validation': 'step0_preflight',
    'Install deps': 'step0_preflight',
    'Install numactl': 'step0_preflight',
    'Rebuild 1w features': 'step1_features',
    'Build 1w features': 'step1_features',
    'Rebuild 1d features': 'step1_features',
    'Build 1d features': 'step1_features',
    'Rebuild 4h features': 'step1_features',
    'Build 4h features': 'step1_features',
    'Rebuild 1h features': 'step1_features',
    'Build 1h features': 'step1_features',
    'Rebuild 15m features': 'step1_features',
    'Build 15m features': 'step1_features',
    'Build 1w crosses': 'step2_crosses',
    'Build 1d crosses': 'step2_crosses',
    'Build 4h crosses': 'step2_crosses',
    'Build 1h crosses': 'step2_crosses',
    'Build 15m crosses': 'step2_crosses',
    'Optuna search 1w': 'step4_optuna',
    'Optuna search 1d': 'step4_optuna',
    'Optuna search 4h': 'step4_optuna',
    'Optuna search 1h': 'step4_optuna',
    'Optuna search 15m': 'step4_optuna',
    'Train 1w': 'step5_retrain',
    'Train 1d': 'step5_retrain',
    'Train 4h': 'step5_retrain',
    'Train 1h': 'step5_retrain',
    'Train 15m': 'step5_retrain',
    'Optimizer 1w': 'step6_optimizer',
    'Optimizer 1d': 'step6_optimizer',
    'Optimizer 4h': 'step6_optimizer',
    'Optimizer 1h': 'step6_optimizer',
    'Optimizer 15m': 'step6_optimizer',
    'Meta 1w': 'step7_meta',
    'Meta 1d': 'step7_meta',
    'Meta 4h': 'step7_meta',
    'Meta 1h': 'step7_meta',
    'Meta 15m': 'step7_meta',
    'LSTM 1w': 'step8_lstm',
    'LSTM 1d': 'step8_lstm',
    'LSTM 4h': 'step8_lstm',
    'LSTM 1h': 'step8_lstm',
    'LSTM 15m': 'step8_lstm',
    'PBO 1w': 'step9_pbo',
    'PBO 1d': 'step9_pbo',
    'PBO 4h': 'step9_pbo',
    'PBO 1h': 'step9_pbo',
    'PBO 15m': 'step9_pbo',
    'Audit 1w': 'step10_audit',
    'Audit 1d': 'step10_audit',
    'Audit 4h': 'step10_audit',
    'Audit 1h': 'step10_audit',
    'Audit 15m': 'step10_audit',
    'SHAP cross feature analysis': 'step11_shap',
    'final-summary': 'complete',
    '5': 'step5_retrain',
    'step5': 'step5_retrain',
    'retrain': 'step5_retrain',
    'train': 'step5_retrain',
}

def _canonical_phase(name):
    return _PHASE_ALIASES.get(name, name)

def _phase_def(phase):
    return _TF_CONTRACT.get('phases', {}).get(phase, {})

def _phase_policy(phase):
    return str(_phase_def(phase).get('policy', 'required')).strip().lower()

def _phase_should_run(phase):
    return _phase_policy(phase) not in ('skip', 'skipped')

def _phase_skip_reason(phase):
    return str(_phase_def(phase).get('skip_reason', '')).strip()

def _phase_seq(phase):
    try:
        return int(_phase_def(phase).get('phase_seq'))
    except Exception:
        return None

def _phase_required_artifacts(phase):
    return list(_phase_def(phase).get('required_artifacts', []))

def _phase_artifact_paths(phase):
    return [artifact_path(name) for name in _phase_required_artifacts(phase)]

def _missing_phase_artifacts(phase):
    return [path for path in _phase_artifact_paths(phase) if not os.path.exists(path)]

def _phase_sentinel_path(phase):
    return run_path(f'{phase}.validated.json')

def _phase_failure_path(phase):
    return run_path(f'{phase}.failed.json')

def _write_phase_sentinel(phase, status='validated', note=None):
    try:
        payload = {
            'run_id': RUN_ID,
            'tf': TF,
            'phase': phase,
            'phase_seq': PHASE_SEQ,
            'status': status,
            'artifact_root': ARTIFACT_ROOT,
            'run_root': RUN_ROOT,
            'code_root': CODE_ROOT,
            'shared_db_root': SHARED_DB_ROOT,
            'artifact_contract': _TF_CONTRACT.get('source'),
            'release_manifest': run_path('release_manifest.json'),
            'required_artifacts': _phase_required_artifacts(phase),
            'required_artifact_paths': _phase_artifact_paths(phase),
            'updated_at': time.time(),
        }
        if note:
            payload['note'] = note
        with open(_phase_sentinel_path(phase), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        log(f"  WARNING: failed to write phase sentinel for {phase}: {e}")

def _write_phase_failure(phase, reason, detail=None, extra=None, status='failed'):
    try:
        payload = {
            'run_id': RUN_ID,
            'tf': TF,
            'phase': phase,
            'phase_seq': _phase_seq(phase),
            'status': status,
            'reason': reason,
            'artifact_root': ARTIFACT_ROOT,
            'run_root': RUN_ROOT,
            'code_root': CODE_ROOT,
            'shared_db_root': SHARED_DB_ROOT,
            'artifact_contract': _TF_CONTRACT.get('source'),
            'release_manifest': run_path('release_manifest.json'),
            'required_artifacts': _phase_required_artifacts(phase),
            'required_artifact_paths': _phase_artifact_paths(phase),
            'updated_at': time.time(),
        }
        if detail:
            payload['detail'] = detail
        if extra:
            payload.update(extra)
        with open(_phase_failure_path(phase), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        log(f"  WARNING: failed to write phase failure for {phase}: {e}")

def _clear_phase_failure(phase):
    try:
        os.remove(_phase_failure_path(phase))
    except OSError:
        pass

def _normalize_resume_phase(raw_phase):
    if not raw_phase:
        return ''
    return _canonical_phase(str(raw_phase).strip())

RESUME_FROM_PHASE = _normalize_resume_phase(_parse_resume_from(sys.argv))

if RESUME_FROM_PHASE and RESUME_FROM_PHASE not in _TF_CONTRACT.get('phases', {}):
    raise SystemExit(
        f"Unknown resume phase {RESUME_FROM_PHASE!r} for tf={TF}. "
        f"Known phases: {', '.join(sorted(_TF_CONTRACT.get('phases', {}).keys()))}"
    )

def _phase_is_before_resume(phase):
    if not RESUME_FROM_PHASE:
        return False
    resume_seq = _phase_seq(RESUME_FROM_PHASE)
    phase_seq = _phase_seq(phase)
    if resume_seq is None or phase_seq is None:
        return False
    return phase_seq < resume_seq

def _resume_tolerated_missing_artifacts(phase, missing):
    tolerated = []
    if phase == 'step4_optuna' and missing:
        _dataset_bin = artifact_path(f'lgbm_dataset_{TF}.bin')
        _optuna_cfg = artifact_path(f'optuna_configs_{TF}.json')
        for path in missing:
            if path == _dataset_bin and os.path.exists(_optuna_cfg):
                tolerated.append(path)
    return tolerated

def _resume_validate_phase(phase, note):
    missing = _missing_phase_artifacts(phase)
    tolerated = _resume_tolerated_missing_artifacts(phase, missing)
    missing = [path for path in missing if path not in tolerated]
    if tolerated:
        log(
            f"  Resume tolerance: allowing cached artifact miss for {phase}: "
            + ', '.join(os.path.basename(p) for p in tolerated)
        )
    if missing:
        _detail = f"Cannot resume from {RESUME_FROM_PHASE}: {phase} missing required artifacts"
        log(f"*** CRITICAL: {_detail}: {', '.join(os.path.basename(p) for p in missing)} ***")
        _write_phase_failure(
            phase,
            reason='resume-boundary-missing-artifacts',
            detail=_detail,
            extra={'missing_artifacts': missing, 'resume_from': RESUME_FROM_PHASE},
        )
        sys.exit(1)
    _set_step(phase)
    log(f"Resume boundary: reusing {phase} artifacts")
    _write_phase_sentinel(phase, status='validated', note=note)
    _mark_progress(f"{phase}:validated")

def _artifact_state(paths):
    state = {}
    for path in paths:
        if os.path.exists(path):
            try:
                st = os.stat(path)
                state[path] = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
            except OSError:
                state[path] = {'size': None, 'mtime_ns': None}
        else:
            state[path] = None
    return state

def _artifact_changed_since(path, before_state, started_ns):
    now = _artifact_state([path]).get(path)
    prev = before_state.get(path)
    if now is None:
        return False
    if prev is None:
        return True
    if now.get('mtime_ns') is not None and now['mtime_ns'] >= started_ns:
        return True
    return now != prev

def _backup_existing_artifacts(paths, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for path in paths:
        if not os.path.exists(path):
            continue
        dest = os.path.join(dest_dir, os.path.basename(path))
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def _load_json_file(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_pickle_file(path):
    if not os.path.exists(path):
        return None
    try:
        import pickle
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _step5_runtime_failure_context():
    json_paths = [
        artifact_path(f'lgbm_ckpt_{TF}_final.meta.json'),
        artifact_path(f'cpcv_checkpoint_{TF}.meta.json'),
    ]
    for path in json_paths:
        payload = _load_json_file(path)
        if payload:
            payload = dict(payload)
            payload.setdefault('metadata_path', path)
            return payload
    ckpt_payload = _load_pickle_file(artifact_path(f'cpcv_checkpoint_{TF}.pkl'))
    if ckpt_payload and isinstance(ckpt_payload.get('meta'), dict):
        payload = dict(ckpt_payload['meta'])
        payload.setdefault('metadata_path', artifact_path(f'cpcv_checkpoint_{TF}.pkl'))
        return payload
    return {}


def _step5_failure_reason(context):
    err_type = str(context.get('error_type', '')).strip()
    stage = str(context.get('stage', '')).strip()
    if err_type == 'BrokenProcessPool':
        return 'parallel_pool_broken'
    if context.get('status') == 'failed' and stage == 'cpcv_parallel':
        return 'worker_native_crash'
    if context.get('status') == 'failed' and stage:
        return f'{stage}-failed'
    return 'command-failed'


def _step5_failure_detail(context):
    if not context:
        return ''
    stage = str(context.get('stage', 'step5')).strip() or 'step5'
    err_type = str(context.get('error_type', '')).strip()
    err = str(context.get('error', '')).strip()
    completed = context.get('completed_fold_count')
    total = context.get('total_folds')
    if completed is not None and total is not None:
        prefix = f"{stage} failed after {completed}/{total} completed folds"
    else:
        prefix = f"{stage} failed"
    if err_type and err:
        return f"{prefix}: {err_type}: {err}"
    if err_type:
        return f"{prefix}: {err_type}"
    if err:
        return f"{prefix}: {err}"
    return prefix


def _train_phase_env_map():
    env_map = _tf_profile_env_defaults()
    mode = _tf_execution_mode()
    if mode.startswith('cpu_first'):
        env_map['ALLOW_CPU'] = '1'
    return env_map

def _emit_runtime_contract():
    cudf_available = importlib.util.find_spec('cudf') is not None
    allow_cpu = os.environ.get('ALLOW_CPU') == '1'
    if (not cudf_available) and (not allow_cpu):
        os.environ['ALLOW_CPU'] = '1'
        allow_cpu = True
    mode = 'gpu' if (cudf_available and not allow_cpu) else 'cpu-fallback'
    print(f"  Runtime contract: tf={TF}, cudf={'present' if cudf_available else 'missing'}, allow_cpu={allow_cpu}, mode={mode}")
    if cudf_available and allow_cpu:
        print("  WARNING: cudf is present but ALLOW_CPU=1 is forcing CPU mode")
    if (not cudf_available) and allow_cpu:
        print("  ALLOW_CPU contract active: cuDF unavailable, running CPU fallback")

#  Assembly-line: TF sequence for prefetching next TF's features 
_TF_SEQUENCE = ['1w', '1d', '4h', '1h', '15m']
_next_tf_build_thread = None  # Background thread handle for assembly-line overlap
_next_tf_build_ok = None      # Result of background feature build

# --- 15m-specific env vars to prevent OOM during cross gen ---
if TF == '15m':
    os.environ.setdefault('V2_RIGHT_CHUNK', '500')
    os.environ.setdefault('V2_BATCH_MAX', '500')

# Min base feature threshold  parquets with fewer cols need rebuild
# All TFs should have ~2,600-3,400 base features when built with V2 layers.
# A parquet with <2000 cols means V2 layers were NOT applied (old V1 build path).
MIN_BASE_FEATURES = 2000

START = time.time()
FAILURES = []
CURRENT_STEP = "startup"
LAST_PROGRESS_TS = START
WATCHDOG_WARNED_AT = 0.0
RUN_ID = os.environ.get('SAVAGE22_RUN_ID', f'{TF}-{os.getpid()}')
PHASE_SEQ = 0
HEARTBEAT_SEQ = 0
HEARTBEAT_FILE = run_path(f'cloud_run_{TF}_heartbeat.json')
WATCHDOG_STOP = threading.Event()
_WATCHDOG_INTERVAL = int(os.environ.get('SAVAGE22_HEARTBEAT_SEC', '60'))
_PROGRESS_WARN_SEC = int(os.environ.get('SAVAGE22_PROGRESS_WARN_SEC', '1800'))
_RESOURCE_SAMPLE_SEC = int(os.environ.get('SAVAGE22_RESOURCE_SAMPLE_SEC', '300'))
_LAST_RESOURCE_SNAPSHOT_TS = 0.0
_LAST_RESOURCE_SNAPSHOT = {}
FINAL_HEARTBEAT_REASON = None

def _gb(v):
    try:
        return round(float(v) / (1024 ** 3), 2)
    except Exception:
        return None

def _resource_snapshot():
    """Collect low-overhead heartbeat telemetry with safe fallbacks."""
    global _LAST_RESOURCE_SNAPSHOT_TS, _LAST_RESOURCE_SNAPSHOT
    now = time.time()
    if now - _LAST_RESOURCE_SNAPSHOT_TS < _RESOURCE_SAMPLE_SEC and _LAST_RESOURCE_SNAPSHOT:
        return _LAST_RESOURCE_SNAPSHOT

    snap = {'ts': now}
    try:
        if _HAS_PSUTIL:
            p = psutil.Process(os.getpid())
            with p.oneshot():
                snap['proc_rss_gb'] = _gb(p.memory_info().rss)
                snap['proc_cpu_pct'] = round(p.cpu_percent(interval=None), 1)
            vm = psutil.virtual_memory()
            snap['ram_used_gb'] = _gb(vm.used)
            snap['ram_total_gb'] = _gb(vm.total)
            snap['ram_percent'] = vm.percent
    except Exception:
        pass
    try:
        import shutil as _shutil
        snap['disk_free_gb'] = _gb(_shutil.disk_usage('.').free)
    except Exception:
        snap['disk_free_gb'] = None

    snap['vram_used_gb'] = snap.get('vram_used_gb', None)
    snap['vram_total_gb'] = snap.get('vram_total_gb', None)
    snap['vram_util_pct'] = snap.get('vram_util_pct', None)
    try:
        import shutil as _shutil2
        if _shutil2.which('nvidia-smi'):
            out = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            )
            line = out.strip().splitlines()[0]
            used_str, total_str, util_str = [x.strip() for x in line.split(',')]
            snap['vram_used_gb'] = _gb(int(used_str) * 1024 * 1024)
            snap['vram_total_gb'] = _gb(int(total_str) * 1024 * 1024)
            snap['vram_util_pct'] = float(util_str.replace('%', ''))
    except Exception:
        pass

    _LAST_RESOURCE_SNAPSHOT_TS = now
    _LAST_RESOURCE_SNAPSHOT = snap
    return snap


def _detect_large_machine_profile():
    gpu_count = 0
    try:
        import shutil as _shutil3
        if _shutil3.which('nvidia-smi'):
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            gpu_count = len([line for line in out.splitlines() if line.strip()])
    except Exception:
        gpu_count = 0
    cpu_count = os.cpu_count() or 0
    return gpu_count >= 8 and cpu_count >= 256, gpu_count, cpu_count

def _set_step(name):
    global CURRENT_STEP, PHASE_SEQ
    CURRENT_STEP = _canonical_phase(name)
    _phase_info = _phase_def(CURRENT_STEP)
    _contract_seq = _phase_info.get('phase_seq')
    if isinstance(_contract_seq, int):
        PHASE_SEQ = _contract_seq
    else:
        PHASE_SEQ += 1
    _mark_progress(f"step:{name}")

def _mark_progress(reason=None):
    global LAST_PROGRESS_TS
    LAST_PROGRESS_TS = time.time()
    _write_heartbeat(reason=reason)

def _write_heartbeat(reason=None):
    global HEARTBEAT_SEQ
    try:
        HEARTBEAT_SEQ += 1
        now = time.time()
        status = 'failed' if FAILURES else 'running'
        if reason == 'complete':
            status = 'complete'
        elif reason and ('validated' in reason or reason.endswith(':validated')):
            status = 'validated'
        payload = {
            'run_id': RUN_ID,
            'tf': TF,
            'pid': os.getpid(),
            'phase': CURRENT_STEP,
            'canonical_phase': CURRENT_STEP,
            'phase_seq': PHASE_SEQ,
            'contract_phase_seq': _phase_def(CURRENT_STEP).get('phase_seq', PHASE_SEQ),
            'status': status,
            'started_at': START,
            'updated_at': now,
            'last_progress_at': LAST_PROGRESS_TS,
            'heartbeat_seq': HEARTBEAT_SEQ,
            'elapsed_sec': round(time.time() - START, 1),
            'last_progress_age_sec': round(time.time() - LAST_PROGRESS_TS, 1),
            'failures': list(FAILURES),
            'allow_cpu': os.environ.get('ALLOW_CPU', ''),
            'resources': _resource_snapshot(),
            'artifact_root': ARTIFACT_ROOT,
            'run_root': RUN_ROOT,
            'code_root': CODE_ROOT,
            'shared_db_root': SHARED_DB_ROOT,
            'heartbeat_path': HEARTBEAT_FILE,
            'artifact_contract': _TF_CONTRACT.get('source'),
            'release_manifest': run_path('release_manifest.json'),
            'expected_artifacts': _phase_required_artifacts(CURRENT_STEP),
        }
        if reason:
            payload['reason'] = reason
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

def _watchdog_loop():
    global WATCHDOG_WARNED_AT
    while not WATCHDOG_STOP.wait(_WATCHDOG_INTERVAL):
        age = time.time() - LAST_PROGRESS_TS
        _write_heartbeat()
        if age >= _PROGRESS_WARN_SEC and time.time() - WATCHDOG_WARNED_AT >= _PROGRESS_WARN_SEC:
            health = _resource_snapshot()
            mem = health.get('ram_percent', 'n/a')
            vram = health.get('vram_util_pct', 'n/a')
            log(f"[WATCHDOG] No progress for {age/60:.1f} min during step '{CURRENT_STEP}' "
                f"(RAM {mem}%, VRAM {vram}%)")
            WATCHDOG_WARNED_AT = time.time()

def elapsed():
    return f"[{time.time()-START:.0f}s]"

def log(msg):
    print(f"{elapsed()} {msg}", flush=True)


_LARGE_MACHINE_PROFILE, _LM_GPU_COUNT, _LM_CPU_COUNT = _detect_large_machine_profile()
if _LARGE_MACHINE_PROFILE:
    os.environ.setdefault('CPCV_PARALLEL_GPUS', str(_LM_GPU_COUNT))
    os.environ.setdefault('OPTUNA_N_JOBS', str(min(_LM_GPU_COUNT, 8)))
    os.environ.setdefault('OPTUNA_FINAL_RETRAIN_MAX_PARALLEL_FOLDS', str(min(_LM_GPU_COUNT, 8)))
    os.environ.setdefault('V3_CPCV_WORKERS', str(min(_LM_GPU_COUNT * 2, 16)))
    os.environ.setdefault('SAVAGE22_AUTO_ASSEMBLY', '1')
    log(f"Large-machine profile enabled: {_LM_GPU_COUNT} GPUs, {_LM_CPU_COUNT} cores")

def run(cmd, name, critical=True):
    """Run command. If critical=True and it fails, abort entire pipeline."""
    _set_step(name)
    t0 = time.time()
    log(f"=== {name} ===")
    r = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    missing = []
    if ok:
        missing = _missing_phase_artifacts(CURRENT_STEP)
        if missing:
            ok = False
    _mark_progress(f"{name}:{'validated' if ok else 'fail'}")
    if ok:
        _write_phase_sentinel(CURRENT_STEP, status='validated')
    log(f"{name}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    if not ok:
        if missing:
            log(f"  Missing required artifacts for {CURRENT_STEP}: {', '.join(os.path.basename(p) for p in missing)}")
        FAILURES.append(name)
        _write_heartbeat(reason=f"{name}:critical-fail" if critical else f"{name}:fail")
        if critical:
            log(f"*** CRITICAL FAILURE: {name}  aborting ***")
            _print_summary()
            sys.exit(1)
    return ok

def run_tee(cmd, name, logfile, critical=True, verify_phase_artifacts=True,
            write_phase_sentinel=True, record_failure=True):
    """Run command with output tee'd to logfile  Python Popen drain, no shell pipe.
    Replaces bash 'tee' pipeline which breaks on long runs (SIGPIPE, buffer saturation)."""
    from pathlib import Path
    _set_step(name)
    t0 = time.time()
    log(f"=== {name} ===")
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, 'wb', buffering=0) as logf:
        proc = subprocess.Popen(
            ['bash', '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                _mark_progress(f"{name}:stream")
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
        finally:
            if proc.stdout:
                proc.stdout.close()
        rc = proc.wait()
    dt = time.time() - t0
    ok = rc == 0
    missing = []
    if ok and verify_phase_artifacts:
        missing = _missing_phase_artifacts(CURRENT_STEP)
        if missing:
            ok = False
    _mark_progress(f"{name}:{'validated' if ok else 'fail'}")
    if ok and write_phase_sentinel:
        _write_phase_sentinel(CURRENT_STEP, status='validated')
    log(f"{name}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    if not ok:
        if missing:
            log(f"  Missing required artifacts for {CURRENT_STEP}: {', '.join(os.path.basename(p) for p in missing)}")
        if record_failure:
            FAILURES.append(name)
            _write_heartbeat(reason=f"{name}:critical-fail" if critical else f"{name}:fail")
        if critical:
            log(f"*** CRITICAL FAILURE: {name}  aborting ***")
            _print_summary()
            sys.exit(1)
    return ok

def _print_summary():
    elapsed_total = time.time() - START
    print(f"\n{'='*60}", flush=True)
    if FAILURES:
        print(f"  PIPELINE FAILED: {TF} ({elapsed_total:.0f}s / {elapsed_total/60:.1f} min)", flush=True)
        print(f"  Failures: {', '.join(FAILURES)}", flush=True)
    else:
        print(f"  PIPELINE COMPLETE: {TF} ({elapsed_total:.0f}s / {elapsed_total/60:.1f} min)", flush=True)
    print(f"{'='*60}", flush=True)
    # Assembly-line: report background build status if active
    if _next_tf_build_thread is not None:
        if _next_tf_build_thread.is_alive():
            print(f"  [ASSEMBLY-LINE] Background feature build still running for next TF", flush=True)
        elif _next_tf_build_ok is not None:
            _status = 'OK' if _next_tf_build_ok else 'FAILED'
            print(f"  [ASSEMBLY-LINE] Next TF feature build: {_status}", flush=True)
    # List all artifacts
    artifacts = list(dict.fromkeys(artifact_path(a) for a in (
        *(_phase_required_artifacts('complete') or []),
        f'model_{TF}_cpcv_backup.json',
        f'meta_model_{TF}.pkl',
        f'lstm_{TF}.pt',
        f'feature_importance_stability_{TF}.json',
        f'shap_analysis_{TF}.json',
        'ml_multi_tf_configs.json',
        # Inference cross artifacts (for live trading)
        f'inference_{TF}_thresholds.json',
        f'inference_{TF}_cross_pairs.npz',
        f'inference_{TF}_ctx_names.json',
        f'inference_{TF}_base_cols.json',
        f'inference_{TF}_cross_names.json',
    )))
    print("  Artifacts:", flush=True)
    for a in artifacts:
        if os.path.exists(a):
            sz = os.path.getsize(a) / (1024*1024)
            print(f"    {sz:8.1f} MB  {os.path.basename(a)}", flush=True)
        else:
            print(f"    MISSING     {os.path.basename(a)}", flush=True)


# ============================================================
# HEADER
# ============================================================
try:
    from hardware_detect import get_available_ram_gb, get_cpu_count
    ram_gb = get_available_ram_gb()
    cpu_count = get_cpu_count()
except ImportError:
    try:
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except (ValueError, OSError):
        ram_gb = 0
    cpu_count = os.cpu_count() or 1
# Cloud prerequisite: verify we get the bulk of host RAM (not a tiny cgroup slice)
if os.path.exists('/proc/meminfo'):
    try:
        _host_ram_gb = 0
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    _host_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
        if _host_ram_gb > 0:
            _ram_pct = (ram_gb / _host_ram_gb) * 100 if _host_ram_gb > 0 else 100
            if ram_gb < _host_ram_gb * 0.5:
                print(f"  WARNING: cgroup reports {ram_gb:.0f}GB ({_ram_pct:.0f}%) of {_host_ram_gb:.0f}GB host RAM", flush=True)
                print(f"  Using host RAM value ({_host_ram_gb:.0f}GB)  cgroup detection may be inaccurate", flush=True)
                ram_gb = _host_ram_gb
            elif ram_gb < _host_ram_gb:
                print(f"  RAM: {ram_gb:.0f}GB cgroup / {_host_ram_gb:.0f}GB host ({_ram_pct:.0f}%)", flush=True)
                ram_gb = _host_ram_gb  # use host value for min-RAM checks
    except Exception:
        pass
print(f"{'='*60}", flush=True)
print(f"  CLOUD PIPELINE: {TF}", flush=True)
print(f"  Cores: {cpu_count} (cgroup-aware)", flush=True)
print(f"  RAM:   {ram_gb:.0f} GB (cgroup-aware)", flush=True)
print(f"  CWD:   {os.getcwd()}", flush=True)
if _jemalloc_found:
    print(f"  jemalloc: {_jemalloc_found} (LD_PRELOAD active)", flush=True)
else:
    print(f"  jemalloc: NOT FOUND  using system malloc (install libjemalloc-dev for less fragmentation)", flush=True)
print(f"{'='*60}", flush=True)

# --- RAM validation per TF ---
TF_MIN_RAM = {'1w': 64, '1d': 128, '4h': 256, '1h': 512, '15m': 768}  # Reduced: targeted crossing + int8 cut RAM 50-70%
if ram_gb > 0 and ram_gb < TF_MIN_RAM.get(TF, 64):
    print(f"ERROR: {TF} needs {TF_MIN_RAM[TF]}GB RAM, only {ram_gb:.0f}GB available", flush=True)
    sys.exit(1)
else:
    print(f"  RAM check: {ram_gb:.0f} GB >= {TF_MIN_RAM.get(TF, 64)} GB required for {TF}  OK", flush=True)

# ============================================================
# PRE-FLIGHT 0: Deployment verification (catches stale files, wrong binaries, missing DBs)
# ============================================================
run(f'{sys.executable} deploy_verify.py --tf {TF}', 'Deployment verification')

# ============================================================
# PRE-FLIGHT 1: Deterministic validation (catches config bugs before $$ is spent)
# ============================================================
run(f'{sys.executable} validate.py --tf {TF} --cloud', 'Pre-flight validation')

# ============================================================
# STEP 0: Kill stale python, install deps
# ============================================================
# Kill stale pipeline processes (NOT this script  exclude own PID)
_my_pid = os.getpid()
os.system(f'pgrep -f "python.*(ml_multi_tf|cross_generator|optuna|exhaustive|meta_label|lstm_seq|backtest|backtesting|build_.*features)" | grep -v {_my_pid} | xargs -r kill -9 2>/dev/null; true')
time.sleep(1)

# Install ALL dependencies  works on any base image (pytorch, ubuntu, etc.).
run('pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib '
    '"pandas<3.0" "numpy<2.3" pyarrow optuna hmmlearn numba tqdm pyyaml '
    'alembic cmaes colorlog sqlalchemy threadpoolctl sparse-dot-mkl 2>&1 | tail -5',
    'Install deps')

# OPT-14: Install numactl for NUMA-aware process binding on multi-socket cloud machines
run('apt-get install -y -qq numactl 2>/dev/null || true', 'Install numactl', critical=False)

# --- Stale artifact nuclear clean (delete old-version features/crosses) ---
log("=== Stale artifact cleanup ===")
_stale_patterns = ['features_*_all.json', 'v2_cross_names_*.json', 'v2_crosses_*.npz']  # NPZ+JSON must be deleted together
_stale_count = 0
_kept_count = 0
for _pat in _stale_patterns:
    for _stale in glob.glob(os.path.join(ARTIFACT_ROOT, _pat)):
        # Keep current TF's artifacts  they may represent hours of cross gen work
        _basename = os.path.basename(_stale)
        _keep_current = (
            _basename.startswith(f'features_{TF}_')
            or _basename.startswith(f'features_BTC_{TF}')
            or ((not _TF_CONTRACT.get('skip_crosses')) and (
                _basename.startswith(f'v2_crosses_BTC_{TF}')
                or _basename.startswith(f'v2_cross_names_BTC_{TF}')
            ))
        )
        if _keep_current:
            log(f"  Keeping current TF artifact: {_basename}")
            _kept_count += 1
            continue
        log(f"  Removing stale artifact: {_basename}")
        os.remove(_stale)
        _stale_count += 1
if _stale_count:
    log(f"  Removed {_stale_count} stale artifacts from previous runs")
if _kept_count:
    log(f"  Preserved {_kept_count} artifacts for current TF ({TF})")
if not _stale_count and not _kept_count:
    log(f"  No stale artifacts found")

# --- FIX 24: Lockfile  prevent duplicate pipeline runs for same TF ---
_lockfile = run_path(f'cloud_run_{TF}.lock')
if os.path.exists(_lockfile):
    # Check if the process that created it is still alive
    try:
        _lock_pid = int(open(_lockfile).read().strip())
        os.kill(_lock_pid, 0)  # Check if PID exists
        log(f"*** ABORT: Another pipeline running for {TF} (PID {_lock_pid}) ***")
        sys.exit(1)
    except (ValueError, ProcessLookupError, OSError):
        log(f"  Stale lockfile found  previous run crashed. Removing.")
        os.remove(_lockfile)
with open(_lockfile, 'w') as f:
    f.write(str(os.getpid()))

# --- Lockfile cleanup on exit (normal, crash, or signal) ---
import atexit, signal as _signal
def _cleanup_lock():
    try: os.remove(_lockfile)
    except: pass
def _stop_watchdog(reason=None):
    WATCHDOG_STOP.set()
    if reason == 'atexit' and FINAL_HEARTBEAT_REASON in ('complete', 'failed'):
        reason = FINAL_HEARTBEAT_REASON
    _write_heartbeat(reason=reason)
def _handle_exit_signal(signum, _frame):
    _stop_watchdog(reason=f"signal:{signum}")
    _cleanup_lock()
    sys.exit(0)
atexit.register(_cleanup_lock)
atexit.register(lambda: _stop_watchdog(reason='atexit'))
_watchdog_thread = threading.Thread(target=_watchdog_loop, name=f"cloud-watchdog-{TF}", daemon=True)
_watchdog_thread.start()
_write_heartbeat(reason='watchdog-started')
_signal.signal(_signal.SIGTERM, _handle_exit_signal)
_signal.signal(_signal.SIGINT, _handle_exit_signal)

# ============================================================
# STEP 1: Fix btc_prices.db symbol format
# ============================================================
log("=== Verify btc_prices.db ===")

_btc_db_path = db_path('btc_prices.db')
_multi_asset_db_path = db_path('multi_asset_prices.db')
if not os.path.exists(_btc_db_path) or os.path.getsize(_btc_db_path) == 0:
    # Try multi_asset_prices.db as fallback
    if os.path.exists(_multi_asset_db_path):
        log("btc_prices.db missing  copying from multi_asset_prices.db")
        import shutil
        shutil.copy2(_multi_asset_db_path, _btc_db_path)
    else:
        log("*** CRITICAL: No price database found! ***")
        sys.exit(1)

# Check symbol format and fix if needed
conn = sqlite3.connect(_btc_db_path)
r_usdt = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'").fetchone()[0]
r_bare = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC'").fetchone()[0]
log(f"  BTC/USDT rows: {r_usdt}, BTC rows: {r_bare}")

if r_usdt == 0 and r_bare > 0:
    log("  Symbol format is 'BTC'  adding '/USDT' suffix for compatibility...")
    conn.execute("UPDATE ohlcv SET symbol = symbol || '/USDT' WHERE symbol NOT LIKE '%/%'")
    conn.commit()
    # Verify fix
    r_check = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'").fetchone()[0]
    log(f"  After fix: {r_check} daily BTC/USDT rows")
    if r_check == 0:
        log("*** CRITICAL: btc_prices.db symbol fix failed ***")
        conn.close()
        sys.exit(1)
elif r_usdt > 0:
    log(f"  btc_prices.db OK: {r_usdt} daily BTC/USDT rows")
else:
    log("*** CRITICAL: No BTC data in btc_prices.db ***")
    conn.close()
    sys.exit(1)

# Log available timeframes for BTC
tfs = conn.execute(
    "SELECT timeframe, COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' GROUP BY timeframe ORDER BY timeframe"
).fetchall()
for tf_name, cnt in tfs:
    log(f"    {tf_name}: {cnt} rows")
conn.close()

# --- FIX 25: Disk space check before feature rebuild / cross gen ---
import shutil
_disk = shutil.disk_usage('.')
_free_gb = _disk.free / (1024**3)
if _free_gb < 20:
    log(f"*** ABORT: Only {_free_gb:.1f} GB free disk space (need 20+) ***")
    sys.exit(1)
log(f"Disk space OK: {_free_gb:.1f} GB free")

# --- 16-DB verification at startup ---
_REQUIRED_DBS = [
    'btc_prices.db', 'tweets.db', 'news_articles.db', 'sports_results.db',
    'space_weather.db', 'onchain_data.db', 'macro_data.db', 'astrology_full.db',
    'ephemeris_cache.db', 'fear_greed.db', 'funding_rates.db', 'google_trends.db',
    'open_interest.db', 'multi_asset_prices.db', 'llm_cache.db', 'v2_signals.db',
]
log("=== Database verification ===")
_db_missing = []
for _db in _REQUIRED_DBS:
    # Check both CWD and /workspace
    _found = os.path.exists(_db) or os.path.exists(os.path.join('/workspace', _db))
    _status = 'PASS' if _found else 'FAIL'
    if not _found:
        _db_missing.append(_db)
    log(f"  {_status}: {_db}")
if _db_missing:
    log(f"*** CRITICAL: {len(_db_missing)} required databases missing: {', '.join(_db_missing)} ***")
    log(f"  Upload ALL .db files before running pipeline. See CLOUD_TRAINING_PROTOCOL.md")
    sys.exit(1)
log(f"  All {len(_REQUIRED_DBS)} databases present  OK")

# ============================================================
# STEP 2: Rebuild features if parquet missing or incomplete
# ============================================================
parquet_path = artifact_path(f'features_BTC_{TF}.parquet')
need_rebuild = False

if not os.path.exists(parquet_path):
    # Also check non-BTC name (some build scripts save without symbol)
    alt_path = artifact_path(f'features_{TF}.parquet')
    if os.path.exists(alt_path):
        os.rename(alt_path, parquet_path)
        log(f"Renamed {alt_path}  {parquet_path}")
    else:
        log(f"Parquet {parquet_path} not found  need rebuild")
        need_rebuild = True

if not need_rebuild and os.path.exists(parquet_path):
    import pandas as pd
    pq = pd.read_parquet(parquet_path)
    n_cols = pq.shape[1]
    n_rows = pq.shape[0]
    pq_cols = set(pq.columns)
    log(f"Parquet check: {n_rows} rows x {n_cols} cols")
    del pq
    if n_cols < MIN_BASE_FEATURES:
        log(f"  Only {n_cols} cols (need {MIN_BASE_FEATURES}+)  need rebuild")
        need_rebuild = True
    else:
        # V3.3 FEATURE FINGERPRINT: detect stale parquets missing new features.
        # If feature_library.py was updated with new features but parquet was built
        # with old code, the parquet is stale and must be rebuilt.
        _V33_FINGERPRINT_COLS = [
            'vortex_family_group',     # compute_vortex_sacred_geometry_features
            'mars_speed',              # compute_planetary_expansion_features (get_planetary_speeds)
            'moon_distance_norm',      # compute_lunar_electromagnetic_features
            'loshu_row',               # compute_numerology_expansion_features
        ]
        _missing_v33 = [c for c in _V33_FINGERPRINT_COLS if c not in pq_cols]
        if _missing_v33:
            log(f"  STALE PARQUET: missing v3.3 features: {_missing_v33}")
            log(f"  Parquet was built with old feature_library.py  forcing rebuild")
            need_rebuild = True
        else:
            log(f"  Parquet OK: {n_cols} base features (v3.3 fingerprint verified)")

if need_rebuild and _phase_is_before_resume('step1_features'):
    _resume_validate_phase('step1_features', note=f'resume_boundary_for_{RESUME_FROM_PHASE}')
elif need_rebuild:
    # Prefer build_features_v2.py (includes V2 layers: 4-tier binarization, entropy,
    # hurst, fib levels, moon signs, aspects, extra lags, etc.)
    # The old build_{TF}_features.py scripts only call build_all_features() without
    # V2 layers, producing ~1,200 base features instead of ~3,000+.
    # Look for build scripts in both CWD and the script's own directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    def _find_script(name):
        if os.path.exists(name):
            return name
        alt = os.path.join(_script_dir, name)
        if os.path.exists(alt):
            return alt
        return None

    _v2 = _find_script('build_features_v2.py')
    if _v2:
        build_script = _v2
        build_cmd = f'python -X utf8 -u {build_script} --symbol BTC --tf {TF}'
    else:
        build_script = _find_script(f'build_{TF}_features.py')
        if not build_script:
            build_script = _find_script('build_features_complete.py')
        if not build_script:
            log(f"*** CRITICAL: No build script for {TF} ***")
            sys.exit(1)
        build_cmd = f'python -X utf8 -u {build_script}'

    log(f"Rebuilding {TF} features using {build_script}...")
    run_tee(build_cmd,
            f'Rebuild {TF} features', run_path(f'logs/rebuild_{TF}.log'))

    # The build script may save to _SCRIPT_DIR instead of CWD  check both locations
    if not os.path.exists(parquet_path):
        # Check all possible locations
        candidates = [
            artifact_path(f'features_{TF}.parquet'),
            os.path.join(_SCRIPT_DIR, f'features_BTC_{TF}.parquet'),
            os.path.join(_SCRIPT_DIR, f'features_{TF}.parquet'),
        ]
        found = None
        for alt in candidates:
            if os.path.exists(alt):
                found = alt
                break
        if found:
            if found != parquet_path:
                os.symlink(os.path.abspath(found), parquet_path)
                log(f"Symlinked {parquet_path}  {found}")
        else:
            log(f"*** CRITICAL: Feature rebuild produced no parquet ***")
            log(f"  Checked: {parquet_path}, {', '.join(candidates)}")
            sys.exit(1)

    # Verify rebuilt parquet
    import pandas as pd
    pq = pd.read_parquet(parquet_path)
    log(f"Rebuilt parquet: {pq.shape[0]} rows x {pq.shape[1]} cols")
    if pq.shape[1] < MIN_BASE_FEATURES:
        log(f"*** CRITICAL: Rebuilt parquet still only {pq.shape[1]} cols ***")
        sys.exit(1)
    del pq
    _set_step('step1_features')
    _write_phase_sentinel('step1_features', status='validated', note='rebuilt_features')
    _mark_progress('step1_features:validated')
elif _phase_is_before_resume('step1_features'):
    _resume_validate_phase('step1_features', note=f'resume_boundary_for_{RESUME_FROM_PHASE}')
elif os.path.exists(parquet_path):
    _set_step('step1_features')
    _write_phase_sentinel('step1_features', status='validated', note='reused_cached_features')
    _mark_progress('step1_features:validated')

# Create non-BTC compatibility symlink inside ARTIFACT_ROOT for scripts that look for features_{tf}.parquet
plain_pq = artifact_path(f'features_{TF}.parquet')
if os.path.exists(parquet_path) and not os.path.exists(plain_pq):
    os.symlink(parquet_path, plain_pq)
    log(f"Symlinked {plain_pq}  {parquet_path}")

# --- FIX 25: Disk space check before cross gen ---
_disk = shutil.disk_usage('.')
_free_gb = _disk.free / (1024**3)
if _free_gb < 20:
    log(f"*** ABORT: Only {_free_gb:.1f} GB free disk space (need 20+) ***")
    sys.exit(1)

# ============================================================
# STEP 3: Build crosses (skip if NPZ already exists)
# ============================================================
# Per-TF cross feature toggle: 1w has too few rows (1158) for 2.8M crosses to be meaningful.
# Base features alone (TA + esoteric + astro + gematria + numerology) give better signal on 1w.
# The matrix thesis scales with DATA  more rows = more crosses add value.
SKIP_CROSSES_TFS = {TF} if _TF_CONTRACT.get('skip_crosses') else set()
if TF in SKIP_CROSSES_TFS:
    log(f"  SKIP CROSSES for {TF}  base features only (too few rows for cross features to add signal)")
    log(f"  Matrix signal comes from base feature diversity on {TF}")

# --- Cross gen thread safety (SAV-53) ---
# OMP_NUM_THREADS=4 set at top of file (before any numpy/scipy import).
# Reinforced here for clarity. 4 threads prevents thread exhaustion on cloud machines
# while still allowing MKL SpGEMM parallelism. NUMBA prange uses its own thread count.
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ.setdefault('OMP_PROC_BIND', 'spread')
os.environ.setdefault('OMP_PLACES', 'cores')
os.environ.setdefault('OMP_SCHEDULE', 'static')
log(f"Cross gen phase: OMP_NUM_THREADS=4, NUMBA_NUM_THREADS=4, MKL_DYNAMIC=FALSE (thread-safe for all TFs)")

cross_phase = 'step2_crosses'
npz_path = artifact_path(f'v2_crosses_BTC_{TF}.npz')
cn_path = artifact_path(f'v2_cross_names_BTC_{TF}.json')

_npz_valid = False
_skip_crosses = TF in SKIP_CROSSES_TFS
if _phase_is_before_resume(cross_phase):
    _resume_validate_phase(cross_phase, note=f'resume_boundary_for_{RESUME_FROM_PHASE}')
    _npz_valid = True
elif _skip_crosses:
    _set_step(cross_phase)
    _npz_valid = True  # maintained 1w run uses base features only
    log(f"  Crosses DISABLED for {TF}  training on base features only")
    _write_phase_sentinel(cross_phase, status='validated', note='skipped_by_contract')
    _mark_progress(f"{cross_phase}:validated")
elif os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000:
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    # Validate NPZ col count  stale NPZs from v3.0 (min_nonzero=8) have far fewer crosses
    _MIN_CROSS_COLS = {'1w': 500_000, '1d': 1_000_000, '4h': 1_000_000, '1h': 1_000_000, '15m': 1_000_000}
    _min_cols = _MIN_CROSS_COLS.get(TF, 1_000_000)
    try:
        if not os.path.exists(cn_path):
            log(f"  Cross names JSON missing for cached NPZ  will rebuild")
            os.remove(npz_path)
            raise FileNotFoundError(cn_path)
        from scipy import sparse as _sp
        _npz_shape = _sp.load_npz(npz_path).shape
        if _npz_shape[1] >= _min_cols:
            _npz_valid = True
            log(f"Cross NPZ valid ({npz_size:.1f} MB, {_npz_shape[1]:,} cols >= {_min_cols:,} min)  SKIPPING cross gen")
            _set_step(cross_phase)
            _write_phase_sentinel(cross_phase, status='validated', note='reused_cached_cross_artifacts')
            _mark_progress(f"{cross_phase}:validated")
        else:
            log(f"Cross NPZ STALE ({_npz_shape[1]:,} cols < {_min_cols:,} min)  will rebuild")
            os.remove(npz_path)
            if os.path.exists(cn_path):
                os.remove(cn_path)
                log(f"  Removed stale cross names: {cn_path}")
    except Exception as _e:
        log(f"  NPZ validation failed ({_e})  will rebuild")
if _npz_valid:
    pass  # NPZ valid, skip cross gen
else:
    run_tee(f'python -X utf8 -u {_script("v2_cross_generator.py")} --tf {TF} --symbol BTC --save-sparse',
            f'Build {TF} crosses', run_path(f'logs/cross_{TF}.log'))
    # Check artifact-root output
    if not os.path.exists(npz_path):
        log(f"*** CRITICAL: {npz_path} not created by cross generator ***")
        sys.exit(1)
    if not os.path.exists(cn_path):
        log(f"*** CRITICAL: {cn_path} not created by cross generator ***")
        sys.exit(1)
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    log(f"Cross NPZ: {npz_size:.1f} MB")

# Verify inference artifacts were created by cross generator
_inf_artifacts = [artifact_path(f'inference_{TF}_thresholds.json'),
                  artifact_path(f'inference_{TF}_cross_pairs.npz'),
                  artifact_path(f'inference_{TF}_ctx_names.json'),
                  artifact_path(f'inference_{TF}_base_cols.json'),
                  artifact_path(f'inference_{TF}_cross_names.json')]
_inf_missing = [a for a in _inf_artifacts if not os.path.exists(a)]
if _inf_missing:
    log(f"  WARNING: Missing inference artifacts: {', '.join(os.path.basename(a) for a in _inf_missing)}")
    log(f"  Live trading will not have cross features for {TF}")
else:
    log(f"  Inference artifacts OK: all {len(_inf_artifacts)} files present")

# ============================================================
# STEP 5: Optuna hyperparameter search (BEFORE training)
# Saves optuna_configs_{tf}.json with best params  does NOT save a production model.
# Step 4 reads optuna_configs_{tf}.json and uses those params for the real CPCV training.
# ============================================================
# --- FIX #45: Reset thread pools between phases to prevent MKL/OpenMP/Numba conflicts ---
# Cross gen sets OMP_NUM_THREADS=cpu_count + MKL_DYNAMIC=FALSE. Training needs a clean slate.
os.environ.pop('OMP_NUM_THREADS', None)
os.environ.pop('NUMBA_NUM_THREADS', None)
os.environ.pop('MKL_DYNAMIC', None)
os.environ.pop('OMP_PROC_BIND', None)
os.environ.pop('OMP_PLACES', None)
os.environ.pop('OMP_SCHEDULE', None)
# OPT: Re-set OpenMP tuning for training phase (spread threads across cores for LightGBM)
os.environ.setdefault('OMP_PROC_BIND', 'spread')
os.environ.setdefault('OMP_PLACES', 'cores')
os.environ.setdefault('OMP_SCHEDULE', 'static')
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=cpu_count, user_api='blas')
    threadpool_limits(limits=cpu_count, user_api='openmp')
    log(f"Training phase: thread pools reset to {cpu_count} cores (blas+openmp), OMP_PROC_BIND=spread")
except ImportError:
    log(f"Training phase: threadpoolctl not available, env vars cleared")

# OPT-14: NUMA-aware process binding for multi-socket cloud machines
# Detect NUMA topology and bind training to node 0 for memory locality
_NUMA_PREFIX = ''
try:
    _numa_out = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True, timeout=5)
    if _numa_out.returncode == 0:
        _numa_nodes = [l for l in _numa_out.stdout.splitlines() if l.startswith('node') and 'cpus:' in l]
        _n_numa_nodes = len(_numa_nodes)
        if _n_numa_nodes > 1:
            # Test if numactl actually works (containers may lack SYS_NICE capability)
            _numa_test = subprocess.run(['numactl', '--interleave=all', 'true'], capture_output=True, timeout=5)
            if _numa_test.returncode == 0:
                log(f"NUMA: {_n_numa_nodes} nodes detected - using memory interleave across all nodes")
                _NUMA_PREFIX = 'numactl --interleave=all '
            else:
                log(f"NUMA: {_n_numa_nodes} nodes detected but numactl lacks permission (container) - skipping")
            for _nl in _numa_nodes:
                log(f"  {_nl.strip()}")
        else:
            log(f"NUMA: single node  no binding needed")
    else:
        log(f"NUMA: numactl not available or failed")
except (FileNotFoundError, subprocess.TimeoutExpired):
    log(f"NUMA: numactl not installed  skipping NUMA binding")

# ============================================================
# ASSEMBLY-LINE: Prefetch next TF's features in background thread
# Feature builds are CPU-bound, Optuna/training are GPU-bound  no contention.
# Only overlaps feature BUILD (not cross gen, which needs GPU).
# ============================================================
def _assembly_line_build_next_tf():
    """Build features for the next TF in a background thread.
    Sets global _next_tf_build_ok with the result."""
    global _next_tf_build_ok
    try:
        _idx = _TF_SEQUENCE.index(TF)
    except ValueError:
        _next_tf_build_ok = False
        return
    if _idx + 1 >= len(_TF_SEQUENCE):
        return  # Last TF, nothing to prefetch
    next_tf = _TF_SEQUENCE[_idx + 1]
    next_parquet = artifact_path(f'features_BTC_{next_tf}.parquet')

    # Check if already built
    if os.path.exists(next_parquet):
        import pandas as _pd_check
        try:
            _ncols = _pd_check.read_parquet(next_parquet, columns=None).shape[1]
            if _ncols >= MIN_BASE_FEATURES:
                log(f"  [ASSEMBLY-LINE] {next_tf} features already exist ({_ncols} cols)  skip prefetch")
                _next_tf_build_ok = True
                return
            else:
                log(f"  [ASSEMBLY-LINE] {next_tf} parquet stale ({_ncols} cols)  rebuilding")
        except Exception:
            log(f"  [ASSEMBLY-LINE] {next_tf} parquet unreadable  rebuilding")

    # Find the build script
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    def _find(name):
        if os.path.exists(name):
            return name
        alt = os.path.join(_script_dir, name)
        if os.path.exists(alt):
            return alt
        return None

    _v2 = _find('build_features_v2.py')
    if _v2:
        cmd = f'python -X utf8 -u {_v2} --symbol BTC --tf {next_tf}'
    else:
        _alt = _find(f'build_{next_tf}_features.py') or _find('build_features_complete.py')
        if not _alt:
            log(f"  [ASSEMBLY-LINE] No build script for {next_tf}  cannot prefetch")
            _next_tf_build_ok = False
            return
        cmd = f'python -X utf8 -u {_alt}'

    log(f"  [ASSEMBLY-LINE] Starting background feature build: {next_tf}")
    t0 = time.time()
    logfile = run_path(f'logs/assembly_build_{next_tf}.log')
    try:
        with open(logfile, 'w') as lf:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
            proc.wait()
        dt = time.time() - t0
        if proc.returncode == 0:
            log(f"  [ASSEMBLY-LINE] {next_tf} feature build OK ({dt:.0f}s)  see {logfile}")
            _next_tf_build_ok = True
        else:
            log(f"  [ASSEMBLY-LINE] {next_tf} feature build FAILED (exit {proc.returncode}, {dt:.0f}s)  see {logfile}")
            _next_tf_build_ok = False
    except Exception as e:
        log(f"  [ASSEMBLY-LINE] {next_tf} feature build ERROR: {e}")
        _next_tf_build_ok = False

if ASSEMBLY_LINE:
    try:
        _idx = _TF_SEQUENCE.index(TF)
        if _idx + 1 < len(_TF_SEQUENCE):
            _next_tf_name = _TF_SEQUENCE[_idx + 1]
            log(f"[ASSEMBLY-LINE] Will prefetch {_next_tf_name} features while Optuna/training runs")
            _next_tf_build_thread = threading.Thread(
                target=_assembly_line_build_next_tf,
                name=f'assembly-build-{_next_tf_name}',
                daemon=True,
            )
            _next_tf_build_thread.start()
        else:
            log(f"[ASSEMBLY-LINE] {TF} is the last TF in sequence  nothing to prefetch")
    except ValueError:
        log(f"[ASSEMBLY-LINE] {TF} not in sequence {_TF_SEQUENCE}  skipping prefetch")

optuna_config_path = artifact_path(f'optuna_configs_{TF}.json')
_optuna_cached_ok = False
if _phase_is_before_resume('step4_optuna'):
    _resume_validate_phase('step4_optuna', note=f'resume_boundary_for_{RESUME_FROM_PHASE}')
    _optuna_cached_ok = True
elif os.path.exists(optuna_config_path):
    try:
        with open(optuna_config_path, 'r', encoding='utf-8') as _f_optuna:
            _optuna_cached = json.load(_f_optuna)
        _optuna_cached_ok = isinstance(_optuna_cached, dict) and 'best_params' in _optuna_cached
    except Exception:
        _optuna_cached_ok = False
    if _optuna_cached_ok:
        _set_step('step4_optuna')
        log(f"Optuna config already exists ({optuna_config_path})  skipping search")
        _write_phase_sentinel('step4_optuna', status='validated', note='reused_cached_optuna_config')
        _mark_progress('step4_optuna:validated')
    else:
        log(f"Optuna config exists but is missing best_params  will rerun search")
if not _optuna_cached_ok:
    run_tee(f'env OPTUNA_SKIP_FINAL_RETRAIN=1 {_NUMA_PREFIX}python -X utf8 -u {_script("run_optuna_local.py")} --tf {TF}',
            f'Optuna search {TF}', run_path(f'logs/optuna_{TF}.log'), critical=False)
    if os.path.exists(optuna_config_path):
        import json as _json_step5
        with open(optuna_config_path) as _f5:
            _oc = _json_step5.load(_f5)
        log(f"Optuna search complete: accuracy={_oc.get('final_mean_accuracy', 'N/A')}, "
            f"sortino={_oc.get('final_mean_sortino', 'N/A')}")
        log(f"Best params: {_oc.get('best_params', {})}")
    else:
        log(f"WARNING: Optuna search did not produce {optuna_config_path}  Step 4 will use config.py defaults")

# ============================================================
# STEP 5: Retrain with winning params. Must produce fresh train artifacts.
# Resume target: --resume-from step5_retrain
# ============================================================

# Clean stale CPCV checkpoint  prevents resuming from a previous run's completed folds
# which would skip all training and produce results from old/different data
_cpcv_ckpt = artifact_path(f'cpcv_checkpoint_{TF}.pkl')
if os.path.exists(_cpcv_ckpt):
    log(f"  Preserving CPCV checkpoint candidate: {_cpcv_ckpt}")

train_log = run_path(f'logs/train_{TF}.log')
_train_phase = 'step5_retrain'
_train_required_paths = _phase_artifact_paths(_train_phase)
_train_pre_state = _artifact_state(_train_required_paths)
_train_backup_dir = run_path(f'resume/{TF}/step5_preexisting')
_train_backups = _backup_existing_artifacts(_train_required_paths, _train_backup_dir)
if _train_backups:
    log(f"Step 5 resume safety: backed up {len(_train_backups)} pre-existing train artifact(s) to {_train_backup_dir}")
_resume_cmd = f'python -X utf8 -u {_script("cloud_run_tf.py")} --tf {TF} --resume-from step5_retrain'
_train_env_map = _train_phase_env_map()
_train_degradation_policy = ''
_train_min_parallelism = 0
if _contract_phase_degradation_policy:
    try:
        _train_degradation_policy = _contract_phase_degradation_policy(TF, _train_phase, _PIPELINE_CONTRACT_SOURCE)
        _train_min_parallelism = int(_contract_phase_min_parallelism(TF, _train_phase, _PIPELINE_CONTRACT_SOURCE) or 0)
    except Exception:
        _train_degradation_policy = ''
        _train_min_parallelism = 0
_train_cmd_env = dict(_train_env_map)
_train_cmd_env.update({
    'V3_HOT_PATH_TRAINING': '1',
    'V3_RUN_FI_STABILITY': '1',
    'PYTHONFAULTHANDLER': '1',
    'V3_FAIL_ON_SEQUENTIAL': '1' if _train_degradation_policy == 'fail_fast' and _train_min_parallelism > 1 else '0',
    'V3_MIN_PARALLELISM': str(_train_min_parallelism),
    'V3_COPY_SHM_ARRAYS': '1' if _tf_execution_mode() == 'cpu_first' else '0',
})
_train_env_prefix = 'env ' + ' '.join(
    f'{key}={shlex.quote(str(value))}' for key, value in sorted(_train_cmd_env.items())
) + ' '
_clear_phase_failure(_train_phase)
_write_phase_failure(
    _train_phase,
    reason='pending',
    detail='step5_retrain starting',
    status='pending',
    extra={
        'resume_hint': _resume_cmd,
        'train_log': train_log,
        'preexisting_artifacts': [p for p in _train_required_paths if _train_pre_state.get(p)],
        'execution_mode': _tf_execution_mode(),
        'train_env': _train_env_map,
        'preserved_prereq_artifacts': {
            'step1_features': _phase_artifact_paths('step1_features'),
            'step2_crosses': _phase_artifact_paths('step2_crosses'),
            'step4_optuna': _phase_artifact_paths('step4_optuna'),
        },
        'cpcv_checkpoint': _cpcv_ckpt if os.path.exists(_cpcv_ckpt) else '',
    },
)
_train_started_ns = time.time_ns()
_train_ok = run_tee(
    f'{_train_env_prefix}{_NUMA_PREFIX}python -X utf8 -u {_script("ml_multi_tf.py")} --tf {TF}',
    f'Train {TF}',
    train_log,
    critical=False,
    verify_phase_artifacts=False,
    write_phase_sentinel=False,
    record_failure=False,
)
if not _train_ok:
    _train_runtime_failure = _step5_runtime_failure_context()
    _train_reason = _step5_failure_reason(_train_runtime_failure)
    _train_missing = _missing_phase_artifacts(_train_phase)
    _detail = 'Training command failed before fresh step5 artifacts were validated'
    _runtime_detail = _step5_failure_detail(_train_runtime_failure)
    if _runtime_detail:
        _detail = _runtime_detail
    if _train_missing:
        _detail += f"; missing: {', '.join(os.path.basename(p) for p in _train_missing)}"
    _write_phase_failure(
        _train_phase,
        reason=_train_reason,
        detail=_detail,
        extra={
            'resume_hint': _resume_cmd,
            'train_log': train_log,
            'missing_artifacts': _train_missing,
            'execution_mode': _tf_execution_mode(),
            'train_env': _train_env_map,
            'runtime_failure': _train_runtime_failure,
        },
    )
    FAILURES.append(f'{_train_phase}:{_train_reason}')
    _write_heartbeat(reason=f'{_train_phase}:{_train_reason}')
    log(f"*** CRITICAL FAILURE: {_train_phase} failed. Resume with: {_resume_cmd} ***")
    _print_summary()
    sys.exit(1)

# CRITICAL VERIFICATION: Check that cross features were loaded (SPARSE or DENSE both valid)
log("=== CROSS FEATURE VERIFICATION ===")
crosses_loaded = False
combined_line = ""
_accuracy_floor_hit = False
_model_saved_line = ""
if os.path.exists(train_log):
    with open(train_log, 'r', errors='replace') as f:
        for line in f:
            if 'Features:' in line and ('SPARSE' in line or 'DENSE' in line):
                crosses_loaded = True
                log(f"  VERIFIED: {line.strip()}")
            if 'Combined sparse' in line or 'Combined' in line:
                combined_line = line.strip()
            if 'ACCURACY BELOW FLOOR' in line:
                _accuracy_floor_hit = True
            if 'Model saved:' in line:
                _model_saved_line = line.strip()

if crosses_loaded:
    log("  PASS: Training loaded cross features")
    if combined_line:
        log(f"  {combined_line}")
else:
    log("*** CRITICAL: Training did NOT load cross features! ***")
    log("  Check cross_{TF}.log and train_{TF}.log")
    _write_phase_failure(
        _train_phase,
        reason='cross-features-not-loaded',
        detail='Training completed without logging sparse/dense feature load confirmation',
        extra={'resume_hint': _resume_cmd, 'train_log': train_log},
    )
    FAILURES.append(f'{_train_phase}:cross-features-not-loaded')
    _write_heartbeat(reason=f'{_train_phase}:cross-features-not-loaded')
    _print_summary()
    sys.exit(1)

# CRITICAL: Verify model was actually saved (accuracy floor can silently skip save)
_model_path = artifact_path(f'model_{TF}.json')
if not os.path.exists(_model_path):
    log(f"*** CRITICAL: model_{TF}.json NOT FOUND after training ***")
    log(f"  Training exited OK but model was not saved.")
    log(f"  Most likely cause: final accuracy < 0.40 (accuracy floor).")
    log(f"  Check {train_log} for 'ACCURACY BELOW FLOOR' message.")
    _write_phase_failure(
        _train_phase,
        reason='model-missing-after-train',
        detail='Training exited without producing model artifact',
        extra={
            'resume_hint': _resume_cmd,
            'train_log': train_log,
            'accuracy_floor_detected': _accuracy_floor_hit,
        },
    )
    FAILURES.append(f'{_train_phase}:model-missing')
    _print_summary()
    sys.exit(1)

_stale_train_artifacts = [
    path for path in _train_required_paths
    if not _artifact_changed_since(path, _train_pre_state, _train_started_ns)
]
if _stale_train_artifacts:
    log("*** CRITICAL: Step 5 did not refresh all train artifacts. ***")
    log(f"  Stale artifacts: {', '.join(os.path.basename(p) for p in _stale_train_artifacts)}")
    if _model_saved_line:
        log(f"  Last save evidence: {_model_saved_line}")
    _write_phase_failure(
        _train_phase,
        reason='stale-train-artifacts',
        detail='Retrain reused pre-existing baseline artifacts instead of producing fresh outputs',
        extra={
            'resume_hint': _resume_cmd,
            'train_log': train_log,
            'stale_artifacts': _stale_train_artifacts,
            'accuracy_floor_detected': _accuracy_floor_hit,
            'model_saved_line': _model_saved_line,
        },
    )
    FAILURES.append(f'{_train_phase}:stale-artifacts')
    _write_heartbeat(reason=f'{_train_phase}:stale-artifacts')
    _print_summary()
    sys.exit(1)

_clear_phase_failure(_train_phase)
_write_phase_sentinel(_train_phase, status='validated', note='fresh_retrain_artifacts')
_mark_progress(f'{_train_phase}:validated')

# PROTECT Step 4 model from any downstream overwrite
import shutil
_backup_path = artifact_path(f'model_{TF}_cpcv_backup.json')
if os.path.exists(_model_path):
    shutil.copy2(_model_path, _backup_path)
    log(f"  Model backed up: {_backup_path} ({os.path.getsize(_model_path)/1024:.0f} KB)")

# ============================================================
# VERSIONING SYSTEM: Deploy model to version registry
# ============================================================
log("=== DEPLOYING MODEL TO VERSION REGISTRY ===")

# Extract accuracy from train log
_model_accuracy = 0.0
if os.path.exists(train_log):
    with open(train_log, 'r', errors='replace') as f:
        for line in f:
            if 'Model saved:' in line and 'accuracy:' in line:
                # Parse: "Model saved: /path/model_1w.json (accuracy: 0.934)"
                try:
                    _acc_str = line.split('accuracy:')[1].strip().rstrip(')')
                    _model_accuracy = float(_acc_str)
                    log(f"  Extracted accuracy: {_model_accuracy:.3f}")
                    break
                except (IndexError, ValueError) as _e:
                    log(f"  WARNING: Failed to parse accuracy from log: {_e}")

if _model_accuracy == 0.0:
    log("  WARNING: Could not extract accuracy from train log, using 0.0")

# Check if features file exists
_feat_path = artifact_path(f'features_{TF}_all.json')
_feat_arg = f'--features {_feat_path}' if os.path.exists(_feat_path) else ''

# Deploy to version registry
import subprocess as _sp
_deploy_cmd = f'python {_script("deploy_model.py")} deploy --tf {TF} --model {_model_path} --accuracy {_model_accuracy} {_feat_arg}'
log(f"  Running: {_deploy_cmd}")
try:
    _deploy_result = _sp.run(_deploy_cmd, shell=True, capture_output=True, text=True, check=True)
    log(_deploy_result.stdout)
    log("   Model successfully deployed to version registry")
except _sp.CalledProcessError as _e:
    log(f"  WARNING: Model versioning failed (non-critical): {_e}")
    log(_e.stderr)
    # Don't fail the pipeline on versioning errors (it's a new feature)

# NOTE: Step 5 (Optuna search) now runs BEFORE Step 4. It saves only params (optuna_configs_{tf}.json),
# NOT a model. Step 4 reads those params and uses them for full CPCV training.

# ============================================================
# STEP 6: Exhaustive trade optimizer
# ============================================================
run_tee(f'python -X utf8 -u {_script("exhaustive_optimizer.py")} --tf {TF}',
        f'Optimizer {TF}', run_path(f'logs/optimizer_{TF}.log'), critical=False)

# ============================================================
# STEPS 7,8,9  Run in PARALLEL (all depend only on Step 4)
# Step 10 (Audit) runs AFTER  depends on Step 6 optimizer output
# ============================================================
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

def _run_step(name, cmd, logfile):
    """Wrapper for parallel step execution."""
    try:
        ok = run_tee(cmd, name, logfile, critical=False)
        return (name, ok)
    except Exception as e:
        log(f"  {name} failed: {e}")
        return (name, False)

_parallel_phase_specs = [
    ('step7_meta', f'Meta {TF}',  f'python -X utf8 -u {_script("meta_labeling.py")} --tf {TF} --db-dir {shlex.quote(ARTIFACT_ROOT)}',  run_path(f'logs/meta_{TF}.log')),
    ('step8_lstm', f'LSTM {TF}',  f'python -X utf8 -u {_script("lstm_sequence_model.py")} --tf {TF} --train',  run_path(f'logs/lstm_{TF}.log')),
    ('step9_pbo',  f'PBO {TF}',   f'python -X utf8 -u {_script("backtest_validation.py")} --tf {TF} --db-dir {shlex.quote(ARTIFACT_ROOT)}',  run_path(f'logs/pbo_{TF}.log')),
]
_parallel_steps = []
for _phase_name, _step_name, _cmd, _logfile in _parallel_phase_specs:
    if _phase_should_run(_phase_name):
        _parallel_steps.append((_step_name, _cmd, _logfile))
    else:
        _set_step(_phase_name)
        _skip_note = _phase_skip_reason(_phase_name) or 'skipped_by_contract'
        log(f"=== {_step_name} skipped by contract: {_skip_note} ===")
        _write_phase_sentinel(_phase_name, status='validated', note=_skip_note)
        _mark_progress(f"{_phase_name}:validated")

if _parallel_steps:
    log(f"=== Steps 7,8,9 launching in parallel ({len(_parallel_steps)} tasks) ===")
    with ThreadPoolExecutor(max_workers=len(_parallel_steps)) as _step_pool:
        _step_futures = {_step_pool.submit(_run_step, n, c, l): n for n, c, l in _parallel_steps}
        for _sf in _as_completed(_step_futures):
            _sname, _sok = _sf.result()
            log(f"  {_sname}: {'OK' if _sok else 'FAIL'}")
else:
    log("=== Steps 7,8,9 skipped by contract ===")

# ============================================================
# STEP 10: Audit (sequential  depends on Step 6 optimizer output)
# ============================================================
if _phase_should_run('step10_audit'):
    _audit_ok = run_tee(
        f'python -X utf8 -u {_script("backtesting_audit.py")} --tf {TF}',
        f'Audit {TF}', run_path(f'logs/audit_{TF}.log'), critical=False
    )
    log(f"  Audit {TF}: {'OK' if _audit_ok else 'FAIL'}")
else:
    _set_step('step10_audit')
    _audit_skip_note = _phase_skip_reason('step10_audit') or 'skipped_by_contract'
    log(f"=== Audit {TF} skipped by contract: {_audit_skip_note} ===")
    _write_phase_sentinel('step10_audit', status='validated', note=_audit_skip_note)
    _mark_progress('step10_audit:validated')

# ============================================================
# STEP 11: SHAP Cross Feature Validation (non-fatal)
# ============================================================
# === SHAP Cross Feature Validation ===
log(f"=== SHAP cross feature analysis ===")
try:
    import json
    import numpy as np
    import lightgbm as lgb

    # Load model
    model = lgb.Booster(model_file=artifact_path(f'model_{TF}.json'))

    # Get features with non-zero split count (reduces 3.34M -> likely ~50K-200K active)
    all_features = model.feature_name()
    split_scores = dict(zip(all_features, model.feature_importance(importance_type='split')))
    split_importance = split_scores
    active_features = [f for f, v in split_importance.items() if v > 0]
    log(f"  Active features (split > 0): {len(active_features)} / {len(split_importance)}")

    # Count cross features that are active
    cross_prefixes = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_', 'mx_', 'vx_', 'asp_', 'mn_', 'pn_')
    active_crosses = [f for f in active_features if f.startswith(cross_prefixes)]
    active_base = [f for f in active_features if not f.startswith(cross_prefixes)]
    log(f"  Active cross features: {len(active_crosses)}")
    log(f"  Active base features: {len(active_base)}")

    # Use split importance only  pred_contrib + .toarray() OOMs on 2.9M+ sparse crosses
    # Split importance is memory-safe and already shows which cross features contribute
    import pandas as pd

    gain_scores = dict(zip(all_features, model.feature_importance(importance_type='gain')))
    gain_importance = gain_scores

    # Build importance DataFrame
    imp_df = pd.DataFrame([
        {'feature': f, 'splits': split_importance.get(f, 0), 'gain': gain_importance.get(f, 0)}
        for f in all_features
    ])
    imp_df['is_cross'] = imp_df['feature'].str.startswith(cross_prefixes)

    # Family aggregation
    def get_family(name):
        for p in cross_prefixes:
            if name.startswith(p): return p.rstrip('_')
        parts = name.split('_')
        return parts[0] if parts else 'unknown'

    imp_df['family'] = imp_df['feature'].apply(get_family)
    family_imp = imp_df.groupby('family')['gain'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)

    # Report
    log(f"  Top 20 feature families by gain:")
    for i, (fam, row) in enumerate(family_imp.head(20).iterrows()):
        log(f"    {i+1:2d}. {fam:<20s} gain_sum={row['sum']:10.1f}  mean={row['mean']:.2f}  count={int(row['count'])}")

    # Cross vs base comparison
    cross_gain = imp_df[imp_df['is_cross']]['gain'].sum()
    base_gain = imp_df[~imp_df['is_cross']]['gain'].sum()
    cross_pct = 100 * cross_gain / (cross_gain + base_gain) if (cross_gain + base_gain) > 0 else 0
    log(f"  Cross feature gain: {cross_pct:.1f}% of total ({cross_gain:.1f} / {cross_gain + base_gain:.1f})")

    # Save report
    shap_report = {
        'active_features': len(active_features),
        'active_crosses': len(active_crosses),
        'active_base': len(active_base),
        'cross_shap_pct': round(cross_pct, 2),
        'method': 'split_importance (pred_contrib skipped  OOM on sparse crosses)',
        'top_20_families': family_imp.head(20).to_dict(),
        'top_50_features': imp_df.nlargest(50, 'gain')[['feature', 'gain', 'splits', 'is_cross']].to_dict('records'),
    }
    with open(artifact_path(f'shap_analysis_{TF}.json'), 'w') as f:
        json.dump(shap_report, f, indent=2, default=str)
    log(f"  Saved: {artifact_path(f'shap_analysis_{TF}.json')}")

except Exception as e:
    log(f"  SHAP analysis error (non-fatal): {e}")

# ============================================================
# ASSEMBLY-LINE: Wait for background feature build to finish
# ============================================================
_assembly_wait = os.environ.get('SAVAGE22_ASSEMBLY_WAIT', '0') == '1'
if _next_tf_build_thread is not None and _next_tf_build_thread.is_alive() and _assembly_wait:
    _next_tf_name = _TF_SEQUENCE[_TF_SEQUENCE.index(TF) + 1]
    log(f"[ASSEMBLY-LINE] Waiting for {_next_tf_name} feature build to finish...")
    _next_tf_build_thread.join(timeout=7200)  # 2 hour max wait
    if _next_tf_build_thread.is_alive():
        log(f"[ASSEMBLY-LINE] WARNING: {_next_tf_name} build still running after 2h timeout  proceeding")
    elif _next_tf_build_ok:
        log(f"[ASSEMBLY-LINE] {_next_tf_name} features ready for next pipeline run")
    else:
        log(f"[ASSEMBLY-LINE] {_next_tf_name} feature build failed  next run will rebuild")
elif _next_tf_build_thread is not None and _next_tf_build_thread.is_alive():
    _next_tf_name = _TF_SEQUENCE[_TF_SEQUENCE.index(TF) + 1]
    log(f"[ASSEMBLY-LINE] Leaving {_next_tf_name} feature build running in background (SAVAGE22_ASSEMBLY_WAIT=0)")

# ============================================================
# FINAL SUMMARY
# ============================================================
_set_step('final-summary')
if len(FAILURES) == 0:
    _complete_missing = [a for a in _phase_required_artifacts('complete') if not os.path.exists(artifact_path(a))]
    if _complete_missing:
        log(f"*** CRITICAL: missing complete artifacts: {', '.join(_complete_missing)} ***")
        FAILURES.append('complete-artifacts')
    if TF in SKIP_CROSSES_TFS and not os.path.exists(_phase_sentinel_path('step2_crosses')):
        log("*** CRITICAL: missing validated skip sentinel for step2_crosses ***")
        FAILURES.append('step2_crosses-sentinel')

_print_summary()

if len(FAILURES) == 0:
    FINAL_HEARTBEAT_REASON = 'complete'
    _write_phase_sentinel('complete', status='complete', note='run_complete')
    with open(run_path(f'DONE_{TF}'), 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"Total time: {time.time()-START:.0f}s\n")
        f.write(f"Failures: None\n")
    log(f"Wrote {run_path(f'DONE_{TF}')} marker file")
    _write_heartbeat(reason='complete')
else:
    FINAL_HEARTBEAT_REASON = 'failed'
    log(f"*** NOT writing DONE_{TF}  {len(FAILURES)} failures: {', '.join(FAILURES)} ***")
    _write_heartbeat(reason='failed')

# --- FIX 24: Remove lockfile on clean exit ---
WATCHDOG_STOP.set()
if os.path.exists(_lockfile):
    os.remove(_lockfile)

