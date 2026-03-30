# Merge Notes: Duplicate `_llvm_ctpop_i64` Resolution

## Problem
Both `numba_cross_kernels.py` (branch `ceo/backend-dev-22d6ed1e`) and `bitpack_utils.py` (branch `ceo/backend-dev-702e95d2`) define `_llvm_ctpop_i64` with different signatures.

## Comparison

### numba_cross_kernels.py (line 24)
```python
def _llvm_ctpop_i64(typingctx, x):
    sig = types.int64(types.int64)  # int64 -> int64
    def codegen(context, builder, signature, args):
        fn_type = ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        fn = builder.module.declare_intrinsic('llvm.ctpop', [ir.IntType(64)], fn_type)
        return builder.call(fn, args)
    return sig, codegen
```

### bitpack_utils.py (line 26)
```python
def _llvm_ctpop_i64(typingctx, val):
    sig = types.int64(types.uint64)  # uint64 -> int64
    def codegen(context, builder, signature, args):
        [val] = args
        fn_type = ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        fn = builder.module.declare_intrinsic('llvm.ctpop.i64', fnty=fn_type)
        return builder.call(fn, [val])
    return sig, codegen
```

## Key Differences

| Aspect | numba_cross_kernels | bitpack_utils |
|--------|-------------------|---------------|
| Input type | `int64` | `uint64` |
| Intrinsic name | `llvm.ctpop` (generic) | `llvm.ctpop.i64` (explicit) |
| Arg destructuring | Uses raw `args` tuple | Unpacks `[val] = args` |

## Recommendation: Keep `bitpack_utils.py` version

1. **`uint64` input is correct** — bitpacked data is unsigned by nature. Passing `int64` works but is semantically wrong for popcount on bitmasks.
2. **Explicit intrinsic name** (`llvm.ctpop.i64`) is more portable across LLVM versions than relying on type-based overload resolution.
3. **Arg unpacking** (`[val] = args`) is cleaner and matches Numba intrinsic conventions.

## Merge Action
- Keep `bitpack_utils.py:_llvm_ctpop_i64` as the canonical implementation
- Update `numba_cross_kernels.py` to import from `bitpack_utils` instead of defining its own copy
- Verify callers in `numba_cross_kernels.py` pass `uint64` (or cast if needed)
