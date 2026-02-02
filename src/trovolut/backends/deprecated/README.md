# Deprecated Backend Files

This folder contains files that were moved during the backend reorganization to create a cleaner structure with 4 main backends: `cpu`, `gpu`, `qpu`, and `gpu_mixed_precision`.

## Structure

### `emulator/` (DEPRECATED - merged into gpu_mixed_precision)
Files from the old `emulator` backend:
- `dynamics.py` - EmulatorDynamics class (merged into gpu_mixed_precision)
- `evaluator.py` - Evaluator code (review if needed)
- `metrics.py` - Metrics code (review if needed)
- `__init__.py` - Module init

**Reason for deprecation:** The `emulator` backend was redundant with `gpu_emulator`. Its core functionality (EmulatorDynamics with mixed precision) has been consolidated into the renamed `gpu_mixed_precision` backend.

### `gpu_emulator/` (DEPRECATED files, backend renamed to gpu_mixed_precision)
Files that were redundant or not core to mixed precision:
- `dynamics.py` - Basic Suzuki step functions (suzuki_step, product_formula, qdrift)
  - **Status:** REDUNDANT - These same functions exist in `cpu/dynamics.py`
  - **Action:** Already available in CPU backend, no need to duplicate
  
- `evaluator.py` - EvaluatorBuilder class
  - **Status:** REVIEW NEEDED - May be used by filemaker/workflow
  - **Action:** Check if needed for constraint-based evaluation, otherwise deprecate

- `hamsim.py` - Hamiltonian simulation utilities
  - **Status:** REVIEW NEEDED - Check if it has unique mixed precision simulation code
  - **Action:** If it has unique quantization utilities, keep in gpu_mixed_precision

## What Was Kept

### In `gpu_mixed_precision/` (renamed from gpu_emulator)
- `mixed_fps/` - Core mixed precision code
  - `mp_trotter.py` - Mixed precision Trotter implementation
  - `simulate_quantization.py` - Quantization simulation utilities
- `mp_evals.py` - Mixed precision evaluations
- `constraints.py` - GPU-specific constraints
- Consolidated EmulatorDynamics as MixedPrecisionDynamics

## Review Checklist

Before deleting these files, verify:

1. ✓ Suzuki functions in `gpu_emulator/dynamics.py` are available in CPU backend
2. ⚠ Check if `gpu_emulator/evaluator.py` (EvaluatorBuilder) is used elsewhere
3. ⚠ Check if `gpu_emulator/hamsim.py` has unique code not in mixed_fps
4. ✓ EmulatorDynamics functionality is preserved in gpu_mixed_precision
5. ⚠ Check if any tests depend on the old `emulator` backend name

## Safe to Delete?

After verification:
- ✓ `emulator/` - YES, once EmulatorDynamics is verified in new location
- ✓ `gpu_emulator/dynamics.py` - YES, suzuki functions are in CPU  
- ⚠ `gpu_emulator/evaluator.py` - Check filemaker dependencies first
- ⚠ `gpu_emulator/hamsim.py` - Review for unique code first

## Migration Date

Deprecated: 2026-02-02
Review by: [TBD]
Delete after: [TBD - after verification]
