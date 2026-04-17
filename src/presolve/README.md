# GPU Presolve / Postsolve

This folder implements the GPU-first presolve/postsolve pipeline for HPRLP.

## Public API

- `presolve_gpu(lp::LP_info_gpu, params::HPRLP_parameters; presolve_params::PresolveParams=PresolveParams())`
- `postsolve_gpu(x_red, y_red, z_red, rec::PresolveRecord_gpu)`

## Three-Layer Architecture

- Layer 1: scheduler (`gpu_presolve.jl`, `gpu_postsolve.jl`)
- Layer 2: rules (`rules/`)
- Layer 3: GPU primitives (`gpu_presolve_kernels.jl`)

## Presolve Workflow

Each presolve iteration executes in fixed order:

1. Row phase: `stats -> plan -> apply`
2. Col phase: `stats -> plan -> apply`

The loop stops when no change is produced in one full iteration, or when `max_iters` is reached.
A final cleanup pass can apply `redundant_bounds` after the iterative loop.

## Current Rule Scope

- Row phase:
  - `empty_rows`
  - `singleton_rows`
  - `activity_checks`
  - `primal_propagation`
  - `parallel_rows`
- Col phase:
  - `close_bounds`
  - `empty_cols`
  - `singleton_cols`
  - `doubleton_eq`
  - `dual_fix`
  - `parallel_cols`
- Cleanup pass:
  - `redundant_bounds`

Per-rule enabling and ordering can be customized through `PresolveParams`.

## Postsolve Scope

- Restores primal variables on GPU from the recorded postsolve tape.
- Restores row and bound multipliers with a hybrid strategy:
  - direct GPU replay for the rule families already recorded exactly in the tape;
  - rule-specific local recovery for reductions that still depend on original-model context.
- Optional original-dual refinement is available through
  `apply_postsolve_original_dual_refinement`.

## Diagnostics

- `rule_verification.jl` contains isolated-rule validation and diagnostic helpers.
