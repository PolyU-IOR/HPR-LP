# GPU Presolve / Postsolve

This folder implements the GPU-first presolve/postsolve pipeline for GPUPresolver.

## Public API

- `presolve_gpu(lp::LP_info_gpu; presolve_params::PresolveParams=PresolveParams(), verbose=false)`
- `presolve_gpu(qp::QP_info_gpu; presolve_params::PresolveParams=PresolveParams(), verbose=false)`
- `postsolve_gpu(x_red, y_red, z_red, rec::PresolveRecord_gpu)`

## Three-Layer Architecture

- Layer 1: scheduler (`gpu_presolve.jl`, `gpu_presolve_qp.jl`, `gpu_postsolve.jl`)
- Layer 2: rules (`rules/`, `rules/lp_rules/`, `rules/qp_rules/`)
- Layer 3: GPU primitives (`gpu_presolve_kernels.jl`)

Rule files are grouped by applicability:

- `rules/`: common rules shared by LP and QP where the reduction does not need
  objective-specific logic.
- `rules/lp_rules/`: LP column rules whose correctness depends on a linear objective.
- `rules/qp_rules/`: QP-aware column rules that inspect or update `Q`.

## Presolve Workflow

The LP GPU presolver supports two schedulers:

- `fixed`: each iteration runs a row phase followed by a column phase in the
  configured rule order.
- `tiered`: an adaptive scheduler with an optional bootstrap phase, light
  passes, medium passes, heavy passes, and cleanup passes.

Each phase follows the same internal shape:

1. compute stats
2. build a plan
3. apply the plan and update the postsolve record

The loop stops when no change is produced, when `max_iters` is reached, or when
`max_time` is exceeded. A final cleanup pass can apply `redundant_bounds` after
the iterative loop.

## Current Rule Scope

- Common rules:
  - `empty_rows`
  - `singleton_rows`
  - `activity_checks`
  - `primal_propagation`
  - `parallel_rows`
  - `close_bounds`
  - `redundant_bounds`
- LP column rules:
  - `empty_cols`
  - `singleton_cols`
  - `doubleton_eq`
  - `dual_fix`
  - `parallel_cols`
  - `fme_projection`
  - `structural_l1_substitution`
- QP-aware column rules:
  - `empty_cols`
  - `singleton_cols_eq`
  - `singleton_cols_dual_infer`
  - `parallel_cols`
  - `dual_fix`
  - `doubleton_eq`
  - `linear_eq_agg`

The default package configuration uses the LP `tiered` scheduler and enables
`structural_l1_substitution` with `structural_l1.gpu_only = true`. FME projection
is available as an experimental LP rule but remains disabled by default.

The QP public entrypoint is a thin wrapper over the QP-native GPU scheduler.
It no longer replays QP reductions in the API layer.

Per-rule enabling and ordering can be customized through `PresolveParams`.

## Postsolve Scope

- Restores primal variables on GPU from the recorded postsolve tape.
- Restores row and bound multipliers with a hybrid strategy:
  - direct GPU replay for the rule families already recorded exactly in the tape;
  - rule-specific local recovery for reductions that still depend on original-model context.

## Diagnostics

- `rule_verification.jl` contains isolated-rule validation and diagnostic helpers.
