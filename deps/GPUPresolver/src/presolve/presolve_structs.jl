"""
Shared GPU presolve/postsolve data structures.
"""

using CUDA
using CUDA: CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR

const GPU_PRESOLVER_TRUE_ENV_VALUES = ("1", "true", "TRUE", "yes", "YES")

@inline function _gpu_presolver_env_value(name::String, default::String)
    return get(ENV, name, default)
end

@inline function _gpu_presolver_env_enabled(name::String; default::String="0")
    return _gpu_presolver_env_value(name, default) in GPU_PRESOLVER_TRUE_ENV_VALUES
end

Base.@kwdef mutable struct PresolveParams
    max_iters::Int = 10
    max_time::Float64 = Inf
    verbose::Bool = false
    debug_checks::Bool = false
    trace_enabled::Bool = false
    trace_path::Union{Nothing,String} = nothing
    record_postsolve_tape::Bool = true
    record_postsolve_tape_cpu::Bool = false

    feasibility_tol::Float64 = 1e-9
    bound_tol::Float64 = 1e-9
    zero_tol::Float64 = 1e-12
    primal_propagation_min_tighten_abs::Float64 = 1.0e-2
    doubleton_eq_single_batch_per_iter::Bool = false
    doubleton_eq_max_fill_in_proxy::Int = 10
    doubleton_eq_scan::Bool = false
    doubleton_eq_min_selected_per_batch::Int = 256
    doubleton_eq_min_selected_ratio::Float64 = 0.005
    doubleton_eq_max_batch_rounds::Int = 0
    doubleton_eq_max_time::Float64 = 0.0
    qp_doubleton_max_q_fill_abs::Int = 64
    qp_doubleton_max_q_fill_ratio::Float64 = 2.0
    qp_singleton_max_support::Int = -1
    qp_singleton_max_q_fill_abs::Int = -1
    qp_singleton_max_q_fill_ratio::Float64 = Inf
    qp_singleton_cols_eq_require_qdiag_zero::Bool = false
    qp_doubleton_eq_require_qdiag_zero::Bool = false
    qp_linear_eq_agg_max_support::Int = 8
    fme_zero_objective_only::Bool = true
    fme_pair_limit::Int = 8
    fme_nnz_ratio_limit::Float64 = 1.0
    fme_nnz_abs_slack::Int = 0
    fme_max_elims_per_call::Int = 0
    fme_allow_main_flow_without_tape::Bool = false
    fme_include_variable_bounds::Bool = false
    fme_use_simple_screen::Bool = false
    fme_simple_side_limit::Int = 2
    # When true, `apply_rule_fme_projection!` prints per-call `fme_simple:` diagnostics.
    # Keep false for normal runs (global `verbose` alone does not enable these lines).
    verbose_fme::Bool = false
    structural_l1_pattern::Symbol = :auto
    structural_l1_allow_main_flow_without_tape::Bool = true
    structural_l1_gpu_only::Bool = false
    structural_l1_residual_bound_as_free_min::Float64 = Inf
    host_iteration_callback::Any = nothing

    # Canonical rule switches.
    enable_close_bounds::Bool = true
    enable_empty_rows::Bool = true
    enable_singleton_rows::Bool = true
    enable_activity_checks::Bool = true
    enable_primal_propagation::Bool = true
    enable_parallel_rows::Bool = true
    enable_empty_cols::Bool = true
    enable_singleton_cols_eq::Bool = true
    enable_singleton_cols_dual_infer::Bool = true
    enable_doubleton_eq::Bool = true
    enable_linear_eq_agg::Bool = false
    enable_dual_fix::Bool = true
    enable_parallel_cols::Bool = true
    enable_fme_projection::Bool = false
    enable_structural_l1_substitution::Bool = false
    enable_redundant_bounds::Bool = true

    # Canonical rule scheduling.
    row_rule_order::Vector{Symbol} = [:empty_rows, :singleton_rows, :activity_checks, :primal_propagation, :parallel_rows]
    col_rule_order::Vector{Symbol} = [:close_bounds, :structural_l1_substitution, :empty_cols, :singleton_cols_dual_infer, :singleton_cols_eq, :doubleton_eq, :linear_eq_agg, :dual_fix, :parallel_cols, :fme_projection]

    # GPU scheduler selector:
    #   :fixed   -> standard fixed-order loop
    #   :tiered  -> adaptive multi-phase scheduler with bootstrap
    gpu_presolve_scheduler::Symbol = :fixed
    enable_tiered_bootstrap::Bool = true
    tiered_cleanup_max_rounds::Int = 2
    tiered_light_continue_ratio::Float64 = 0.995
    tiered_cycle_stop_ratio::Float64 = 0.999
    tiered_max_light_streak::Int = 3
    tiered_global_period::Int = 2

    postsolve_tol::Float64 = 1.0e-7
    # Optional fallback for PARALLEL_ROW dual split when exact-local interval is empty.
    parallel_row_doc_recovery::Bool = true
end

mutable struct PresolveStats_gpu
    # Row-side stats.
    row_nnz::CuVector{Int32}
    empty_row_mask::CuVector{UInt8}
    singleton_row_mask::CuVector{UInt8}
    singleton_row_col::CuVector{Int32}
    singleton_row_val::CuVector{Float64}

    # Col-side stats.
    col_nnz::CuVector{Int32}
    empty_col_mask::CuVector{UInt8}
    singleton_col_mask::CuVector{UInt8}
    singleton_col_row::CuVector{Int32}
    singleton_col_val::CuVector{Float64}
    row_nnz_valid::Bool
    col_nnz_valid::Bool
    structural_screen_counts::CuVector{Int32}
    structural_screen_flag::CuVector{UInt8}
    structural_screen_pattern_code::Int32
    structural_screen_valid::Bool
end

function PresolveStats_gpu(m::Integer, n::Integer)
    mi = Int(m)
    ni = Int(n)
    return PresolveStats_gpu(
        CUDA.zeros(Int32, mi),
        CUDA.zeros(UInt8, mi),
        CUDA.zeros(UInt8, mi),
        CUDA.fill(Int32(-1), mi),
        CUDA.zeros(Float64, mi),
        CUDA.zeros(Int32, ni),
        CUDA.zeros(UInt8, ni),
        CUDA.zeros(UInt8, ni),
        CUDA.fill(Int32(-1), ni),
        CUDA.zeros(Float64, ni),
        false,
        false,
        CUDA.zeros(Int32, 3),
        CUDA.zeros(UInt8, 1),
        Int32(-1),
        false,
    )
end

mutable struct PresolvePlan_gpu
    keep_row_mask::CuVector{UInt8}
    keep_col_mask::CuVector{UInt8}

    # Optional rewritten matrix in current LP indexing.
    new_A::Union{Nothing,CuSparseMatrixCSR{Float64,Int32}}
    new_AT_leading_slack::Union{Nothing,Int32}
    new_AT_slack_after::Union{Nothing,CuVector{Int32}}

    # Proposed objective coefficient updates in current phase.
    new_c::CuVector{Float64}

    # Proposed variable-bound updates in current phase.
    new_l::CuVector{Float64}
    new_u::CuVector{Float64}

    # Proposed row-side updates in current phase.
    new_AL::CuVector{Float64}
    new_AU::CuVector{Float64}

    # Objective shift staged by rules (applied in apply stage).
    obj_constant_delta::Float64

    # Local (current-LP) fixed-column decisions from empty-column rule.
    fixed_idx::CuVector{Int32}
    fixed_val::CuVector{Float64}

    # Local singleton-column structural pairs.
    singleton_col_row_idx::CuVector{Int32}
    singleton_col_col_idx::CuVector{Int32}

    # Local parallel-column merge metadata.
    merged_col_from::CuVector{Int32}
    merged_col_to::CuVector{Int32}
    merged_col_ratio::CuVector{Float64}
    merged_col_from_l::CuVector{Float64}
    merged_col_from_u::CuVector{Float64}
    merged_col_to_l::CuVector{Float64}
    merged_col_to_u::CuVector{Float64}

    # Phase-local typed replay metadata in current LP indexing.
    tape::PostsolveTape
    tape_gpu::PostsolveTape_gpu
    structural_primal_recovery::Union{Nothing,Any}

    has_row_action::Bool
    has_col_action::Bool
    has_change::Bool
    has_infeasible::Bool
    has_unbounded::Bool
    status_message::String
end

function PresolvePlan_gpu(
    m::Integer,
    n::Integer,
    c::CuVector{Float64},
    AL::CuVector{Float64},
    AU::CuVector{Float64},
    l::CuVector{Float64},
    u::CuVector{Float64},
)
    return PresolvePlan_gpu(
        CUDA.fill(UInt8(1), Int(m)),
        CUDA.fill(UInt8(1), Int(n)),
        nothing,
        nothing,
        nothing,
        copy(c),
        copy(l),
        copy(u),
        copy(AL),
        copy(AU),
        0.0,
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        PostsolveTape(),
        PostsolveTape_gpu(),
        nothing,
        false,
        false,
        false,
        false,
        false,
        "",
    )
end

struct StructuralL1SplitRecovery
    t_col::Int32
    e_col::Int32
    rho::Float64
end

struct StructuralOuterPairRecovery
    bound_col::Int32
    free_col::Int32
end

struct StructuralLinkedSlackRecovery
    slack_col::Int32
    t_col::Int32
    factor::Float64
end

struct StructuralMaxSlackRecovery
    slack_col::Int32
    t_cols::Vector{Int32}
    factors::Vector{Float64}
end

struct StructuralL1PrimalRecoveryStep
    pattern::Symbol
    splits::Vector{StructuralL1SplitRecovery}
    outer_pairs::Vector{StructuralOuterPairRecovery}
    linked_slacks::Vector{StructuralLinkedSlackRecovery}
    max_slacks::Vector{StructuralMaxSlackRecovery}
end

mutable struct PresolveRecord_gpu
    m0::Int32
    n0::Int32
    m1::Int32
    n1::Int32

    # Cumulative mappings (always relative to original LP).
    row_org2red::CuVector{Int32}
    row_red2org::CuVector{Int32}
    col_org2red::CuVector{Int32}
    col_red2org::CuVector{Int32}

    # Cumulative recovery data.
    fixed_idx::CuVector{Int32}
    fixed_val::CuVector{Float64}
    removed_row_idx::CuVector{Int32}
    removed_col_idx::CuVector{Int32}

    # Cumulative singleton-column structural logs.
    singleton_col_row_idx::CuVector{Int32}
    singleton_col_col_idx::CuVector{Int32}

    # Cumulative parallel-column merge logs.
    merged_col_from::CuVector{Int32}
    merged_col_to::CuVector{Int32}
    merged_col_ratio::CuVector{Float64}
    merged_col_from_l::CuVector{Float64}
    merged_col_from_u::CuVector{Float64}
    merged_col_to_l::CuVector{Float64}
    merged_col_to_u::CuVector{Float64}

    obj_constant_old::Float64
    obj_constant_new::Float64

    rule_counters::Dict{Symbol,Int}
    tape::PostsolveTape
    tape_gpu::Union{Nothing,PostsolveTape_gpu}
    tape_gpu_parts::Vector{PostsolveTape_gpu}
    structural_primal_recoveries::Vector{StructuralL1PrimalRecoveryStep}
end

function PresolveRecord_gpu()
    return PresolveRecord_gpu(
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        0.0,
        0.0,
        Dict{Symbol,Int}(),
        PostsolveTape(),
        nothing,
        PostsolveTape_gpu[],
        StructuralL1PrimalRecoveryStep[],
    )
end

@inline function _identity_map_gpu(len::Integer)
    if len <= 0
        return CuVector{Int32}(undef, 0)
    end
    return CuVector(Int32.(1:Int(len)))
end

"""
Build a no-op cumulative record where reduced and original models are identical.
"""
function presolve_identity_record(m0::Integer, n0::Integer, obj_constant::Float64)
    m0i = Int32(m0)
    n0i = Int32(n0)
    row_red2org = _identity_map_gpu(m0)
    row_org2red = _identity_map_gpu(m0)
    col_red2org = _identity_map_gpu(n0)
    col_org2red = _identity_map_gpu(n0)

    return PresolveRecord_gpu(
        m0i,
        n0i,
        m0i,
        n0i,
        row_org2red,
        row_red2org,
        col_org2red,
        col_red2org,
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        obj_constant,
        obj_constant,
        Dict{Symbol,Int}(),
        PostsolveTape(),
        nothing,
        PostsolveTape_gpu[],
        StructuralL1PrimalRecoveryStep[],
    )
end
