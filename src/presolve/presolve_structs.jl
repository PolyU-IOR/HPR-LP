"""
Shared GPU presolve/postsolve data structures.
"""

using CUDA
using CUDA: CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR

Base.@kwdef mutable struct PresolveParams
    max_iters::Int = 10
    verbose::Bool = false
    debug_checks::Bool = false
    record_postsolve_tape::Bool = true
    record_postsolve_tape_cpu::Bool = false

    feasibility_tol::Float64 = 1e-9
    bound_tol::Float64 = 1e-9
    zero_tol::Float64 = 1e-12
    doubleton_eq_max_shift::Int = 10
    doubleton_eq_single_batch_per_iter::Bool = false

    # Canonical rule switches.
    enable_close_bounds::Bool = true
    enable_empty_rows::Bool = true
    enable_singleton_rows::Bool = true
    enable_activity_checks::Bool = true
    enable_primal_propagation::Bool = true
    enable_parallel_rows::Bool = true
    enable_empty_cols::Bool = true
    enable_singleton_cols::Bool = true
    enable_doubleton_eq::Bool = false
    enable_dual_fix::Bool = true
    enable_parallel_cols::Bool = true
    enable_redundant_bounds::Bool = true

    # Canonical rule scheduling.
    row_rule_order::Vector{Symbol} = [:empty_rows, :singleton_rows, :activity_checks, :primal_propagation, :parallel_rows]
    col_rule_order::Vector{Symbol} = [:close_bounds, :empty_cols, :singleton_cols, :doubleton_eq, :dual_fix, :parallel_cols]

    # Legacy aliases kept for backward compatibility.
    enable_remove_empty_rows::Bool = true
    enable_remove_empty_cols::Bool = true

    # Postsolve: optionally apply guarded original-model dual refinement after replay
    apply_postsolve_original_dual_refinement::Bool = false
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

    # Local singleton-column structural pairs (scaffold, no elimination yet).
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
        false,
        false,
        false,
        false,
        false,
        "",
    )
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

    # Cumulative singleton-column structural logs (scaffold only).
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
    )
end
