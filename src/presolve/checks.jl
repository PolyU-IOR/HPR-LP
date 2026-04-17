"""
Optional debug checks for presolve/postsolve bring-up.
"""

using CUDA
using CUDA: CuVector

"""
Validate map consistency:
- `org2red[i] == -1` for removed entries.
- `red2org[r]` points back to valid original indices.
"""
function debug_assert_maps!(
    name::String,
    org2red::CuVector{Int32},
    red2org::CuVector{Int32},
    org_dim::Int,
    red_dim::Int,
)
    @assert length(org2red) == org_dim "[$name map] org2red length mismatch"
    @assert length(red2org) == red_dim "[$name map] red2org length mismatch"

    org2red_h = Array(org2red)
    red2org_h = Array(red2org)

    @assert all((red2org_h .>= 1) .& (red2org_h .<= org_dim)) "[$name map] red2org out of range"
    @assert length(unique(red2org_h)) == length(red2org_h) "[$name map] red2org is not injective"

    for i in eachindex(org2red_h)
        ridx = org2red_h[i]
        if ridx == -1
            continue
        end
        @assert 1 <= ridx <= red_dim "[$name map] org2red out of range"
        @assert red2org_h[ridx] == i "[$name map] org2red/red2org mismatch"
    end
    return nothing
end

"""
Basic postsolve dimension checks.
"""
function postsolve_basic_checks(x_red, y_red, z_red, rec::PresolveRecord_gpu)
    @assert length(x_red) == Int(rec.n1) "postsolve: x_red length != rec.n1"
    @assert length(y_red) == Int(rec.m1) "postsolve: y_red length != rec.m1"
    @assert length(z_red) == Int(rec.n1) "postsolve: z_red length != rec.n1"

    @assert length(rec.row_red2org) == Int(rec.m1) "postsolve: row_red2org length mismatch"
    @assert length(rec.col_red2org) == Int(rec.n1) "postsolve: col_red2org length mismatch"
    @assert length(rec.row_org2red) == Int(rec.m0) "postsolve: row_org2red length mismatch"
    @assert length(rec.col_org2red) == Int(rec.n0) "postsolve: col_org2red length mismatch"

    @assert length(rec.fixed_idx) == length(rec.fixed_val) "postsolve: fixed_idx/fixed_val length mismatch"
    return nothing
end

"""
Lightweight phase check used during presolve debugging.
"""
function presolve_phase_basic_checks!(
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    plan::PresolvePlan_gpu,
)
    m, n = size(lp.A)
    @assert length(stats.row_nnz) == m
    @assert length(stats.empty_row_mask) == m
    @assert length(stats.singleton_row_mask) == m
    @assert length(stats.col_nnz) == n
    @assert length(stats.empty_col_mask) == n
    @assert length(stats.singleton_col_mask) == n

    @assert length(plan.keep_row_mask) == m
    @assert length(plan.keep_col_mask) == n
    @assert length(plan.new_l) == n
    @assert length(plan.new_u) == n
    return nothing
end

function _stable_presolve_objective_sum(vals::Vector{Float64})
    isempty(vals) && return 0.0

    scale = maximum(abs.(vals))
    scale == 0.0 && return 0.0

    total = 0.0
    @inbounds for v in vals
        total += v / scale
    end
    delta = scale * total
    isfinite(delta) && return delta

    total_big = big(0.0)
    @inbounds for v in vals
        total_big += BigFloat(v)
    end
    return Float64(total_big)
end

function _stable_presolve_objective_sum(vals::CuVector{Float64})
    length(vals) == 0 && return 0.0

    scale = maximum(abs.(vals))
    scale == 0.0 && return 0.0

    delta = scale * sum(vals ./ scale)
    isfinite(delta) && return Float64(delta)

    return _stable_presolve_objective_sum(Array(vals))
end

function _stable_presolve_masked_product_sum(
    lhs::CuVector{Float64},
    rhs::CuVector{Float64},
    mask::CuVector{UInt8},
)
    _, selected_idx, selected_count = build_maps_from_mask(mask)
    Int(selected_count) == 0 && return 0.0

    lhs_h = Array(gather_by_red2org(lhs, selected_idx))
    rhs_h = Array(gather_by_red2org(rhs, selected_idx))

    total_big = big(0.0)
    @inbounds for k in eachindex(lhs_h)
        total_big += BigFloat(lhs_h[k]) * BigFloat(rhs_h[k])
    end
    return Float64(total_big)
end
