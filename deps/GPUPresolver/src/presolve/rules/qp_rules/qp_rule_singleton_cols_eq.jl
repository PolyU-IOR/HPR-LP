"""
QP layer-2 rule: equality-row singleton-column elimination.

This is the exact QP substitution rule from the note:
- choose one equality-row singleton column per pass
- substitute x_j = beta + gamma' * x_R into the objective
- delete column j and delete/retain the singleton row accordingly

The implementation is intentionally conservative on batching. It processes at
most one candidate per call so that the algebra stays exact even when multiple
singleton columns would interact through Q.
"""

@inline function _mark_infeasible_qp_singleton_cols_eq!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

function _kernel_qp_accumulate_sparse_updates!(
    target,
    idx,
    vals,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds CUDA.@atomic target[idx[t]] += vals[t]
    end
    return
end

function _qp_csr_row_payload_host(
    csr::CuSparseMatrixCSR{Float64,Int32},
    row::Int32,
)
    start_ptr = Int(_copy_scalar_to_host(csr.rowPtr, Int(row)))
    stop_ptr = Int(_copy_scalar_to_host(csr.rowPtr, Int(row) + 1)) - 1
    if stop_ptr < start_ptr
        return Int32[], Float64[]
    end

    len = stop_ptr - start_ptr + 1
    cols = Vector{Int32}(undef, len)
    vals = Vector{Float64}(undef, len)
    copyto!(cols, 1, csr.colVal, start_ptr, len)
    copyto!(vals, 1, csr.nzVal, start_ptr, len)
    return cols, vals
end

function _qp_singleton_eq_support_payload(
    qp::QP_info_gpu,
    plan::QPresolvePlan_gpu,
    elim_col::Int32,
    pair_row::Int32,
)
    eliminated_cols = CuVector(Int32[elim_col])
    pair_rows = CuVector(Int32[pair_row])
    support_counts = CUDA.zeros(Int32, 1)
    @cuda threads=1 blocks=1 _kernel_singleton_col_support_counts!(
        support_counts,
        eliminated_cols,
        pair_rows,
        plan.keep_col_mask,
        qp.A.rowPtr,
        qp.A.colVal,
        Int32(1),
    )

    total_support = Int(_copy_scalar_to_host(support_counts, 1))
    support_starts = CuVector(Int32[1])
    flat_support_cols = CuVector{Int32}(undef, total_support)
    flat_support_coeffs = CuVector{Float64}(undef, total_support)
    if total_support > 0
        @cuda threads=1 blocks=1 _kernel_singleton_col_support_payload!(
            flat_support_cols,
            flat_support_coeffs,
            support_starts,
            eliminated_cols,
            pair_rows,
            plan.keep_col_mask,
            qp.A.rowPtr,
            qp.A.colVal,
            qp.A.nzVal,
            Int32(1),
        )
    end

    return support_counts, support_starts, flat_support_cols, flat_support_coeffs
end

function _qp_singleton_eq_delta_host(
    support_cols_h::Vector{Int32},
    support_coeffs_h::Vector{Float64},
    q_col_rows_h::Vector{Int32},
    q_col_vals_h::Vector{Float64},
    q_row_cols_h::Vector{Int32},
    q_row_vals_h::Vector{Float64},
    elim_col::Int32,
    beta::Float64,
    gamma_h::Vector{Float64},
    cj::Float64,
    qjj::Float64,
    zero_tol::Float64,
)
    c_delta = Dict{Int32,Float64}()
    for t in eachindex(q_col_rows_h)
        row = q_col_rows_h[t]
        val = q_col_vals_h[t]
        row == elim_col && continue
        abs(val) > zero_tol || continue
        c_delta[row] = get(c_delta, row, 0.0) + beta * val
    end

    affine_scale = cj + beta * qjj
    for t in eachindex(support_cols_h)
        col = support_cols_h[t]
        delta = affine_scale * gamma_h[t]
        abs(delta) > zero_tol || continue
        c_delta[col] = get(c_delta, col, 0.0) + delta
    end

    q_delta = Dict{Tuple{Int32,Int32},Float64}()
    for p in eachindex(q_col_rows_h)
        row = q_col_rows_h[p]
        val = q_col_vals_h[p]
        row == elim_col && continue
        abs(val) > zero_tol || continue
        for t in eachindex(support_cols_h)
            col = support_cols_h[t]
            delta = val * gamma_h[t]
            abs(delta) > zero_tol || continue
            key = (row, col)
            q_delta[key] = get(q_delta, key, 0.0) + delta
        end
    end

    for p in eachindex(q_row_cols_h)
        col = q_row_cols_h[p]
        val = q_row_vals_h[p]
        col == elim_col && continue
        abs(val) > zero_tol || continue
        for t in eachindex(support_cols_h)
            row = support_cols_h[t]
            delta = gamma_h[t] * val
            abs(delta) > zero_tol || continue
            key = (row, col)
            q_delta[key] = get(q_delta, key, 0.0) + delta
        end
    end

    for p in eachindex(support_cols_h)
        row = support_cols_h[p]
        gamma_p = gamma_h[p]
        for t in eachindex(support_cols_h)
            col = support_cols_h[t]
            delta = qjj * gamma_p * gamma_h[t]
            abs(delta) > zero_tol || continue
            key = (row, col)
            q_delta[key] = get(q_delta, key, 0.0) + delta
        end
    end

    c_idx_h = Int32[]
    c_val_h = Float64[]
    for (idx, val) in c_delta
        abs(val) > zero_tol || continue
        push!(c_idx_h, idx)
        push!(c_val_h, val)
    end

    q_rows_h = Int32[]
    q_cols_h = Int32[]
    q_vals_h = Float64[]
    for ((row, col), val) in q_delta
        abs(val) > zero_tol || continue
        push!(q_rows_h, row)
        push!(q_cols_h, col)
        push!(q_vals_h, val)
    end

    return c_idx_h, c_val_h, q_rows_h, q_cols_h, q_vals_h
end

function _qp_singleton_eq_fill_guard_host(
    qp::QP_info_gpu,
    support_cols_h::Vector{Int32},
    gamma_h::Vector{Float64},
    qjj::Float64,
    zero_tol::Float64,
    max_support::Int,
    max_fill_abs::Int,
    max_fill_ratio::Float64,
)
    abs(qjj) <= zero_tol && return true

    support_count = length(support_cols_h)
    if max_support >= 0 && support_count > max_support
        return false
    end

    support_pos = Dict{Int32,Int}()
    for (idx, col) in pairs(support_cols_h)
        support_pos[col] = idx
    end

    existing_block = falses(support_count, support_count)
    block_nnz = 0
    for (row_idx, row_col) in pairs(support_cols_h)
        row_cols_h, row_vals_h = _qp_csr_row_payload_host(qp.Q, row_col)
        for t in eachindex(row_cols_h)
            abs(row_vals_h[t]) > zero_tol || continue
            col_idx = get(support_pos, row_cols_h[t], 0)
            col_idx == 0 && continue
            if !existing_block[row_idx, col_idx]
                existing_block[row_idx, col_idx] = true
                block_nnz += 1
            end
        end
    end

    fill_count = 0
    for row_idx in eachindex(support_cols_h)
        gamma_row = gamma_h[row_idx]
        for col_idx in eachindex(support_cols_h)
            abs(qjj * gamma_row * gamma_h[col_idx]) > zero_tol || continue
            existing_block[row_idx, col_idx] && continue
            fill_count += 1
            if max_fill_abs >= 0 && fill_count > max_fill_abs
                return false
            end
        end
    end

    if isfinite(max_fill_ratio) && max_fill_ratio >= 0.0
        return fill_count <= max_fill_ratio * max(block_nnz, 1)
    end
    return true
end

function _append_qp_singleton_eq_postsolve_record!(
    plan::QPresolvePlan_gpu,
    pparams::PresolveParams,
    elim_col::Int32,
    pair_row::Int32,
    support_counts::CuVector{Int32},
    support_starts::CuVector{Int32},
    flat_support_cols::CuVector{Int32},
    flat_support_coeffs::CuVector{Float64},
    pivot_coeff::Float64,
    rhs_old::Float64,
    old_l::Float64,
    old_u::Float64,
    elim_obj::Float64,
    row_deleted::Bool,
)
    pparams.record_postsolve_tape || return nothing

    append_sub_col_records_from_payload_gpu!(
        plan.tape_gpu,
        CuVector(Int32[elim_col]),
        CuVector(Int32[pair_row]),
        support_counts,
        support_starts,
        flat_support_cols,
        flat_support_coeffs,
        CuVector(Float64[pivot_coeff]),
        CuVector(Float64[rhs_old]),
        CuVector(Float64[old_l]),
        CuVector(Float64[old_u]),
        CuVector(Float64[elim_obj]),
        CuVector(UInt8[row_deleted ? 1 : 0]);
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )

    if pparams.record_postsolve_tape_cpu
        append_sub_col_records_from_payload!(
            plan.tape,
            CuVector(Int32[elim_col]),
            CuVector(Int32[pair_row]),
            support_counts,
            support_starts,
            flat_support_cols,
            flat_support_coeffs,
            CuVector(Float64[pivot_coeff]),
            CuVector(Float64[rhs_old]),
            CuVector(Float64[old_l]),
            CuVector(Float64[old_u]),
            CuVector(Float64[elim_obj]),
            CuVector(UInt8[row_deleted ? 1 : 0]);
            dual_mode=POSTSOLVE_DUAL_MINIMAL,
        )
    end
    return nothing
end

function _kernel_qp_mark_separable_singleton_eq_candidates!(
    candidate_mask,
    row_owner,
    q_offdiag_nnz,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        @inbounds owner = row_owner[row]
        if owner != typemax(Int32) && q_offdiag_nnz[owner] == Int32(0)
            @inbounds candidate_mask[row] = UInt8(1)
        end
    end
    return
end

function apply_rule_qp_singleton_cols_eq!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    m = length(plan.keep_row_mask)
    n = length(plan.keep_col_mask)
    if m == 0 || n == 0
        return nothing
    end

    row_owner = CUDA.fill(typemax(Int32), m)
    status_flag = CUDA.zeros(Int32, 1)
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_col_row_owner!(
        status_flag,
        row_owner,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
        qp.A.rowPtr,
        qp.A.colVal,
        qp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        _SINGLETON_COL_RULE_EQ,
        Int32(n),
    )

    candidate_mask = CUDA.zeros(UInt8, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_mark_separable_singleton_eq_candidates!(
        candidate_mask,
        row_owner,
        stats.q_offdiag_nnz,
        Int32(m),
    )
    _, candidate_rows, candidate_count = build_maps_from_mask(candidate_mask)
    Int(candidate_count) == 0 && return nothing

    pair_row = Int32(_copy_scalar_to_host(candidate_rows, 1))
    elim_col = Int32(_copy_scalar_to_host(row_owner, Int(pair_row)))
    aij = Float64(_copy_scalar_to_host(stats.singleton_col_val, Int(elim_col)))
    b = Float64(_copy_scalar_to_host(plan.new_AU, Int(pair_row)))
    old_l = Float64(_copy_scalar_to_host(plan.new_l, Int(elim_col)))
    old_u = Float64(_copy_scalar_to_host(plan.new_u, Int(elim_col)))
    cj = Float64(_copy_scalar_to_host(plan.new_c, Int(elim_col)))
    qjj = Float64(_copy_scalar_to_host(plan.new_q_diag, Int(elim_col)))
    if pparams.qp_singleton_cols_eq_require_qdiag_zero && abs(qjj) > pparams.zero_tol
        return nothing
    end

    support_counts, support_starts, flat_support_cols, flat_support_coeffs =
        _qp_singleton_eq_support_payload(qp, plan, elim_col, pair_row)
    support_cols_h = _copy_vector_to_host(flat_support_cols)
    support_coeffs_h = _copy_vector_to_host(flat_support_coeffs)
    support_count = length(support_cols_h)
    support_count == 0 && return nothing

    l_h = Vector{Float64}(undef, support_count)
    u_h = Vector{Float64}(undef, support_count)
    for t in eachindex(support_cols_h)
        col = Int(support_cols_h[t])
        l_h[t] = Float64(_copy_scalar_to_host(plan.new_l, col))
        u_h[t] = Float64(_copy_scalar_to_host(plan.new_u, col))
    end

    rest_min = 0.0
    rest_max = 0.0
    for t in eachindex(support_coeffs_h)
        a = support_coeffs_h[t]
        if a >= 0.0
            rest_min += a * l_h[t]
            rest_max += a * u_h[t]
        else
            rest_min += a * u_h[t]
            rest_max += a * l_h[t]
        end
    end

    implied_lb = if aij > 0.0
        (b - rest_max) / aij
    else
        (b - rest_min) / aij
    end
    implied_ub = if aij > 0.0
        (b - rest_min) / aij
    else
        (b - rest_max) / aij
    end

    if implied_lb > old_u + pparams.bound_tol || implied_ub < old_l - pparams.bound_tol
        _mark_infeasible_qp_singleton_cols_eq!(
            plan,
            "QP singleton-cols-eq infeasibility: implied row interval is disjoint from the current bounds.",
        )
        return nothing
    end

    free_from_below = !isfinite(old_l) || implied_lb >= old_l - pparams.bound_tol
    free_from_above = !isfinite(old_u) || implied_ub <= old_u + pparams.bound_tol
    (free_from_below || free_from_above) || return nothing

    beta = b / aij
    gamma_h = -support_coeffs_h ./ aij
    _qp_singleton_eq_fill_guard_host(
        qp,
        support_cols_h,
        gamma_h,
        qjj,
        pparams.zero_tol,
        pparams.qp_singleton_max_support,
        pparams.qp_singleton_max_q_fill_abs,
        pparams.qp_singleton_max_q_fill_ratio,
    ) || return nothing

    q_col_rows_h, q_col_vals_h = _qp_csr_row_payload_host(qp.QT, elim_col)
    q_row_cols_h, q_row_vals_h = _qp_csr_row_payload_host(qp.Q, elim_col)
    c_idx_h, c_val_h, q_rows_h, q_cols_h, q_vals_h = _qp_singleton_eq_delta_host(
        support_cols_h,
        support_coeffs_h,
        q_col_rows_h,
        q_col_vals_h,
        q_row_cols_h,
        q_row_vals_h,
        elim_col,
        beta,
        gamma_h,
        cj,
        qjj,
        pparams.zero_tol,
    )

    if !isempty(c_idx_h)
        c_idx = CuVector(c_idx_h)
        c_vals = CuVector(c_val_h)
        blocks_c = cld(length(c_idx_h), GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_c _kernel_qp_accumulate_sparse_updates!(
            plan.new_c,
            c_idx,
            c_vals,
            Int32(length(c_idx_h)),
        )
    end

    if !isempty(q_rows_h)
        delta_Q = _qp_build_delta_csr_direct(
            CuVector(q_rows_h),
            CuVector(q_cols_h),
            CuVector(q_vals_h),
            size(qp.Q),
        )
        plan.new_Q = qp.Q + delta_Q
    else
        plan.new_Q = qp.Q
    end

    row_deleted = free_from_below && free_from_above
    if row_deleted
        CUDA.@allowscalar plan.keep_row_mask[pair_row] = UInt8(0)
    elseif free_from_above
        rhs_shift = b - aij * old_l
        if aij > 0.0
            CUDA.@allowscalar plan.new_AL[pair_row] = -Inf
            CUDA.@allowscalar plan.new_AU[pair_row] = rhs_shift
        else
            CUDA.@allowscalar plan.new_AL[pair_row] = rhs_shift
            CUDA.@allowscalar plan.new_AU[pair_row] = Inf
        end
    else
        rhs_shift = b - aij * old_u
        if aij > 0.0
            CUDA.@allowscalar plan.new_AL[pair_row] = rhs_shift
            CUDA.@allowscalar plan.new_AU[pair_row] = Inf
        else
            CUDA.@allowscalar plan.new_AL[pair_row] = -Inf
            CUDA.@allowscalar plan.new_AU[pair_row] = rhs_shift
        end
    end

    CUDA.@allowscalar plan.keep_col_mask[elim_col] = UInt8(0)
    plan.obj_constant_delta += cj * beta + 0.5 * qjj * beta * beta
    plan.singleton_col_row_idx = concat_cuvector(plan.singleton_col_row_idx, CuVector(Int32[pair_row]))
    plan.singleton_col_col_idx = concat_cuvector(plan.singleton_col_col_idx, CuVector(Int32[elim_col]))

    _append_qp_singleton_eq_postsolve_record!(
        plan,
        pparams,
        elim_col,
        pair_row,
        support_counts,
        support_starts,
        flat_support_cols,
        flat_support_coeffs,
        aij,
        b,
        old_l,
        old_u,
        cj,
        row_deleted,
    )

    plan.has_change = true
    return nothing
end
