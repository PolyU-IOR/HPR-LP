function _kernel_mark_qp_linear_eq_agg_candidates!(
    candidate_mask,
    pivot_col_out,
    keep_row,
    keep_col,
    row_nnz,
    col_nnz,
    q_diag,
    q_offdiag_nnz,
    c,
    AL,
    AU,
    row_ptr,
    col_val,
    zero_tol,
    bound_tol,
    support_cap,
    m,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        @inbounds keep_row[row] != UInt8(0) || return
        nnz_row = @inbounds row_nnz[row]
        (nnz_row >= Int32(3) && nnz_row <= support_cap) || return

        row_l = @inbounds AL[row]
        row_u = @inbounds AU[row]
        (isfinite(row_l) && isfinite(row_u) && abs(row_u - row_l) <= bound_tol) || return

        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
        pivot_col = Int32(0)
        best_col_nnz = typemax(Int32)

        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                @inbounds keep_col[col] != UInt8(0) || continue
                @inbounds q_offdiag_nnz[col] == Int32(0) || continue
                abs(@inbounds(q_diag[col])) <= zero_tol || continue
                abs(@inbounds(c[col])) <= zero_tol || continue
                cn = @inbounds col_nnz[col]
                if cn > Int32(1) && (cn < best_col_nnz || (cn == best_col_nnz && col < pivot_col))
                    best_col_nnz = cn
                    pivot_col = col
                end
            end
        end

        pivot_col == Int32(0) && return
        @inbounds candidate_mask[row] = UInt8(1)
        @inbounds pivot_col_out[row] = pivot_col
    end
    return
end

function _qp_linear_eq_agg_delta_host(
    support_cols_h::Vector{Int32},
    incident_rows_h::Vector{Int32},
    incident_vals_h::Vector{Float64},
    pair_row::Int32,
    beta::Float64,
    gamma_h::Vector{Float64},
    zero_tol::Float64,
)
    delta_rows = Int32[]
    delta_cols = Int32[]
    delta_vals = Float64[]
    shift_rows = Int32[]
    shift_vals = Float64[]

    for t in eachindex(incident_rows_h)
        row = incident_rows_h[t]
        arj = incident_vals_h[t]
        row == pair_row && continue
        abs(arj) > zero_tol || continue
        push!(shift_rows, row)
        push!(shift_vals, -arj * beta)
        for s in eachindex(support_cols_h)
            delta = arj * gamma_h[s]
            abs(delta) > zero_tol || continue
            push!(delta_rows, row)
            push!(delta_cols, support_cols_h[s])
            push!(delta_vals, delta)
        end
    end

    return delta_rows, delta_cols, delta_vals, shift_rows, shift_vals
end

function apply_rule_qp_linear_eq_agg!(
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
    (m == 0 || n == 0) && return nothing

    candidate_mask = CUDA.zeros(UInt8, m)
    pivot_col_out = CUDA.zeros(Int32, m)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_mark_qp_linear_eq_agg_candidates!(
        candidate_mask,
        pivot_col_out,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.row_nnz,
        stats.col_nnz,
        plan.new_q_diag,
        stats.q_offdiag_nnz,
        plan.new_c,
        plan.new_AL,
        plan.new_AU,
        qp.A.rowPtr,
        qp.A.colVal,
        pparams.zero_tol,
        pparams.bound_tol,
        Int32(pparams.qp_linear_eq_agg_max_support),
        Int32(m),
    )
    _, candidate_rows, candidate_count = build_maps_from_mask(candidate_mask)
    Int(candidate_count) == 0 && return nothing

    pair_row = Int32(_copy_scalar_to_host(candidate_rows, 1))
    elim_col = Int32(_copy_scalar_to_host(pivot_col_out, Int(pair_row)))

    row_cols_h, row_vals_h = _qp_csr_row_payload_host(qp.A, pair_row)
    pivot_pos = findfirst(==(elim_col), row_cols_h)
    isnothing(pivot_pos) && return nothing
    aij = row_vals_h[pivot_pos]
    abs(aij) > pparams.zero_tol || return nothing

    b = Float64(_copy_scalar_to_host(plan.new_AU, Int(pair_row)))
    old_l = Float64(_copy_scalar_to_host(plan.new_l, Int(elim_col)))
    old_u = Float64(_copy_scalar_to_host(plan.new_u, Int(elim_col)))
    cj = Float64(_copy_scalar_to_host(plan.new_c, Int(elim_col)))
    qjj = Float64(_copy_scalar_to_host(plan.new_q_diag, Int(elim_col)))
    qoff = Int(_copy_scalar_to_host(stats.q_offdiag_nnz, Int(elim_col)))
    (abs(cj) <= pparams.zero_tol && abs(qjj) <= pparams.zero_tol && qoff == 0) || return nothing

    support_cols_h = Int32[]
    support_coeffs_h = Float64[]
    for t in eachindex(row_cols_h)
        row_cols_h[t] == elim_col && continue
        push!(support_cols_h, row_cols_h[t])
        push!(support_coeffs_h, row_vals_h[t])
    end
    isempty(support_cols_h) && return nothing

    beta = b / aij
    gamma_h = -support_coeffs_h ./ aij
    incident_rows_h, incident_vals_h = _qp_csr_row_payload_host(qp.AT, elim_col)
    delta_rows_h, delta_cols_h, delta_vals_h, shift_rows_h, shift_vals_h = _qp_linear_eq_agg_delta_host(
        support_cols_h,
        incident_rows_h,
        incident_vals_h,
        pair_row,
        beta,
        gamma_h,
        pparams.zero_tol,
    )

    if !isempty(delta_rows_h)
        delta_A = _build_doubleton_eq_delta_csr(
            CuVector(delta_rows_h),
            CuVector(delta_cols_h),
            CuVector(delta_vals_h),
            size(qp.A),
        )
        plan.new_A = qp.A + delta_A
    else
        plan.new_A = qp.A
    end

    if !isempty(shift_rows_h)
        shift_rows = CuVector(shift_rows_h)
        shift_vals = CuVector(shift_vals_h)
        blocks_shift = cld(length(shift_rows_h), GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_shift _kernel_qp_accumulate_sparse_updates!(
            plan.new_AL,
            shift_rows,
            shift_vals,
            Int32(length(shift_rows_h)),
        )
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_shift _kernel_qp_accumulate_sparse_updates!(
            plan.new_AU,
            shift_rows,
            shift_vals,
            Int32(length(shift_rows_h)),
        )
    end

    row_deleted = false
    if aij > 0.0
        row_lb = isfinite(old_u) ? b - aij * old_u : -Inf
        row_ub = isfinite(old_l) ? b - aij * old_l : Inf
        if !isfinite(old_u) && !isfinite(old_l)
            row_deleted = true
        else
            CUDA.@allowscalar plan.new_AL[pair_row] = row_lb
            CUDA.@allowscalar plan.new_AU[pair_row] = row_ub
        end
    else
        row_lb = isfinite(old_l) ? b - aij * old_l : -Inf
        row_ub = isfinite(old_u) ? b - aij * old_u : Inf
        if !isfinite(old_l) && !isfinite(old_u)
            row_deleted = true
        else
            CUDA.@allowscalar plan.new_AL[pair_row] = row_lb
            CUDA.@allowscalar plan.new_AU[pair_row] = row_ub
        end
    end

    if row_deleted
        CUDA.@allowscalar plan.keep_row_mask[pair_row] = UInt8(0)
    end

    CUDA.@allowscalar plan.keep_col_mask[elim_col] = UInt8(0)
    plan.singleton_col_row_idx = concat_cuvector(plan.singleton_col_row_idx, CuVector(Int32[pair_row]))
    plan.singleton_col_col_idx = concat_cuvector(plan.singleton_col_col_idx, CuVector(Int32[elim_col]))

    support_counts = CuVector(Int32[length(support_cols_h)])
    support_starts = CuVector(Int32[1])
    flat_support_cols = CuVector(support_cols_h)
    flat_support_coeffs = CuVector(support_coeffs_h)
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
        0.0,
        row_deleted,
    )

    plan.has_change = true
    return nothing
end
