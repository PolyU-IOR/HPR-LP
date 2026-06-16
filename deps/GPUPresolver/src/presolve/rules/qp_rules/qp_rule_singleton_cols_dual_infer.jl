"""
QP layer-2 rule: singleton-column dual inference.

Conservative QP-aware variant:
- only inequality-row singleton columns
- only separable columns with Q_{Rj} = 0
- tighten the live row to equality only when the one-dimensional derivative
  keeps a fixed sign over the row-implied interval

This rule does not change Q or remove columns.
"""

@inline function _mark_infeasible_qp_singleton_cols_dual_infer!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

function _kernel_qp_singleton_dual_infer_row_owner!(
    status_flag,
    row_owner,
    keep_row,
    keep_col,
    singleton_mask,
    support_row,
    support_val,
    q_offdiag_nnz,
    c_cur,
    q_diag_cur,
    l_cur,
    u_cur,
    AL_cur,
    AU_cur,
    row_ptr,
    col_val,
    nz_val,
    tol,
    zero_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if keep_col[j] == UInt8(0) || singleton_mask[j] == UInt8(0) || q_offdiag_nnz[j] != Int32(0)
            return
        end

        @inbounds row = support_row[j]
        if row < 1 || row > length(keep_row) || keep_row[row] == UInt8(0)
            return
        end

        @inbounds a = support_val[j]
        abs(a) > zero_tol || return

        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
        live_row_nnz = Int32(0)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                if keep_col[col] != UInt8(0)
                    live_row_nnz += Int32(1)
                    live_row_nnz > Int32(1) && break
                end
            end
        end
        live_row_nnz > Int32(1) || return

        @inbounds lhs = AL_cur[row]
        @inbounds rhs = AU_cur[row]
        is_eq_row = isfinite(lhs) && isfinite(rhs) && abs(lhs - rhs) <= tol
        is_eq_row && return

        rest_min, rest_max, rest_finite = _singleton_col_activity_bounds(
            row_start,
            row_stop,
            j,
            keep_col,
            col_val,
            nz_val,
            l_cur,
            u_cur,
        )
        rest_finite || return

        impl_free_from_above = _singleton_col_implied_free_from_above(
            a,
            lhs,
            rhs,
            u_cur[j],
            rest_min,
            rest_max,
            tol,
        )
        impl_free_from_below = _singleton_col_implied_free_from_below(
            a,
            lhs,
            rhs,
            l_cur[j],
            rest_min,
            rest_max,
            tol,
        )
        (impl_free_from_above && impl_free_from_below) || return

        implied_lb = -Inf
        implied_ub = Inf
        if a > 0.0
            isfinite(lhs) && (implied_lb = (lhs - rest_max) / a)
            isfinite(rhs) && (implied_ub = (rhs - rest_min) / a)
        else
            isfinite(rhs) && (implied_lb = (rhs - rest_min) / a)
            isfinite(lhs) && (implied_ub = (lhs - rest_max) / a)
        end

        if !isfinite(implied_lb) || !isfinite(implied_ub) || implied_lb > implied_ub + tol
            return
        end

        if implied_lb > u_cur[j] + tol || implied_ub < l_cur[j] - tol
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        @inbounds qjj = q_diag_cur[j]
        @inbounds cj = c_cur[j]
        dmin = 0.0
        dmax = 0.0
        if qjj > zero_tol
            dmin = qjj * implied_lb + cj
            dmax = qjj * implied_ub + cj
        elseif qjj < -zero_tol
            dmin = qjj * implied_ub + cj
            dmax = qjj * implied_lb + cj
        else
            dmin = cj
            dmax = cj
        end

        if dmin >= -tol || dmax <= tol
            CUDA.@atomic row_owner[row] = min(row_owner[row], Int32(j))
        end
    end
    return
end

function apply_rule_qp_singleton_cols_dual_infer!(
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
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_singleton_dual_infer_row_owner!(
        status_flag,
        row_owner,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        stats.q_offdiag_nnz,
        plan.new_c,
        plan.new_q_diag,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
        qp.A.rowPtr,
        qp.A.colVal,
        qp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        Int32(n),
    )

    if Int(_copy_scalar_to_host(status_flag, 1)) != 0
        _mark_infeasible_qp_singleton_cols_dual_infer!(
            plan,
            "QP singleton-cols-dual-infer infeasibility: implied row interval is disjoint from the current bounds.",
        )
        return nothing
    end

    candidate_mask = UInt8.(row_owner .!= typemax(Int32))
    _, candidate_rows, candidate_count = build_maps_from_mask(candidate_mask)
    Int(candidate_count) == 0 && return nothing

    pair_row = Int32(_copy_scalar_to_host(candidate_rows, 1))
    elim_col = Int32(_copy_scalar_to_host(row_owner, Int(pair_row)))
    a = Float64(_copy_scalar_to_host(stats.singleton_col_val, Int(elim_col)))
    lhs = Float64(_copy_scalar_to_host(plan.new_AL, Int(pair_row)))
    rhs = Float64(_copy_scalar_to_host(plan.new_AU, Int(pair_row)))
    qjj = Float64(_copy_scalar_to_host(plan.new_q_diag, Int(elim_col)))
    cj = Float64(_copy_scalar_to_host(plan.new_c, Int(elim_col)))

    support_counts, support_starts, flat_support_cols, flat_support_coeffs =
        _qp_singleton_eq_support_payload(qp, plan, elim_col, pair_row)
    support_cols_h = _copy_vector_to_host(flat_support_cols)
    support_coeffs_h = _copy_vector_to_host(flat_support_coeffs)
    support_count = length(support_cols_h)
    support_count == 0 && return nothing

    rest_min = 0.0
    rest_max = 0.0
    for t in eachindex(support_cols_h)
        col = Int(support_cols_h[t])
        coeff = support_coeffs_h[t]
        l = Float64(_copy_scalar_to_host(plan.new_l, col))
        u = Float64(_copy_scalar_to_host(plan.new_u, col))
        if coeff >= 0.0
            rest_min += coeff * l
            rest_max += coeff * u
        else
            rest_min += coeff * u
            rest_max += coeff * l
        end
    end

    implied_lb = -Inf
    implied_ub = Inf
    if a > 0.0
        isfinite(lhs) && (implied_lb = (lhs - rest_max) / a)
        isfinite(rhs) && (implied_ub = (rhs - rest_min) / a)
    else
        isfinite(rhs) && (implied_lb = (rhs - rest_min) / a)
        isfinite(lhs) && (implied_ub = (lhs - rest_max) / a)
    end

    dmin = 0.0
    dmax = 0.0
    if qjj > pparams.zero_tol
        dmin = qjj * implied_lb + cj
        dmax = qjj * implied_ub + cj
    elseif qjj < -pparams.zero_tol
        dmin = qjj * implied_ub + cj
        dmax = qjj * implied_lb + cj
    else
        dmin = cj
        dmax = cj
    end

    action_side = if dmin >= -pparams.bound_tol
        a > 0.0 ? lhs : rhs
    elseif dmax <= pparams.bound_tol
        a > 0.0 ? rhs : lhs
    else
        return nothing
    end

    CUDA.@allowscalar plan.new_AL[pair_row] = action_side
    CUDA.@allowscalar plan.new_AU[pair_row] = action_side
    append_eq_to_ineq_records_gpu!(
        plan.tape_gpu,
        CuVector(Int32[pair_row]);
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if pparams.record_postsolve_tape_cpu
        plan.tape = PostsolveTape(plan.tape_gpu)
    end

    plan.has_change = true
    return nothing
end
