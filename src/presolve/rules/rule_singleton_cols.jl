"""
Layer-2 rule: singleton columns.

Current GPU scope:
- detect singleton columns from `AT`
- process equality-row singleton columns from the document formula
- align the implied-free two-sided inequality elimination branch with PSLP
- substitute the singleton variable into the objective
- delete the equality row if both box sides are already implied
- otherwise replace the equality by the surviving one-sided inequality

At most one singleton column is processed per row per pass to avoid row-update
conflicts.
"""

const _SINGLETON_COL_INEQ_NO_ACTION = UInt8(0)
const _SINGLETON_COL_INEQ_ELIMINATE = UInt8(1)
const _SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS = UInt8(2)
const _SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS = UInt8(3)

@inline function _mark_unbounded_singleton_cols!(plan::PresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

@inline function _singleton_col_direct_unbounded(
    cj::Float64,
    a::Float64,
    lhs::Float64,
    rhs::Float64,
    lb::Float64,
    ub::Float64,
    zero_tol::Float64,
)
    return ((cj > zero_tol && a > zero_tol && !isfinite(lhs) && !isfinite(lb)) ||
            (cj > zero_tol && a < -zero_tol && !isfinite(rhs) && !isfinite(lb)) ||
            (cj < -zero_tol && a < -zero_tol && !isfinite(lhs) && !isfinite(ub)) ||
            (cj < -zero_tol && a > zero_tol && !isfinite(rhs) && !isfinite(ub)))
end

@inline function _singleton_col_eq_free_from_above(
    implied_ub::Float64,
    ub::Float64,
    tol::Float64,
)
    return !isfinite(ub) || implied_ub <= ub + tol
end

@inline function _singleton_col_eq_free_from_below(
    implied_lb::Float64,
    lb::Float64,
    tol::Float64,
)
    return !isfinite(lb) || implied_lb >= lb - tol
end

@inline function _apply_singleton_col_eq_one_sided_row_update!(
    AL,
    AU,
    row,
    a,
    bound_val,
    keep_lower_part,
)
    @inbounds shifted_rhs = AU[row] - a * bound_val
    if keep_lower_part
        @inbounds AL[row] = shifted_rhs
        @inbounds AU[row] = Inf
    else
        @inbounds AL[row] = -Inf
        @inbounds AU[row] = shifted_rhs
    end
    return nothing
end

function _singleton_col_activity_bounds(
    row_start,
    row_stop,
    excluded_col,
    keep_col,
    col_val,
    nz_val,
    l,
    u,
)
    rest_min = 0.0
    rest_max = 0.0

    if row_start > row_stop
        return (rest_min, rest_max, true)
    end

    for p in row_start:row_stop
        @inbounds col = col_val[p]
        if col == excluded_col || keep_col[col] == UInt8(0)
            continue
        end

        @inbounds a = nz_val[p]
        if a >= 0.0
            @inbounds term_min = a * l[col]
            @inbounds term_max = a * u[col]
        else
            @inbounds term_min = a * u[col]
            @inbounds term_max = a * l[col]
        end

        if isnan(term_min) || isnan(term_max)
            return (0.0, 0.0, false)
        end

        rest_min += term_min
        rest_max += term_max
        if isnan(rest_min) || isnan(rest_max)
            return (0.0, 0.0, false)
        end
    end

    return (rest_min, rest_max, true)
end

@inline function _singleton_col_implied_free_from_above(
    a::Float64,
    lhs::Float64,
    rhs::Float64,
    ub::Float64,
    rest_min::Float64,
    rest_max::Float64,
    tol::Float64,
)
    !isfinite(ub) && return true

    implied_ub = Inf
    if a > 0.0 && isfinite(rhs)
        implied_ub = (rhs - rest_min) / a
    elseif a < 0.0 && isfinite(lhs)
        implied_ub = (lhs - rest_max) / a
    end

    return implied_ub <= ub + tol
end

@inline function _singleton_col_implied_free_from_below(
    a::Float64,
    lhs::Float64,
    rhs::Float64,
    lb::Float64,
    rest_min::Float64,
    rest_max::Float64,
    tol::Float64,
)
    !isfinite(lb) && return true

    implied_lb = -Inf
    if a > 0.0 && isfinite(lhs)
        implied_lb = (lhs - rest_max) / a
    elseif a < 0.0 && isfinite(rhs)
        implied_lb = (rhs - rest_min) / a
    end

    return implied_lb >= lb - tol
end

@inline function _singleton_col_active_side(
    cj::Float64,
    a::Float64,
    lhs::Float64,
    rhs::Float64,
    zero_tol::Float64,
)
    if (cj > zero_tol && a > 0.0) || (cj < -zero_tol && a < 0.0)
        return lhs
    elseif (cj > zero_tol && a < 0.0) || (cj < -zero_tol && a > 0.0)
        return rhs
    end
    return isfinite(lhs) ? lhs : rhs
end

@inline function _singleton_col_ineq_action(
    cj::Float64,
    a::Float64,
    lhs::Float64,
    rhs::Float64,
    impl_free_from_above::Bool,
    impl_free_from_below::Bool,
    zero_tol::Float64,
)
    if impl_free_from_above && impl_free_from_below
        active_side = _singleton_col_active_side(cj, a, lhs, rhs, zero_tol)
        if isfinite(active_side)
            return (_SINGLETON_COL_INEQ_ELIMINATE, active_side)
        end
        return (_SINGLETON_COL_INEQ_NO_ACTION, active_side)
    end

    tighten_lhs_to_rhs =
        ((cj < -zero_tol && a > 0.0 && impl_free_from_above) ||
         (cj > zero_tol && a < 0.0 && impl_free_from_below)) &&
        isfinite(rhs)
    if tighten_lhs_to_rhs
        return (_SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS, rhs)
    end

    tighten_rhs_to_lhs =
        ((cj > zero_tol && a > 0.0 && impl_free_from_below) ||
         (cj < -zero_tol && a < 0.0 && impl_free_from_above)) &&
        isfinite(lhs)
    if tighten_rhs_to_lhs
        return (_SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS, lhs)
    end

    return (_SINGLETON_COL_INEQ_NO_ACTION, 0.0)
end

function _kernel_singleton_col_row_owner!(
    status_flag,
    row_owner,
    keep_row,
    keep_col,
    singleton_mask,
    support_row,
    support_val,
    c_cur,
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
        if keep_col[j] == UInt8(0) || singleton_mask[j] == UInt8(0)
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
        @inbounds cj = c_cur[j]
        if _singleton_col_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

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

        if isfinite(lhs) && isfinite(rhs) && abs(lhs - rhs) <= tol
            x1 = (rhs - rest_min) / a
            x2 = (rhs - rest_max) / a
            implied_lb = min(x1, x2)
            implied_ub = max(x1, x2)

            @inbounds impl_free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol)
            @inbounds impl_free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol)
            (impl_free_from_above || impl_free_from_below) || return
        else
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
            @inbounds ineq_action, _ = _singleton_col_ineq_action(
                c_cur[j],
                a,
                lhs,
                rhs,
                impl_free_from_above,
                impl_free_from_below,
                zero_tol,
            )
            ineq_action != _SINGLETON_COL_INEQ_NO_ACTION || return
        end

        CUDA.@atomic row_owner[row] = min(row_owner[row], Int32(j))
    end
    return
end

function _kernel_process_singleton_cols!(
    status_flag,
    row_delete,
    row_lhs_change,
    row_rhs_change,
    col_delete,
    pair_row,
    chosen_side,
    c_new,
    AL_new,
    AU_new,
    obj_contrib,
    row_owner,
    keep_row,
    keep_col,
    singleton_mask,
    support_row,
    support_val,
    c_cur,
    l_cur,
    u_cur,
    row_ptr,
    col_val,
    nz_val,
    tol,
    zero_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if keep_col[j] == UInt8(0) || singleton_mask[j] == UInt8(0)
            return
        end

        @inbounds row = support_row[j]
        if row < 1 || row > length(keep_row) || keep_row[row] == UInt8(0)
            return
        end
        @inbounds row_owner[row] == Int32(j) || return

        @inbounds a = support_val[j]
        abs(a) > zero_tol || return

        @inbounds lhs = AL_new[row]
        @inbounds rhs = AU_new[row]
        @inbounds cj = c_cur[j]
        if _singleton_col_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
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

        if isfinite(lhs) && isfinite(rhs) && abs(lhs - rhs) <= tol
            x1 = (rhs - rest_min) / a
            x2 = (rhs - rest_max) / a
            implied_lb = min(x1, x2)
            implied_ub = max(x1, x2)

            @inbounds impl_free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol)
            @inbounds impl_free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol)
            (impl_free_from_above || impl_free_from_below) || return

            @inbounds chosen_side[j] = rhs
            @inbounds obj_contrib[j] = cj * rhs / a

            if row_start <= row_stop
                for p in row_start:row_stop
                    @inbounds col = col_val[p]
                    if col == j || keep_col[col] == UInt8(0)
                        continue
                    end
                    @inbounds shift = -(cj * nz_val[p] / a)
                    CUDA.@atomic c_new[col] += shift
                end
            end

            if impl_free_from_above && impl_free_from_below
                @inbounds row_delete[row] = UInt8(1)
            elseif impl_free_from_above
                @inbounds bound_val = l_cur[j]
                _apply_singleton_col_eq_one_sided_row_update!(
                    AL_new,
                    AU_new,
                    row,
                    a,
                    bound_val,
                    a < 0.0,
                )
            else
                @inbounds bound_val = u_cur[j]
                _apply_singleton_col_eq_one_sided_row_update!(
                    AL_new,
                    AU_new,
                    row,
                    a,
                    bound_val,
                    a > 0.0,
                )
            end

            @inbounds col_delete[j] = UInt8(1)
            @inbounds pair_row[j] = row
            return
        end

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
        @inbounds ineq_action, action_side = _singleton_col_ineq_action(
            cj,
            a,
            lhs,
            rhs,
            impl_free_from_above,
            impl_free_from_below,
            zero_tol,
        )
        ineq_action != _SINGLETON_COL_INEQ_NO_ACTION || return

        if ineq_action == _SINGLETON_COL_INEQ_ELIMINATE
            @inbounds chosen_side[j] = action_side
            @inbounds obj_contrib[j] = cj * action_side / a

            if row_start <= row_stop
                for p in row_start:row_stop
                    @inbounds col = col_val[p]
                    if col == j || keep_col[col] == UInt8(0)
                        continue
                    end
                    @inbounds shift = -(cj * nz_val[p] / a)
                    CUDA.@atomic c_new[col] += shift
                end
            end

            @inbounds row_delete[row] = UInt8(1)
            @inbounds col_delete[j] = UInt8(1)
            @inbounds pair_row[j] = row
        elseif ineq_action == _SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS
            @inbounds row_lhs_change[row] = UInt8(1)
            @inbounds AL_new[row] = rhs
            @inbounds AU_new[row] = rhs
        else
            @inbounds row_rhs_change[row] = UInt8(1)
            @inbounds AL_new[row] = lhs
            @inbounds AU_new[row] = lhs
        end
    end
    return
end

function _kernel_singleton_col_support_counts!(
    support_counts,
    eliminated_cols,
    pair_rows,
    keep_col,
    row_ptr,
    col_val,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds elim_col = eliminated_cols[t]
        @inbounds row = pair_rows[t]
        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)

        count = Int32(0)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                if col != elim_col && keep_col[col] != UInt8(0)
                    count += Int32(1)
                end
            end
        end
        @inbounds support_counts[t] = count
    end
    return
end

function _kernel_singleton_col_support_payload!(
    flat_support_cols,
    flat_support_coeffs,
    support_starts,
    eliminated_cols,
    pair_rows,
    keep_col,
    row_ptr,
    col_val,
    nz_val,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds elim_col = eliminated_cols[t]
        @inbounds row = pair_rows[t]
        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
        @inbounds write_pos = support_starts[t]

        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                if col != elim_col && keep_col[col] != UInt8(0)
                    @inbounds flat_support_cols[write_pos] = col
                    @inbounds flat_support_coeffs[write_pos] = nz_val[p]
                    write_pos += Int32(1)
                end
            end
        end
    end
    return
end

function _append_singleton_col_postsolve_records!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    eliminated_cols,
    pair_rows,
    row_delete,
    chosen_side,
    old_l,
    old_u,
    elim_obj,
    record_cpu::Bool,
)
    eliminated_count = length(eliminated_cols)
    eliminated_count == 0 && return nothing

    support_counts = CUDA.zeros(Int32, eliminated_count)
    blocks = cld(eliminated_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_col_support_counts!(
        support_counts,
        eliminated_cols,
        pair_rows,
        plan.keep_col_mask,
        lp.A.rowPtr,
        lp.A.colVal,
        Int32(eliminated_count),
    )

    support_prefix = cumsum(support_counts)
    total_support = Int(_copy_scalar_to_host(support_prefix, eliminated_count))
    support_starts = CUDA.fill(Int32(1), eliminated_count)
    if eliminated_count > 1
        support_starts[2:end] .= support_prefix[1:(end - 1)] .+ Int32(1)
    end

    flat_support_cols = CuVector{Int32}(undef, total_support)
    flat_support_coeffs = CuVector{Float64}(undef, total_support)
    if total_support > 0
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_col_support_payload!(
            flat_support_cols,
            flat_support_coeffs,
            support_starts,
            eliminated_cols,
            pair_rows,
            plan.keep_col_mask,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.A.nzVal,
            Int32(eliminated_count),
        )
    end

    pivot_coeff = gather_by_red2org(stats.singleton_col_val, eliminated_cols)
    row_deleted_for_pairs = gather_by_red2org(row_delete, pair_rows)

    append_sub_col_records_from_payload_gpu!(
        plan.tape_gpu,
        eliminated_cols,
        pair_rows,
        support_counts,
        support_starts,
        flat_support_cols,
        flat_support_coeffs,
        pivot_coeff,
        chosen_side,
        old_l,
        old_u,
        elim_obj,
        row_deleted_for_pairs;
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if record_cpu
        append_sub_col_records_from_payload!(
            plan.tape,
            eliminated_cols,
            pair_rows,
            support_counts,
            support_starts,
            flat_support_cols,
            flat_support_coeffs,
            pivot_coeff,
            chosen_side,
            old_l,
            old_u,
            elim_obj,
            row_deleted_for_pairs;
            dual_mode=POSTSOLVE_DUAL_MINIMAL,
        )
    end
    return nothing
end

"""
Rule: eliminate equality-row singleton columns using the exact substitution
formula from the document.
"""
function apply_rule_singleton_cols!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
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

    status_flag = CUDA.zeros(Int32, 1)
    row_owner = CUDA.fill(typemax(Int32), m)
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
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status != 0
        _mark_unbounded_singleton_cols!(
            plan,
            "Singleton-column unboundedness: an inequality singleton has an improving infinite direction.",
        )
        return nothing
    end

    col_delete = CUDA.zeros(UInt8, n)
    row_delete = CUDA.zeros(UInt8, m)
    row_lhs_change = CUDA.zeros(UInt8, m)
    row_rhs_change = CUDA.zeros(UInt8, m)
    pair_row = CUDA.fill(Int32(0), n)
    chosen_side = CUDA.zeros(Float64, n)
    c_new = copy(plan.new_c)
    AL_new = copy(plan.new_AL)
    AU_new = copy(plan.new_AU)
    obj_contrib = CUDA.zeros(Float64, n)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_process_singleton_cols!(
        status_flag,
        row_delete,
        row_lhs_change,
        row_rhs_change,
        col_delete,
        pair_row,
        chosen_side,
        c_new,
        AL_new,
        AU_new,
        obj_contrib,
        row_owner,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status != 0
        _mark_unbounded_singleton_cols!(
            plan,
            "Singleton-column unboundedness: an inequality singleton has an improving infinite direction.",
        )
        return nothing
    end

    _, eliminated_cols, eliminated_count = build_maps_from_mask(col_delete)
    _, lhs_changed_rows, lhs_changed_count = build_maps_from_mask(row_lhs_change)
    _, rhs_changed_rows, rhs_changed_count = build_maps_from_mask(row_rhs_change)
    row_tightened = Int(lhs_changed_count) > 0 || Int(rhs_changed_count) > 0

    if Int(eliminated_count) == 0 && !row_tightened
        return nothing
    end

    copyto!(
        plan.keep_row_mask,
        UInt8.((plan.keep_row_mask .!= UInt8(0)) .& (row_delete .== UInt8(0))),
    )
    copyto!(
        plan.keep_col_mask,
        UInt8.((plan.keep_col_mask .!= UInt8(0)) .& (col_delete .== UInt8(0))),
    )
    copyto!(plan.new_c, c_new)
    copyto!(plan.new_AL, AL_new)
    copyto!(plan.new_AU, AU_new)

    pair_rows = gather_by_red2org(pair_row, eliminated_cols)
    chosen_side_sel = gather_by_red2org(chosen_side, eliminated_cols)
    old_l = gather_by_red2org(plan.new_l, eliminated_cols)
    old_u = gather_by_red2org(plan.new_u, eliminated_cols)
    elim_obj = gather_by_red2org(plan.new_c, eliminated_cols)
    if pparams.record_postsolve_tape
        if Int(eliminated_count) > 0
            _append_singleton_col_postsolve_records!(
                plan,
                lp,
                stats,
                eliminated_cols,
                pair_rows,
                row_delete,
                chosen_side_sel,
                old_l,
                old_u,
                elim_obj,
                pparams.record_postsolve_tape_cpu,
            )
        end

        if Int(lhs_changed_count) > 0
            for t in 1:Int(lhs_changed_count)
                row = CUDA.@allowscalar Int(lhs_changed_rows[t])
                old_al = CUDA.@allowscalar Float64(plan.new_AL[row])
                old_au = CUDA.@allowscalar Float64(plan.new_AU[row])
                new_al = CUDA.@allowscalar Float64(AL_new[row])
                new_au = CUDA.@allowscalar Float64(AU_new[row])
                append_lhs_change_record!(
                    plan.tape,
                    row,
                    old_al,
                    old_au,
                    new_al,
                    new_au;
                    dual_mode=POSTSOLVE_DUAL_MINIMAL,
                )
            end
        end

        if Int(rhs_changed_count) > 0
            for t in 1:Int(rhs_changed_count)
                row = CUDA.@allowscalar Int(rhs_changed_rows[t])
                old_al = CUDA.@allowscalar Float64(plan.new_AL[row])
                old_au = CUDA.@allowscalar Float64(plan.new_AU[row])
                new_al = CUDA.@allowscalar Float64(AL_new[row])
                new_au = CUDA.@allowscalar Float64(AU_new[row])
                append_rhs_change_record!(
                    plan.tape,
                    row,
                    old_al,
                    old_au,
                    new_al,
                    new_au;
                    dual_mode=POSTSOLVE_DUAL_MINIMAL,
                )
            end
        end
    end

    if Int(eliminated_count) > 0
        plan.singleton_col_row_idx = concat_cuvector(plan.singleton_col_row_idx, pair_rows)
        plan.singleton_col_col_idx = concat_cuvector(plan.singleton_col_col_idx, eliminated_cols)
    end
    plan.obj_constant_delta += _stable_presolve_objective_sum(obj_contrib)
    plan.has_row_action = true
    plan.has_col_action = true
    plan.has_change = true
    return nothing
end
