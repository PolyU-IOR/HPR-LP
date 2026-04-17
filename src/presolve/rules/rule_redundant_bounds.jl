"""
Layer-2 final cleanup rule: redundant bounds.

Current scope:
- remove a finite upper bound only for `(-Inf, uj]` boxes when some live row already implies it
- remove a finite lower bound only for `[lj, +Inf)` boxes when some live row already implies it
- allow one-sided residual proofs; the unused residual side may stay infinite
"""

function _kernel_redundant_bounds_candidates!(
    drop_lower,
    drop_upper,
    l_cur,
    u_cur,
    AL,
    AU,
    row_ptr,
    col_val,
    nz_val,
    at_row_ptr,
    at_row_val,
    at_nz_val,
    zero_tol,
    tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        @inbounds lj = l_cur[j]
        @inbounds uj = u_cur[j]

        if lj == -Inf && isfinite(uj)
            @inbounds p_start = at_row_ptr[j]
            @inbounds p_stop = at_row_ptr[j + 1] - 1
            if p_start <= p_stop
                for p in p_start:p_stop
                    @inbounds i = at_row_val[p]
                    @inbounds a = at_nz_val[p]
                    if abs(a) <= zero_tol
                        continue
                    end

                    @inbounds row_start = row_ptr[i]
                    @inbounds row_stop = row_ptr[i + 1] - 1
                    implied_u = Inf

                    if a > zero_tol && isfinite(AU[i])
                        rest_min = 0.0
                        finite_rest = true
                        for q in row_start:row_stop
                            @inbounds col = col_val[q]
                            col == j && continue
                            @inbounds aq = nz_val[q]
                            @inbounds term_min = aq >= 0.0 ? (aq * l_cur[col]) : (aq * u_cur[col])
                            if !isfinite(term_min)
                                finite_rest = false
                                break
                            end
                            rest_min += term_min
                        end
                        if finite_rest
                            implied_u = (AU[i] - rest_min) / a
                        end
                    elseif a < -zero_tol && isfinite(AL[i])
                        rest_max = 0.0
                        finite_rest = true
                        for q in row_start:row_stop
                            @inbounds col = col_val[q]
                            col == j && continue
                            @inbounds aq = nz_val[q]
                            @inbounds term_max = aq >= 0.0 ? (aq * u_cur[col]) : (aq * l_cur[col])
                            if !isfinite(term_max)
                                finite_rest = false
                                break
                            end
                            rest_max += term_max
                        end
                        if finite_rest
                            implied_u = (AL[i] - rest_max) / a
                        end
                    end

                    if isfinite(implied_u) && implied_u <= uj + tol
                        @inbounds drop_upper[j] = UInt8(1)
                        break
                    end
                end
            end
        elseif isfinite(lj) && uj == Inf
            @inbounds p_start = at_row_ptr[j]
            @inbounds p_stop = at_row_ptr[j + 1] - 1
            if p_start <= p_stop
                for p in p_start:p_stop
                    @inbounds i = at_row_val[p]
                    @inbounds a = at_nz_val[p]
                    if abs(a) <= zero_tol
                        continue
                    end

                    @inbounds row_start = row_ptr[i]
                    @inbounds row_stop = row_ptr[i + 1] - 1
                    implied_l = -Inf

                    if a > zero_tol && isfinite(AL[i])
                        rest_max = 0.0
                        finite_rest = true
                        for q in row_start:row_stop
                            @inbounds col = col_val[q]
                            col == j && continue
                            @inbounds aq = nz_val[q]
                            @inbounds term_max = aq >= 0.0 ? (aq * u_cur[col]) : (aq * l_cur[col])
                            if !isfinite(term_max)
                                finite_rest = false
                                break
                            end
                            rest_max += term_max
                        end
                        if finite_rest
                            implied_l = (AL[i] - rest_max) / a
                        end
                    elseif a < -zero_tol && isfinite(AU[i])
                        rest_min = 0.0
                        finite_rest = true
                        for q in row_start:row_stop
                            @inbounds col = col_val[q]
                            col == j && continue
                            @inbounds aq = nz_val[q]
                            @inbounds term_min = aq >= 0.0 ? (aq * l_cur[col]) : (aq * u_cur[col])
                            if !isfinite(term_min)
                                finite_rest = false
                                break
                            end
                            rest_min += term_min
                        end
                        if finite_rest
                            implied_l = (AU[i] - rest_min) / a
                        end
                    end

                    if isfinite(implied_l) && implied_l >= lj - tol
                        @inbounds drop_lower[j] = UInt8(1)
                        break
                    end
                end
            end
        end
    end
    return
end

function _kernel_compute_redundant_bounds_row_min_keys!(
    row_min_key,
    drop_lower,
    drop_upper,
    row_ptr,
    col_val,
    at_row_ptr,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - 1
        best_key = typemax(UInt64)

        for q in row_start:row_stop
            @inbounds j = col_val[q]
            if drop_lower[j] != UInt8(0) || drop_upper[j] != UInt8(0)
                @inbounds nnz_j = at_row_ptr[j + 1] - at_row_ptr[j]
                key = (UInt64(UInt32(nnz_j)) << 32) | UInt64(UInt32(j))
                if key < best_key
                    best_key = key
                end
            end
        end

        @inbounds row_min_key[i] = best_key
    end
    return
end

function _kernel_select_redundant_bounds_batch!(
    selected_lower,
    selected_upper,
    drop_lower,
    drop_upper,
    row_min_key,
    at_row_ptr,
    at_row_val,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if drop_lower[j] == UInt8(0) && drop_upper[j] == UInt8(0)
            return
        end

        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - 1
        @inbounds nnz_j = at_row_ptr[j + 1] - at_row_ptr[j]
        key = (UInt64(UInt32(nnz_j)) << 32) | UInt64(UInt32(j))

        choose = true
        for p in p_start:p_stop
            @inbounds i = at_row_val[p]
            if row_min_key[i] != key
                choose = false
                break
            end
        end

        if choose
            @inbounds selected_lower[j] = drop_lower[j]
            @inbounds selected_upper[j] = drop_upper[j]
        end
    end
    return
end

function _kernel_apply_selected_redundant_bounds!(
    l_cur,
    u_cur,
    selected_lower,
    selected_upper,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if selected_lower[j] != UInt8(0)
            @inbounds l_cur[j] = -Inf
        elseif selected_upper[j] != UInt8(0)
            @inbounds u_cur[j] = Inf
        end
    end
    return
end

"""
Rule: final one-sided implied-free redundant-bound cleanup.
"""
function apply_rule_redundant_bounds!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    _stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    _, n = size(lp.A)
    if n == 0
        return nothing
    end

    m, _ = size(lp.A)
    drop_lower = CUDA.zeros(UInt8, n)
    drop_upper = CUDA.zeros(UInt8, n)
    row_min_key = CUDA.fill(typemax(UInt64), m)
    selected_lower = CUDA.zeros(UInt8, n)
    selected_upper = CUDA.zeros(UInt8, n)
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    row_blocks = cld(m, GPU_PRESOLVE_THREADS)
    changed_any = false

    while true
        fill!(drop_lower, UInt8(0))
        fill!(drop_upper, UInt8(0))
        fill!(selected_lower, UInt8(0))
        fill!(selected_upper, UInt8(0))
        fill!(row_min_key, typemax(UInt64))

        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_redundant_bounds_candidates!(
            drop_lower,
            drop_upper,
            plan.new_l,
            plan.new_u,
            plan.new_AL,
            plan.new_AU,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.A.nzVal,
            lp.AT.rowPtr,
            lp.AT.colVal,
            lp.AT.nzVal,
            pparams.zero_tol,
            pparams.bound_tol,
            Int32(n),
        )

        @cuda threads=GPU_PRESOLVE_THREADS blocks=row_blocks _kernel_compute_redundant_bounds_row_min_keys!(
            row_min_key,
            drop_lower,
            drop_upper,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.AT.rowPtr,
            Int32(m),
        )

        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_select_redundant_bounds_batch!(
            selected_lower,
            selected_upper,
            drop_lower,
            drop_upper,
            row_min_key,
            lp.AT.rowPtr,
            lp.AT.colVal,
            Int32(n),
        )

        selected_any = Int(sum(Int32.(selected_lower))) > 0 ||
                       Int(sum(Int32.(selected_upper))) > 0
        selected_any || break

        if pparams.record_postsolve_tape
            change_mask = UInt8.((selected_lower .!= UInt8(0)) .| (selected_upper .!= UInt8(0)))
            _, changed_cols, changed_count = build_maps_from_mask(change_mask)
            if Int(changed_count) > 0
                changed_cols_sel = changed_cols
                old_l_sel = gather_by_red2org(plan.new_l, changed_cols_sel)
                old_u_sel = gather_by_red2org(plan.new_u, changed_cols_sel)
                lower_sel = gather_by_red2org(selected_lower, changed_cols_sel)
                upper_sel = gather_by_red2org(selected_upper, changed_cols_sel)
                support_rows = CUDA.fill(typemax(Int32), Int(changed_count))
                new_l_sel = ifelse.(lower_sel .!= UInt8(0), -Inf, old_l_sel)
                new_u_sel = ifelse.(upper_sel .!= UInt8(0), Inf, old_u_sel)

                append_bound_change_records_gpu!(
                    plan.tape_gpu,
                    changed_cols_sel,
                    old_l_sel,
                    old_u_sel,
                    new_l_sel,
                    new_u_sel,
                    lower_sel,
                    upper_sel,
                    support_rows,
                    support_rows;
                    dual_mode=POSTSOLVE_DUAL_MINIMAL,
                )

                if pparams.record_postsolve_tape_cpu
                    for t in 1:Int(changed_count)
                        col = CUDA.@allowscalar Int(changed_cols[t])
                        old_l = CUDA.@allowscalar(Float64(plan.new_l[col]))
                        old_u = CUDA.@allowscalar(Float64(plan.new_u[col]))

                        if CUDA.@allowscalar(selected_lower[col] != UInt8(0))
                            append_bound_change_no_row_record!(
                                plan.tape,
                                col,
                                old_l,
                                old_u,
                                -Inf,
                                old_u;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        elseif CUDA.@allowscalar(selected_upper[col] != UInt8(0))
                            append_bound_change_no_row_record!(
                                plan.tape,
                                col,
                                old_l,
                                old_u,
                                old_l,
                                Inf;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        end
                    end
                end
            end
        end

        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_apply_selected_redundant_bounds!(
            plan.new_l,
            plan.new_u,
            selected_lower,
            selected_upper,
            Int32(n),
        )

        changed_any = true
    end

    if changed_any
        plan.has_col_action = true
        plan.has_change = true
    end

    return nothing
end
