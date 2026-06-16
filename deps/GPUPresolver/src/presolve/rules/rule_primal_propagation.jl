"""
Layer-2 rule: primal propagation.

Current scope:
- exact row-implied bounds for logical tests
- significant exact bound tightening
- infeasibility detection from the exact propagated interval
"""

@inline function _mark_infeasible_primal_propagation!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _term_interval_primal_propagation(
    aij::Float64,
    lj::Float64,
    uj::Float64,
)
    if aij >= 0.0
        return (aij * lj, aij * uj)
    end
    return (aij * uj, aij * lj)
end

function _kernel_primal_propagation_candidates!(
    infeasible_flag,
    candidate_l,
    candidate_u,
    row_min_fin,
    row_max_fin,
    row_min_neg_inf_count,
    row_max_pos_inf_count,
    keep_row,
    AL,
    AU,
    l_cur,
    u_cur,
    row_ptr,
    col_val,
    nz_val,
    zero_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && keep_row[i] != UInt8(0)
        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - 1
        if row_start <= row_stop
            @inbounds lower_i = AL[i]
            @inbounds upper_i = AU[i]
            @inbounds row_min_fin_i = row_min_fin[i]
            @inbounds row_max_fin_i = row_max_fin[i]
            @inbounds row_min_neg_inf_count_i = row_min_neg_inf_count[i]
            @inbounds row_max_pos_inf_count_i = row_max_pos_inf_count[i]
            for p in row_start:row_stop
                @inbounds j = col_val[p]
                @inbounds a = nz_val[p]
                if abs(a) <= zero_tol
                    continue
                end

                @inbounds old_l = l_cur[j]
                @inbounds old_u = u_cur[j]
                term_min, term_max = _term_interval_primal_propagation(a, old_l, old_u)

                rest_min = _row_residual_min_from_summary(
                    row_min_fin_i,
                    row_min_neg_inf_count_i,
                    term_min,
                )
                rest_max = _row_residual_max_from_summary(
                    row_max_fin_i,
                    row_max_pos_inf_count_i,
                    term_max,
                )

                implied_l = -Inf
                implied_u = Inf
                if a > 0.0
                    if isfinite(lower_i) && isfinite(rest_max)
                        implied_l = (lower_i - rest_max) / a
                    end
                    if isfinite(upper_i) && isfinite(rest_min)
                        implied_u = (upper_i - rest_min) / a
                    end
                else
                    if isfinite(upper_i) && isfinite(rest_min)
                        implied_l = (upper_i - rest_min) / a
                    end
                    if isfinite(lower_i) && isfinite(rest_max)
                        implied_u = (lower_i - rest_max) / a
                    end
                end

                new_l = old_l
                new_u = old_u

                if isfinite(implied_l)
                    new_l = max(new_l, implied_l)
                end

                if isfinite(implied_u)
                    new_u = min(new_u, implied_u)
                end

                if new_l > old_l
                    CUDA.@atomic candidate_l[j] = max(candidate_l[j], new_l)
                end
                if new_u < old_u
                    CUDA.@atomic candidate_u[j] = min(candidate_u[j], new_u)
                end
            end
        end
    end
    return
end

function _kernel_finalize_primal_propagation!(
    infeasible_flag,
    finalized_l,
    finalized_u,
    lower_changed,
    upper_changed,
    fixed_mask,
    fixed_val,
    candidate_l,
    candidate_u,
    old_l,
    old_u,
    col_max_abs,
    feas_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        @inbounds lj = old_l[j]
        @inbounds uj = old_u[j]
        @inbounds cand_l = candidate_l[j]
        @inbounds cand_u = candidate_u[j]
        @inbounds vmax = col_max_abs[j]

        new_l = lj
        new_u = uj
        lower_changed_j = UInt8(0)
        upper_changed_j = UInt8(0)
        fixed_j = UInt8(0)
        fixed_at = 0.0

        exact_l = lj
        exact_u = uj
        if isfinite(cand_l)
            exact_l = max(exact_l, cand_l)
        end
        if isfinite(cand_u)
            exact_u = min(exact_u, cand_u)
        end
        if exact_l > exact_u + feas_tol
            @inbounds infeasible_flag[1] = UInt8(1)
            return
        end

        if isfinite(cand_l) && cand_l > lj
            if isfinite(uj)
                if cand_l >= uj + feas_tol
                    @inbounds infeasible_flag[1] = UInt8(1)
                    return
                end

                if cand_l >= uj || (uj - cand_l) * vmax <= feas_tol
                    fixed_j = UInt8(1)
                    fixed_at = uj
                    new_l = uj
                    new_u = uj
                end
            end
        end

        if fixed_j == UInt8(0) && isfinite(cand_l) && cand_l > lj
            finite_lb_tightening =
                !isfinite(lj) ||
                ((cand_l - lj > feas_tol * 1.0e4) &&
                 (cand_l - lj > 1.0e-2 * abs(lj)))

            if finite_lb_tightening
                if cand_l != round(cand_l)
                    cand_l -= 0.5 * feas_tol * abs(cand_l)
                end
                new_l = cand_l
                lower_changed_j = UInt8(1)
            end
        end

        if fixed_j == UInt8(0) && isfinite(cand_u) && cand_u < uj
            if isfinite(new_l)
                if cand_u <= new_l - feas_tol
                    @inbounds infeasible_flag[1] = UInt8(1)
                    return
                end

                if cand_u <= new_l || (cand_u - new_l) * vmax <= feas_tol
                    fixed_j = UInt8(1)
                    fixed_at = new_l
                    new_u = new_l
                end
            end
        end

        if fixed_j == UInt8(0) && isfinite(cand_u) && cand_u < uj
            finite_ub_tightening =
                !isfinite(uj) ||
                ((uj - cand_u > feas_tol * 1.0e4) &&
                 (uj - cand_u > 1.0e-2 * abs(uj)))

            if finite_ub_tightening
                if cand_u != round(cand_u)
                    cand_u += 0.5 * feas_tol * abs(cand_u)
                end
                new_u = cand_u
                upper_changed_j = UInt8(1)
            end
        end

        if fixed_j != UInt8(0)
            lower_changed_j = UInt8(0)
            upper_changed_j = UInt8(0)
        end

        @inbounds finalized_l[j] = new_l
        @inbounds finalized_u[j] = new_u
        @inbounds lower_changed[j] = lower_changed_j
        @inbounds upper_changed[j] = upper_changed_j
        @inbounds fixed_mask[j] = fixed_j
        @inbounds fixed_val[j] = fixed_at
    end
    return
end

function _kernel_capture_primal_propagation_support_rows!(
    support_l_row,
    support_u_row,
    candidate_l,
    candidate_u,
    row_min_fin,
    row_max_fin,
    row_min_neg_inf_count,
    row_max_pos_inf_count,
    keep_row,
    AL,
    AU,
    l_cur,
    u_cur,
    row_ptr,
    col_val,
    nz_val,
    zero_tol,
    tol,
    support_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && keep_row[i] != UInt8(0)
        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - 1
        if row_start <= row_stop
            @inbounds lower_i = AL[i]
            @inbounds upper_i = AU[i]
            @inbounds row_min_fin_i = row_min_fin[i]
            @inbounds row_max_fin_i = row_max_fin[i]
            @inbounds row_min_neg_inf_count_i = row_min_neg_inf_count[i]
            @inbounds row_max_pos_inf_count_i = row_max_pos_inf_count[i]
            for p in row_start:row_stop
                @inbounds j = col_val[p]
                @inbounds a = nz_val[p]
                if abs(a) <= zero_tol
                    continue
                end

                @inbounds old_l = l_cur[j]
                @inbounds old_u = u_cur[j]
                term_min, term_max = _term_interval_primal_propagation(a, old_l, old_u)

                rest_min = _row_residual_min_from_summary(
                    row_min_fin_i,
                    row_min_neg_inf_count_i,
                    term_min,
                )
                rest_max = _row_residual_max_from_summary(
                    row_max_fin_i,
                    row_max_pos_inf_count_i,
                    term_max,
                )

                implied_l = -Inf
                implied_u = Inf
                if a > 0.0
                    if isfinite(lower_i) && isfinite(rest_max)
                        implied_l = (lower_i - rest_max) / a
                    end
                    if isfinite(upper_i) && isfinite(rest_min)
                        implied_u = (upper_i - rest_min) / a
                    end
                else
                    if isfinite(upper_i) && isfinite(rest_min)
                        implied_l = (upper_i - rest_min) / a
                    end
                    if isfinite(lower_i) && isfinite(rest_max)
                        implied_u = (lower_i - rest_max) / a
                    end
                end

                @inbounds cand_l = candidate_l[j]
                if cand_l > old_l + tol && isfinite(implied_l) && abs(implied_l - cand_l) <= support_tol
                    CUDA.@atomic support_l_row[j] = min(support_l_row[j], Int32(i))
                end

                @inbounds cand_u = candidate_u[j]
                if cand_u < old_u - tol && isfinite(implied_u) && abs(implied_u - cand_u) <= support_tol
                    CUDA.@atomic support_u_row[j] = min(support_u_row[j], Int32(i))
                end
            end
        end
    end
    return
end

"""
Rule: tighten bounds from row activity intervals.
"""
function apply_rule_primal_propagation!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    m = length(plan.keep_row_mask)
    if m == 0
        return nothing
    end
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    profile_rule = _gpu_presolver_env_enabled("GPUPRESOLVER_PRIMAL_PROPAGATION_PROFILE")

    candidate_l = copy(plan.new_l)
    candidate_u = copy(plan.new_u)
    finalized_l = copy(plan.new_l)
    finalized_u = copy(plan.new_u)
    row_eligible = UInt8.((plan.keep_row_mask .!= UInt8(0)) .& (stats.row_nnz .> Int32(1)))
    row_min_fin = similar(plan.new_AL)
    row_max_fin = similar(plan.new_AU)
    row_min_neg_inf_count = CUDA.zeros(Int32, m)
    row_max_pos_inf_count = CUDA.zeros(Int32, m)
    infeasible_flag = CUDA.zeros(UInt8, 1)

    t_stage = time()
    compute_row_activity_summary!(
        row_min_fin,
        row_max_fin,
        row_min_neg_inf_count,
        row_max_pos_inf_count,
        lp.A,
        plan.new_l,
        plan.new_u,
        pparams.zero_tol,
    )
    if profile_rule
        CUDA.synchronize()
        println(">>> [primal_propagation] row activity summary = ", round(time() - t_stage; digits=4), "s")
    end

    t_stage = time()
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_primal_propagation_candidates!(
        infeasible_flag,
        candidate_l,
        candidate_u,
        row_min_fin,
        row_max_fin,
        row_min_neg_inf_count,
        row_max_pos_inf_count,
        row_eligible,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.zero_tol,
        Int32(m),
    )
    if profile_rule
        CUDA.synchronize()
        println(">>> [primal_propagation] candidates kernel = ", round(time() - t_stage; digits=4), "s")
    end

    n = length(plan.new_l)
    col_max_abs = CUDA.zeros(Float64, n)
    t_stage = time()
    compute_col_max_abs!(col_max_abs, lp.AT)
    if profile_rule
        CUDA.synchronize()
        println(">>> [primal_propagation] col max abs = ", round(time() - t_stage; digits=4), "s")
    end
    lower_changed = CUDA.zeros(UInt8, n)
    upper_changed = CUDA.zeros(UInt8, n)
    fixed_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    col_blocks = cld(n, GPU_PRESOLVE_THREADS)
    t_stage = time()
    @cuda threads=GPU_PRESOLVE_THREADS blocks=col_blocks _kernel_finalize_primal_propagation!(
        infeasible_flag,
        finalized_l,
        finalized_u,
        lower_changed,
        upper_changed,
        fixed_mask,
        fixed_val,
        candidate_l,
        candidate_u,
        plan.new_l,
        plan.new_u,
        col_max_abs,
        pparams.feasibility_tol,
        Int32(n),
    )
    if profile_rule
        CUDA.synchronize()
        println(">>> [primal_propagation] finalize kernel = ", round(time() - t_stage; digits=4), "s")
    end

    infeasible = CUDA.@allowscalar Bool(infeasible_flag[1] != UInt8(0))
    if infeasible
        _mark_infeasible_primal_propagation!(
            plan,
            "Primal-propagation infeasibility: tightened lower bound exceeds upper bound.",
        )
        return nothing
    end

    finite_fixed_mask = UInt8.(fixed_mask .!= UInt8(0))
    fixed_count = Int(sum(Int32.(finite_fixed_mask)))
    lower_changed_nonfixed = UInt8.((lower_changed .!= UInt8(0)) .& (finite_fixed_mask .== UInt8(0)))
    upper_changed_nonfixed = UInt8.((upper_changed .!= UInt8(0)) .& (finite_fixed_mask .== UInt8(0)))
    bound_changed_any =
        Int(sum(Int32.(lower_changed_nonfixed))) > 0 ||
        Int(sum(Int32.(upper_changed_nonfixed))) > 0

    if bound_changed_any && pparams.record_postsolve_tape
        change_mask = UInt8.((lower_changed_nonfixed .!= UInt8(0)) .| (upper_changed_nonfixed .!= UInt8(0)))
        _, changed_cols, changed_count = build_maps_from_mask(change_mask)
        if Int(changed_count) > 0
            support_l_row = CUDA.fill(typemax(Int32), length(plan.new_l))
            support_u_row = CUDA.fill(typemax(Int32), length(plan.new_u))
            support_tol = max(10.0 * pparams.bound_tol, 1.0e-10)
            t_stage = time()
            @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_capture_primal_propagation_support_rows!(
                support_l_row,
                support_u_row,
                finalized_l,
                finalized_u,
                row_min_fin,
                row_max_fin,
                row_min_neg_inf_count,
                row_max_pos_inf_count,
                row_eligible,
                plan.new_AL,
                plan.new_AU,
                plan.new_l,
                plan.new_u,
                lp.A.rowPtr,
                lp.A.colVal,
                lp.A.nzVal,
                pparams.zero_tol,
                pparams.bound_tol,
                support_tol,
                Int32(m),
            )
            if profile_rule
                CUDA.synchronize()
                println(">>> [primal_propagation] support rows kernel = ", round(time() - t_stage; digits=4), "s")
            end

            t_stage = time()
            old_l_sel = gather_by_red2org(plan.new_l, changed_cols)
            old_u_sel = gather_by_red2org(plan.new_u, changed_cols)
            new_l_sel = gather_by_red2org(finalized_l, changed_cols)
            new_u_sel = gather_by_red2org(finalized_u, changed_cols)
            lower_sel = gather_by_red2org(lower_changed_nonfixed, changed_cols)
            upper_sel = gather_by_red2org(upper_changed_nonfixed, changed_cols)
            support_l_sel = gather_by_red2org(support_l_row, changed_cols)
            support_u_sel = gather_by_red2org(support_u_row, changed_cols)
            if profile_rule
                CUDA.synchronize()
                println(">>> [primal_propagation] gather support payload = ", round(time() - t_stage; digits=4), "s")
            end

            t_stage = time()
            append_bound_change_records_gpu!(
                plan.tape_gpu,
                changed_cols,
                old_l_sel,
                old_u_sel,
                new_l_sel,
                new_u_sel,
                lower_sel,
                upper_sel,
                support_l_sel,
                support_u_sel;
                dual_mode=POSTSOLVE_DUAL_MINIMAL,
            )
            if profile_rule
                CUDA.synchronize()
                println(">>> [primal_propagation] append bound-change tape = ", round(time() - t_stage; digits=4), "s")
            end

            if pparams.record_postsolve_tape_cpu
                for t in 1:Int(changed_count)
                    col = CUDA.@allowscalar Int(changed_cols[t])
                    cur_l = CUDA.@allowscalar(Float64(plan.new_l[col]))
                    cur_u = CUDA.@allowscalar(Float64(plan.new_u[col]))

                    if CUDA.@allowscalar(lower_changed_nonfixed[col] != UInt8(0))
                        new_l = CUDA.@allowscalar(Float64(finalized_l[col]))
                        row = CUDA.@allowscalar(Int32(support_l_row[col]))
                        if row != typemax(Int32)
                            append_bound_change_the_row_record!(
                                plan.tape,
                                col,
                                Int(row),
                                cur_l,
                                cur_u,
                                new_l,
                                cur_u;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        else
                            append_bound_change_no_row_record!(
                                plan.tape,
                                col,
                                cur_l,
                                cur_u,
                                new_l,
                                cur_u;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        end
                        cur_l = new_l
                    end

                    if CUDA.@allowscalar(upper_changed_nonfixed[col] != UInt8(0))
                        new_u = CUDA.@allowscalar(Float64(finalized_u[col]))
                        row = CUDA.@allowscalar(Int32(support_u_row[col]))
                        if row != typemax(Int32)
                            append_bound_change_the_row_record!(
                                plan.tape,
                                col,
                                Int(row),
                                cur_l,
                                cur_u,
                                cur_l,
                                new_u;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        else
                            append_bound_change_no_row_record!(
                                plan.tape,
                                col,
                                cur_l,
                                cur_u,
                                cur_l,
                                new_u;
                                dual_mode=POSTSOLVE_DUAL_MINIMAL,
                            )
                        end
                    end
                end
            end
        end
    end

    row_removed_any = false
    if fixed_count > 0
        row_shift = CUDA.zeros(Float64, m)
        t_stage = time()
        @cuda threads=GPU_PRESOLVE_THREADS blocks=col_blocks _kernel_dual_fix_row_shift!(
            row_shift,
            finite_fixed_mask,
            CUDA.zeros(UInt8, n),
            fixed_val,
            plan.keep_row_mask,
            lp.AT.rowPtr,
            lp.AT.colVal,
            lp.AT.nzVal,
            Int32(n),
        )
        if profile_rule
            CUDA.synchronize()
            println(">>> [primal_propagation] row shift kernel = ", round(time() - t_stage; digits=4), "s")
        end

        keep_col_after = UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(finite_fixed_mask .!= UInt8(0)))
        t_stage = time()
        _, col_red2org_after, _ = build_maps_from_mask(keep_col_after)
        A_after_cols = compact_csr_by_cols(lp.A, col_red2org_after)
        row_nnz_after = CUDA.zeros(Int32, m)
        compute_row_nnz!(row_nnz_after, A_after_cols)
        if profile_rule
            CUDA.synchronize()
            println(">>> [primal_propagation] compact cols + row nnz = ", round(time() - t_stage; digits=4), "s")
        end

        t_stage = time()
        AL_after = plan.new_AL .- row_shift
        AU_after = plan.new_AU .- row_shift
        empty_after = (plan.keep_row_mask .!= UInt8(0)) .& (row_nnz_after .== Int32(0))
        feasible_empty = empty_after .& (AL_after .<= pparams.feasibility_tol) .& (AU_after .>= -pparams.feasibility_tol)
        infeasible_empty = empty_after .& .!feasible_empty
        if any(infeasible_empty)
            _mark_infeasible_primal_propagation!(
                plan,
                "Primal-propagation infeasibility: fixing columns created an infeasible empty row.",
            )
            return nothing
        end

        keep_row_after = UInt8.((plan.keep_row_mask .!= UInt8(0)) .& .!(feasible_empty .!= UInt8(0)))
        row_removed_any = Int(sum(Int32.(feasible_empty .!= UInt8(0)))) > 0
        if profile_rule
            CUDA.synchronize()
            println(">>> [primal_propagation] empty-row feasibility checks = ", round(time() - t_stage; digits=4), "s")
        end

        t_stage = time()
        append_fixed_col_postsolve_records!(
            plan,
            lp,
            pparams,
            finite_fixed_mask,
            fixed_val,
            plan.new_c,
            keep_row_after,
        )

        append_plan_fixed_from_mask!(plan, finite_fixed_mask, fixed_val)
        plan.obj_constant_delta += _stable_presolve_masked_product_sum(plan.new_c, fixed_val, finite_fixed_mask)
        copyto!(plan.keep_col_mask, keep_col_after)
        copyto!(plan.keep_row_mask, keep_row_after)
        copyto!(plan.new_AL, AL_after)
        copyto!(plan.new_AU, AU_after)
        if profile_rule
            CUDA.synchronize()
            println(">>> [primal_propagation] apply fixed updates + tape = ", round(time() - t_stage; digits=4), "s")
        end
    end

    if bound_changed_any
        copyto!(plan.new_l, finalized_l)
        copyto!(plan.new_u, finalized_u)
    end

    if bound_changed_any || fixed_count > 0 || row_removed_any
        if !bound_changed_any
            copyto!(plan.new_l, finalized_l)
            copyto!(plan.new_u, finalized_u)
        end
        plan.has_row_action = true
        plan.has_col_action |= fixed_count > 0
        plan.has_change = true
    end

    return nothing
end
