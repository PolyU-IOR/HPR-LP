"""
Layer-2 rule: activity checks.

Current scope:
- detect infeasible rows from reachable row activity bounds
- remove redundant rows whose full reachable interval is already implied
- drop a redundant lower or upper side from a two-sided row

Bound tightening and forcing logic remain for `primal_propagation`.
"""

@inline function _mark_infeasible_activity_checks!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

function _kernel_activity_checks_classify!(
    status_flags,
    full_redundant,
    drop_lower,
    drop_upper,
    keep_row,
    AL,
    AU,
    l,
    u,
    row_nnz,
    row_ptr,
    col_val,
    nz_val,
    tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds full_redundant[i] = UInt8(0)
        @inbounds drop_lower[i] = UInt8(0)
        @inbounds drop_upper[i] = UInt8(0)

        @inbounds keep_i = keep_row[i] != UInt8(0)
        @inbounds row_nnz_i = row_nnz[i]
        if !keep_i || row_nnz_i <= Int32(1)
            return
        end

        @inbounds lower_i = AL[i]
        @inbounds upper_i = AU[i]
        lower_finite = isfinite(lower_i)
        upper_finite = isfinite(upper_i)
        if lower_finite && upper_finite && abs(upper_i - lower_i) <= tol
            return
        end

        row_min = 0.0
        row_max = 0.0
        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - 1
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                @inbounds a = nz_val[p]
                @inbounds lj = l[col]
                @inbounds uj = u[col]
                if a >= 0.0
                    row_min += a * lj
                    row_max += a * uj
                else
                    row_min += a * uj
                    row_max += a * lj
                end
            end
        end

        infeasible =
            (lower_finite && row_max < lower_i - tol) ||
            (upper_finite && row_min > upper_i + tol)
        if infeasible
            @inbounds status_flags[1] = UInt8(1)
            return
        end

        lower_implied = !lower_finite || row_min >= lower_i - tol
        upper_implied = !upper_finite || row_max <= upper_i + tol
        if lower_implied && upper_implied
            @inbounds full_redundant[i] = UInt8(1)
            @inbounds status_flags[2] = UInt8(1)
        elseif lower_finite && lower_implied
            @inbounds drop_lower[i] = UInt8(1)
            @inbounds status_flags[3] = UInt8(1)
        elseif upper_finite && upper_implied
            @inbounds drop_upper[i] = UInt8(1)
            @inbounds status_flags[4] = UInt8(1)
        end
    end
    return
end

function _kernel_activity_checks_apply!(
    keep_row,
    AL,
    AU,
    full_redundant,
    drop_lower,
    drop_upper,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds full_i = full_redundant[i]
        @inbounds drop_lower_i = drop_lower[i]
        @inbounds drop_upper_i = drop_upper[i]
        if full_i != UInt8(0)
            keep_row[i] = UInt8(0)
        end
        if drop_lower_i != UInt8(0)
            AL[i] = -Inf
        end
        if drop_upper_i != UInt8(0)
            AU[i] = Inf
        end
    end
    return
end

"""
Rule: use row activity bounds to detect infeasible or redundant rows.
"""
function apply_rule_activity_checks!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    tol = pparams.bound_tol
    AL = plan.new_AL
    AU = plan.new_AU
    m = length(plan.keep_row_mask)
    m == 0 && return nothing

    status_flags = CUDA.zeros(UInt8, 4)
    full_redundant = CUDA.zeros(UInt8, m)
    drop_lower = CUDA.zeros(UInt8, m)
    drop_upper = CUDA.zeros(UInt8, m)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_activity_checks_classify!(
        status_flags,
        full_redundant,
        drop_lower,
        drop_upper,
        plan.keep_row_mask,
        AL,
        AU,
        plan.new_l,
        plan.new_u,
        stats.row_nnz,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        tol,
        Int32(m),
    )

    flags = Array(status_flags)
    if flags[1] != UInt8(0)
        _mark_infeasible_activity_checks!(
            plan,
            "Activity-check infeasibility: reachable row activity lies outside some live row bound.",
        )
        return nothing
    end

    changed_any =
        flags[2] != UInt8(0) || flags[3] != UInt8(0) || flags[4] != UInt8(0)
    if changed_any
        if pparams.record_postsolve_tape
            if flags[2] != UInt8(0)
                _, removed_rows, removed_count = build_maps_from_mask(full_redundant)
                if Int(removed_count) > 0
                    old_AL = gather_by_red2org(AL, removed_rows)
                    old_AU = gather_by_red2org(AU, removed_rows)
                    append_activity_check_row_records_gpu!(
                        plan.tape_gpu,
                        DELETED_ROW,
                        removed_rows,
                        old_AL,
                        old_AU,
                        old_AL,
                        old_AU;
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                        cpu_tape=pparams.record_postsolve_tape_cpu ? plan.tape : nothing,
                    )
                end
            end

            if flags[3] != UInt8(0)
                _, changed_rows, changed_count = build_maps_from_mask(drop_lower)
                if Int(changed_count) > 0
                    old_AL = gather_by_red2org(AL, changed_rows)
                    old_AU = gather_by_red2org(AU, changed_rows)
                    new_AL = CUDA.fill(-Inf, Int(changed_count))
                    append_activity_check_row_records_gpu!(
                        plan.tape_gpu,
                        LHS_CHANGE,
                        changed_rows,
                        old_AL,
                        old_AU,
                        new_AL,
                        old_AU;
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                        cpu_tape=pparams.record_postsolve_tape_cpu ? plan.tape : nothing,
                    )
                end
            end

            if flags[4] != UInt8(0)
                _, changed_rows, changed_count = build_maps_from_mask(drop_upper)
                if Int(changed_count) > 0
                    old_AL = gather_by_red2org(AL, changed_rows)
                    old_AU = gather_by_red2org(AU, changed_rows)
                    new_AU = CUDA.fill(Inf, Int(changed_count))
                    append_activity_check_row_records_gpu!(
                        plan.tape_gpu,
                        RHS_CHANGE,
                        changed_rows,
                        old_AL,
                        old_AU,
                        old_AL,
                        new_AU;
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                        cpu_tape=pparams.record_postsolve_tape_cpu ? plan.tape : nothing,
                    )
                end
            end
        end

        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_activity_checks_apply!(
            plan.keep_row_mask,
            plan.new_AL,
            plan.new_AU,
            full_redundant,
            drop_lower,
            drop_upper,
            Int32(m),
        )
    end

    if changed_any
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
