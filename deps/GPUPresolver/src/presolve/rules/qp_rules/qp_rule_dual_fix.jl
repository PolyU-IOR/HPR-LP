"""
QP layer-2 rule: separable dual fixing.

This rule fixes variables only when `Q_{Rj}=0`, so the one-dimensional QP is
independent of all remaining live variables.
"""

@inline function _qp_dual_fix_phi(qjj::Float64, cj::Float64, x::Float64)
    return 0.5 * qjj * x * x + cj * x
end

function _kernel_qp_dual_fix_candidates!(
    status_flag,
    fixed_mask,
    fixed_val,
    obj_contrib,
    keep_col_mask,
    keep_row_mask,
    q_offdiag_nnz,
    c,
    q_diag,
    l,
    u,
    AL,
    AU,
    at_row_ptr,
    at_col_val,
    at_nz_val,
    zero_tol,
    bound_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n && keep_col_mask[j] != UInt8(0)
        if q_offdiag_nnz[j] != Int32(0)
            @inbounds fixed_mask[j] = UInt8(0)
            @inbounds fixed_val[j] = 0.0
            @inbounds obj_contrib[j] = 0.0
            return
        end

        @inbounds lj = l[j]
        @inbounds uj = u[j]
        if lj > uj + bound_tol
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        has_down_lock = false
        has_up_lock = false
        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - Int32(1)
        if p_start <= p_stop
            for p in p_start:p_stop
                @inbounds row = at_col_val[p]
                if keep_row_mask[row] == UInt8(0)
                    continue
                end

                @inbounds aij = at_nz_val[p]
                if aij > zero_tol
                    has_down_lock |= isfinite(AL[row])
                    has_up_lock |= isfinite(AU[row])
                elseif aij < -zero_tol
                    has_down_lock |= isfinite(AU[row])
                    has_up_lock |= isfinite(AL[row])
                end

                if has_down_lock && has_up_lock
                    break
                end
            end
        end

        if has_down_lock && has_up_lock
            @inbounds fixed_mask[j] = UInt8(0)
            @inbounds fixed_val[j] = 0.0
            @inbounds obj_contrib[j] = 0.0
            return
        end

        @inbounds qjj = q_diag[j]
        @inbounds cj = c[j]
        fixed = false
        vj = 0.0

        if qjj > zero_tol
            t_star = -cj / qjj
            if !has_down_lock && isfinite(lj) && t_star <= lj + bound_tol
                vj = lj
                fixed = true
            elseif !has_up_lock && isfinite(uj) && t_star >= uj - bound_tol
                vj = uj
                fixed = true
            end
        elseif qjj < -zero_tol
            if !has_down_lock && !isfinite(lj)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                return
            end
            if !has_up_lock && !isfinite(uj)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                return
            end

            if isfinite(lj) && isfinite(uj)
                phi_l = _qp_dual_fix_phi(qjj, cj, lj)
                phi_u = _qp_dual_fix_phi(qjj, cj, uj)
                if !has_down_lock && phi_l <= phi_u + bound_tol
                    vj = lj
                    fixed = true
                elseif !has_up_lock && phi_u <= phi_l + bound_tol
                    vj = uj
                    fixed = true
                end
            end
        else
            if cj > zero_tol
                if !has_down_lock
                    if isfinite(lj)
                        vj = lj
                        fixed = true
                    else
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                        return
                    end
                end
            elseif cj < -zero_tol
                if !has_up_lock
                    if isfinite(uj)
                        vj = uj
                        fixed = true
                    else
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                        return
                    end
                end
            else
                if !has_down_lock && isfinite(lj)
                    vj = lj
                    fixed = true
                elseif !has_up_lock && isfinite(uj)
                    vj = uj
                    fixed = true
                end
            end
        end

        if fixed
            @inbounds fixed_mask[j] = UInt8(1)
            @inbounds fixed_val[j] = vj
            @inbounds obj_contrib[j] = cj * vj + 0.5 * qjj * vj * vj
        else
            @inbounds fixed_mask[j] = UInt8(0)
            @inbounds fixed_val[j] = 0.0
            @inbounds obj_contrib[j] = 0.0
        end
    end
    return
end

function _kernel_qp_dual_fix_row_shift!(
    row_shift,
    fixed_mask,
    fixed_val,
    keep_row_mask,
    at_row_ptr,
    at_col_val,
    at_nz_val,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n && fixed_mask[j] != UInt8(0)
        @inbounds vj = fixed_val[j]
        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - Int32(1)
        if p_start <= p_stop
            for p in p_start:p_stop
                @inbounds row = at_col_val[p]
                if keep_row_mask[row] != UInt8(0)
                    @inbounds shift = at_nz_val[p] * vj
                    CUDA.@atomic row_shift[row] += shift
                end
            end
        end
    end
    return
end

function apply_rule_qp_dual_fix!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    n = length(plan.keep_col_mask)
    m = length(plan.keep_row_mask)
    n == 0 && return nothing

    status_flag = CUDA.zeros(Int32, 1)
    fixed_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    obj_contrib = CUDA.zeros(Float64, n)

    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_dual_fix_candidates!(
        status_flag,
        fixed_mask,
        fixed_val,
        obj_contrib,
        plan.keep_col_mask,
        plan.keep_row_mask,
        stats.q_offdiag_nnz,
        plan.new_c,
        plan.new_q_diag,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
        qp.AT.rowPtr,
        qp.AT.colVal,
        qp.AT.nzVal,
        pparams.zero_tol,
        pparams.bound_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status == 1
        _mark_infeasible_qp_dual_fix!(plan, "QP dual-fix infeasibility: some separable live column has l > u.")
        return nothing
    elseif status == 2
        _mark_unbounded_qp_dual_fix!(plan, "QP dual-fix unboundedness: separable column has an improving unlocked unbounded direction.")
        return nothing
    end

    append_plan_fixed_from_mask!(plan, fixed_mask, fixed_val) || return nothing

    row_shift = CUDA.zeros(Float64, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_dual_fix_row_shift!(
        row_shift,
        fixed_mask,
        fixed_val,
        plan.keep_row_mask,
        qp.AT.rowPtr,
        qp.AT.colVal,
        qp.AT.nzVal,
        Int32(n),
    )

    plan.new_AL .-= row_shift
    plan.new_AU .-= row_shift
    copyto!(plan.keep_col_mask, UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(fixed_mask .!= UInt8(0))))
    plan.obj_constant_delta += sum(obj_contrib)

    plan.tape_gpu = build_fixed_col_records_gpu(
        plan.fixed_idx,
        plan.fixed_val,
        gather_by_red2org(plan.new_c, plan.fixed_idx),
        qp.AT.rowPtr,
        qp.AT.colVal,
        qp.AT.nzVal,
        plan.keep_row_mask;
        use_keep_row_mask=true,
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if pparams.record_postsolve_tape_cpu
        plan.tape = PostsolveTape(plan.tape_gpu)
    end
    plan.has_change = true
    return nothing
end
