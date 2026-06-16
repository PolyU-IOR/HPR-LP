"""
QP layer-2 rule: separable empty columns.

Rules read `qp` and `stats`, and only update `plan`.
"""

@inline function _mark_infeasible_qp_empty_cols!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _mark_unbounded_qp_empty_cols!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

@inline function _mark_infeasible_qp_dual_fix!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _mark_unbounded_qp_dual_fix!(plan::QPresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

function _kernel_qp_empty_col_candidates!(
    status_flag,
    fixed_mask,
    fixed_val,
    obj_contrib,
    keep_col_mask,
    col_nnz,
    q_offdiag_nnz,
    c,
    q_diag,
    l,
    u,
    zero_tol,
    bound_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n && keep_col_mask[j] != UInt8(0)
        if col_nnz[j] != Int32(0) || q_offdiag_nnz[j] != Int32(0)
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

        @inbounds qjj = q_diag[j]
        @inbounds cj = c[j]
        vj = 0.0

        if qjj > zero_tol
            vj = cj < 0.0 ? -cj / qjj : -cj / qjj
            if isfinite(lj) && vj < lj
                vj = lj
            end
            if isfinite(uj) && vj > uj
                vj = uj
            end
        elseif qjj < -zero_tol
            if !isfinite(lj) || !isfinite(uj)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                return
            end
            phi_l = 0.5 * qjj * lj * lj + cj * lj
            phi_u = 0.5 * qjj * uj * uj + cj * uj
            vj = phi_l <= phi_u ? lj : uj
        else
            if cj > zero_tol
                isfinite(lj) || begin
                    CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                    return
                end
                vj = lj
            elseif cj < -zero_tol
                isfinite(uj) || begin
                    CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(2))
                    return
                end
                vj = uj
            else
                if isfinite(lj)
                    vj = lj
                elseif isfinite(uj)
                    vj = uj
                else
                    vj = 0.0
                end
            end
        end

        @inbounds fixed_mask[j] = UInt8(1)
        @inbounds fixed_val[j] = vj
        @inbounds obj_contrib[j] = cj * vj + 0.5 * qjj * vj * vj
    end
    return
end

function apply_rule_qp_empty_cols!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    n = length(plan.keep_col_mask)
    n == 0 && return nothing

    status_flag = CUDA.zeros(Int32, 1)
    fixed_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    obj_contrib = CUDA.zeros(Float64, n)

    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_empty_col_candidates!(
        status_flag,
        fixed_mask,
        fixed_val,
        obj_contrib,
        plan.keep_col_mask,
        stats.col_nnz,
        stats.q_offdiag_nnz,
        plan.new_c,
        plan.new_q_diag,
        plan.new_l,
        plan.new_u,
        pparams.zero_tol,
        pparams.bound_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status == 1
        _mark_infeasible_qp_empty_cols!(plan, "QP empty-column infeasibility: some separable empty column has l > u.")
        return nothing
    elseif status == 2
        _mark_unbounded_qp_empty_cols!(plan, "QP empty-column unboundedness: separable empty column has an improving unbounded direction.")
        return nothing
    end

    append_plan_fixed_from_mask!(plan, fixed_mask, fixed_val) || return nothing

    copyto!(plan.keep_col_mask, UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(fixed_mask .!= UInt8(0))))
    plan.obj_constant_delta += sum(obj_contrib)

    keep_row = CUDA.fill(UInt8(1), length(plan.keep_row_mask))
    plan.tape_gpu = build_fixed_col_records_gpu(
        plan.fixed_idx,
        plan.fixed_val,
        gather_by_red2org(plan.new_c, plan.fixed_idx),
        qp.AT.rowPtr,
        qp.AT.colVal,
        qp.AT.nzVal,
        keep_row;
        use_keep_row_mask=true,
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if pparams.record_postsolve_tape_cpu
        plan.tape = PostsolveTape(plan.tape_gpu)
    end
    plan.has_change = true
    return nothing
end
