"""
Layer-2 rule: simple dual fix.

Fix sign-driven columns to an improving finite bound when the corresponding
direction is unlocked by all live rows.
"""

@inline function _mark_infeasible_dual_fix!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _mark_unbounded_dual_fix!(plan::PresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

function _kernel_dual_fix_candidates!(
    status_flag,
    fixed_mask,
    infinite_fix_mask,
    fixed_val,
    obj_contrib,
    keep_col,
    keep_row,
    c,
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
    if j <= n && keep_col[j] != UInt8(0)
        @inbounds lj = l[j]
        @inbounds uj = u[j]
        if lj > uj + bound_tol
            CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
            return
        end

        has_down_lock = false
        has_up_lock = false

        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - 1
        if p_start <= p_stop
            for p in p_start:p_stop
                @inbounds row = at_col_val[p]
                if keep_row[row] == UInt8(0)
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

        @inbounds cj = c[j]
        fixed = false
        infinite_fix = false
        vj = 0.0

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
        elseif abs(cj) <= zero_tol
            if !has_down_lock
                fixed = true
                if isfinite(lj)
                    vj = lj
                else
                    vj = -Inf
                    infinite_fix = true
                end
            elseif !has_up_lock
                fixed = true
                if isfinite(uj)
                    vj = uj
                else
                    vj = Inf
                    infinite_fix = true
                end
            end
        end

        if fixed
            @inbounds fixed_mask[j] = UInt8(1)
            @inbounds infinite_fix_mask[j] = infinite_fix ? UInt8(1) : UInt8(0)
            @inbounds fixed_val[j] = vj
            @inbounds obj_contrib[j] = infinite_fix ? 0.0 : (cj * vj)
        else
            @inbounds fixed_mask[j] = UInt8(0)
            @inbounds infinite_fix_mask[j] = UInt8(0)
            @inbounds fixed_val[j] = 0.0
            @inbounds obj_contrib[j] = 0.0
        end
    end
    return
end

function _kernel_dual_fix_row_shift!(
    row_shift,
    fixed_mask,
    infinite_fix_mask,
    fixed_val,
    keep_row,
    at_row_ptr,
    at_col_val,
    at_nz_val,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n &&
       fixed_mask[j] != UInt8(0) &&
       infinite_fix_mask[j] == UInt8(0)
        @inbounds vj = fixed_val[j]
        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - 1
        if p_start <= p_stop
            for p in p_start:p_stop
                @inbounds row = at_col_val[p]
                if keep_row[row] != UInt8(0)
                    @inbounds shift = at_nz_val[p] * vj
                    CUDA.@atomic row_shift[row] += shift
                end
            end
        end
    end
    return
end

function _kernel_dual_fix_row_remove!(
    row_remove,
    fixed_mask,
    infinite_fix_mask,
    keep_row,
    at_row_ptr,
    at_col_val,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n &&
       fixed_mask[j] != UInt8(0) &&
       infinite_fix_mask[j] != UInt8(0)
        @inbounds p_start = at_row_ptr[j]
        @inbounds p_stop = at_row_ptr[j + 1] - 1
        if p_start <= p_stop
            for p in p_start:p_stop
                @inbounds row = at_col_val[p]
                if keep_row[row] != UInt8(0)
                    @inbounds row_remove[row] = UInt8(1)
                end
            end
        end
    end
    return
end

function apply_rule_dual_fix!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    _stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    n = length(plan.keep_col_mask)
    m = length(plan.keep_row_mask)
    if n == 0
        return nothing
    end

    status_flag = CUDA.zeros(Int32, 1)
    fixed_mask = CUDA.zeros(UInt8, n)
    infinite_fix_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    obj_contrib = CUDA.zeros(Float64, n)

    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_dual_fix_candidates!(
        status_flag,
        fixed_mask,
        infinite_fix_mask,
        fixed_val,
        obj_contrib,
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        pparams.zero_tol,
        pparams.bound_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status == 1
        _mark_infeasible_dual_fix!(plan, "Dual-fix infeasibility: some live column has l > u.")
        return nothing
    elseif status == 2
        _mark_unbounded_dual_fix!(plan, "Dual-fix unboundedness: some improving column has no finite improving bound.")
        return nothing
    end

    fixed_count = Int(sum(Int32.(fixed_mask .!= UInt8(0))))
    if fixed_count == 0
        return nothing
    end

    finite_fixed_mask = UInt8.((fixed_mask .!= UInt8(0)) .& (infinite_fix_mask .== UInt8(0)))
    row_shift = CUDA.zeros(Float64, m)
    row_remove = CUDA.zeros(UInt8, m)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_dual_fix_row_shift!(
        row_shift,
        fixed_mask,
        infinite_fix_mask,
        fixed_val,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        Int32(n),
    )

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_dual_fix_row_remove!(
        row_remove,
        fixed_mask,
        infinite_fix_mask,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        Int32(n),
    )

    row_changed = Int(sum(Int32.(row_remove .!= UInt8(0)))) > 0
    keep_row_after = if row_changed
        UInt8.((plan.keep_row_mask .!= UInt8(0)) .& .!(row_remove .!= UInt8(0)))
    else
        copy(plan.keep_row_mask)
    end

    append_fixed_col_inf_postsolve_records!(
        plan,
        lp,
        pparams,
        fixed_mask,
        infinite_fix_mask,
        fixed_val,
        plan.keep_row_mask,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
    )

    append_fixed_col_postsolve_records!(
        plan,
        lp,
        pparams,
        finite_fixed_mask,
        fixed_val,
        plan.new_c,
        keep_row_after,
    )

    copyto!(plan.keep_col_mask, UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(fixed_mask .!= UInt8(0))))
    copyto!(plan.new_AL, plan.new_AL .- row_shift)
    copyto!(plan.new_AU, plan.new_AU .- row_shift)
    if row_changed
        copyto!(plan.keep_row_mask, keep_row_after)
    end

    append_plan_fixed_from_mask!(plan, finite_fixed_mask, fixed_val)
    plan.obj_constant_delta += _stable_presolve_masked_product_sum(plan.new_c, fixed_val, finite_fixed_mask)
    plan.has_row_action |= row_changed
    plan.has_col_action = true
    plan.has_change = true
    return nothing
end
