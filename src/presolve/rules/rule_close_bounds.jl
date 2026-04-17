"""
Layer-2 rule: close-bounds columns.

Fix columns whose lower and upper bounds are numerically equal, shift row sides,
and remove the column from the reduced model.
"""

function _kernel_close_bounds_candidates!(
    fixed_mask,
    fixed_val,
    obj_contrib,
    row_shift,
    keep_col,
    l,
    u,
    c,
    at_row_ptr,
    at_col_val,
    at_nz_val,
    bound_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n && keep_col[j] != UInt8(0)
        @inbounds lj = l[j]
        @inbounds uj = u[j]
        if isfinite(lj) && isfinite(uj) && abs(uj - lj) <= bound_tol
            vj = 0.5 * (lj + uj)
            @inbounds fixed_mask[j] = UInt8(1)
            @inbounds fixed_val[j] = vj
            @inbounds obj_contrib[j] = c[j] * vj

            @inbounds p_start = at_row_ptr[j]
            @inbounds p_stop = at_row_ptr[j + 1] - 1
            if p_start <= p_stop
                for p in p_start:p_stop
                    @inbounds row = at_col_val[p]
                    @inbounds shift = at_nz_val[p] * vj
                    CUDA.@atomic row_shift[row] += shift
                end
            end
        else
            @inbounds fixed_mask[j] = UInt8(0)
            @inbounds fixed_val[j] = 0.0
            @inbounds obj_contrib[j] = 0.0
        end
    end
    return
end

function apply_rule_close_bounds!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
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

    fixed_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    obj_contrib = CUDA.zeros(Float64, n)
    row_shift = CUDA.zeros(Float64, m)

    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_close_bounds_candidates!(
        fixed_mask,
        fixed_val,
        obj_contrib,
        row_shift,
        plan.keep_col_mask,
        plan.new_l,
        plan.new_u,
        plan.new_c,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        pparams.bound_tol,
        Int32(n),
    )

    fixed_count = Int(sum(Int32.(fixed_mask .!= UInt8(0))))
    if fixed_count == 0
        return nothing
    end

    append_fixed_col_postsolve_records!(
        plan,
        lp,
        pparams,
        fixed_mask,
        fixed_val,
        plan.new_c,
        plan.keep_row_mask,
    )

    copyto!(plan.keep_col_mask, UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(fixed_mask .!= UInt8(0))))
    copyto!(plan.new_AL, plan.new_AL .- row_shift)
    copyto!(plan.new_AU, plan.new_AU .- row_shift)
    append_plan_fixed_from_mask!(plan, fixed_mask, fixed_val)

    plan.obj_constant_delta += _stable_presolve_masked_product_sum(plan.new_c, fixed_val, fixed_mask)
    plan.has_col_action = true
    plan.has_change = true
    return nothing
end
