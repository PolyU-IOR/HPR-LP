"""
Layer-2 rule: empty rows.

Rules read `lp` and `stats`, and only update `plan`.
"""

@inline function _mark_infeasible_empty_rows!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

function _kernel_empty_rows_classify!(
    status_flags,
    keep_row,
    AL,
    AU,
    row_nnz,
    tol,
    m::Int32,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        @inbounds live = keep_row[row] != UInt8(0)
        @inbounds is_empty = row_nnz[row] == Int32(0)
        if live && is_empty
            @inbounds al = AL[row]
            @inbounds au = AU[row]
            if al <= tol && au >= -tol
                CUDA.@atomic status_flags[2] = max(status_flags[2], Int32(1))
            else
                CUDA.@atomic status_flags[1] = max(status_flags[1], Int32(1))
            end
        end
    end
    return
end

function _kernel_empty_rows_apply!(
    keep_row,
    AL,
    AU,
    row_nnz,
    tol,
    m::Int32,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= m
        @inbounds live = keep_row[row] != UInt8(0)
        @inbounds is_empty = row_nnz[row] == Int32(0)
        if live && is_empty
            @inbounds al = AL[row]
            @inbounds au = AU[row]
            if al <= tol && au >= -tol
                @inbounds keep_row[row] = UInt8(0)
            end
        end
    end
    return
end

"""
Rule: remove empty rows if `AL <= 0 <= AU`; otherwise mark infeasible.
"""
function apply_rule_empty_rows!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    tol = pparams.feasibility_tol
    m = length(plan.keep_row_mask)
    m == 0 && return nothing

    status_flags = CUDA.zeros(Int32, 2)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_empty_rows_classify!(
        status_flags,
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        stats.row_nnz,
        tol,
        Int32(m),
    )
    flags = Array(status_flags)

    if flags[1] != 0
        _mark_infeasible_empty_rows!(
            plan,
            "Empty-row infeasibility: some empty row requires AL <= 0 <= AU.",
        )
        return nothing
    end

    removed_any = flags[2] != 0
    if removed_any
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_empty_rows_apply!(
            plan.keep_row_mask,
            plan.new_AL,
            plan.new_AU,
            stats.row_nnz,
            tol,
            Int32(m),
        )
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
