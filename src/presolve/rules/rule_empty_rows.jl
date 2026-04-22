"""
Layer-2 rule: empty rows.

Rules read `lp` and `stats`, and only update `plan`.
"""

@inline function _mark_infeasible_empty_rows!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
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
    keep = plan.keep_row_mask .!= UInt8(0)
    empty = stats.empty_row_mask .!= UInt8(0)
    AL = plan.new_AL
    AU = plan.new_AU

    feasible_empty = keep .& empty .& (AL .<= tol) .& (AU .>= -tol)
    infeasible_empty = keep .& empty .& .!feasible_empty

    if any(infeasible_empty)
        _mark_infeasible_empty_rows!(
            plan,
            "Empty-row infeasibility: some empty row requires AL <= 0 <= AU.",
        )
        return nothing
    end

    removed_any = Int(sum(Int32.(feasible_empty))) > 0
    if removed_any
        copyto!(plan.keep_row_mask, UInt8.(keep .& .!feasible_empty))
    end

    if removed_any
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
