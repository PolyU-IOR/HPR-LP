"""
Layer-2 rule: empty columns.

Rules read `lp` and `stats`, and only update `plan`.
"""

@inline function _mark_infeasible_empty_cols!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _mark_unbounded_empty_cols!(plan::PresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

"""
Rule: empty columns can be fixed by objective direction and removed.
"""
function apply_rule_empty_cols!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    bound_tol = pparams.bound_tol
    zero_tol = pparams.zero_tol
    keep = plan.keep_col_mask .!= UInt8(0)
    empty = stats.empty_col_mask .!= UInt8(0)
    active = keep .& empty
    c = plan.new_c
    l = plan.new_l
    u = plan.new_u

    infeasible = active .& (l .> u .+ bound_tol)
    if any(infeasible)
        _mark_infeasible_empty_cols!(plan, "Empty-column infeasibility: some empty column has l > u.")
        return nothing
    end

    unbounded = (active .& (c .> zero_tol) .& .!isfinite.(l)) .|
                (active .& (c .< -zero_tol) .& .!isfinite.(u))
    if any(unbounded)
        _mark_unbounded_empty_cols!(plan, "Empty-column unboundedness: objective-improving empty column has no finite improving bound.")
        return nothing
    end

    fixed_mask = UInt8.(active)
    fixed_val = ifelse.(
        c .> zero_tol,
        l,
        ifelse.(
            c .< -zero_tol,
            u,
            ifelse.(isfinite.(l), l, ifelse.(isfinite.(u), u, 0.0)),
        ),
    )

    changed_any = Int(sum(Int32.(fixed_mask .!= UInt8(0)))) > 0
    if changed_any
        delta = _stable_presolve_masked_product_sum(c, fixed_val, fixed_mask)
        append_fixed_col_postsolve_records!(plan, lp, pparams, fixed_mask, fixed_val, c, plan.keep_row_mask)
        copyto!(plan.keep_col_mask, UInt8.(keep .& .!(fixed_mask .!= UInt8(0))))
        append_plan_fixed_from_mask!(plan, fixed_mask, fixed_val)
        plan.obj_constant_delta += delta
    end

    if changed_any
        plan.has_col_action = true
        plan.has_change = true
    end

    return nothing
end
