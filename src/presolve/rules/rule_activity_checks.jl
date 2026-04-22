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
    keep = plan.keep_row_mask .!= UInt8(0)
    AL = plan.new_AL
    AU = plan.new_AU
    row_nnz = stats.row_nnz
    row_min = similar(AL)
    row_max = similar(AU)
    compute_row_activity_bounds!(row_min, row_max, lp.A, plan.new_l, plan.new_u)

    lower_finite = isfinite.(AL)
    upper_finite = isfinite.(AU)
    equality_rows = lower_finite .& upper_finite .& (abs.(AU .- AL) .<= tol)
    eligible = keep .& .!equality_rows .& (row_nnz .> Int32(1))

    infeasible = eligible .& (
        (lower_finite .& (row_max .< (AL .- tol))) .|
        (upper_finite .& (row_min .> (AU .+ tol)))
    )
    if any(infeasible)
        _mark_infeasible_activity_checks!(
            plan,
            "Activity-check infeasibility: reachable row activity lies outside some live row bound.",
        )
        return nothing
    end

    lower_implied = .!lower_finite .| (row_min .>= (AL .- tol))
    upper_implied = .!upper_finite .| (row_max .<= (AU .+ tol))
    full_redundant = eligible .& lower_implied .& upper_implied
    drop_lower = eligible .& .!full_redundant .& lower_finite .& lower_implied
    drop_upper = eligible .& .!full_redundant .& upper_finite .& upper_implied

    changed_any = Int(sum(Int32.(full_redundant))) > 0 ||
                  Int(sum(Int32.(drop_lower))) > 0 ||
                  Int(sum(Int32.(drop_upper))) > 0
    if changed_any
        if pparams.record_postsolve_tape
            if any(full_redundant)
                _, removed_rows, removed_count = build_maps_from_mask(full_redundant)
                for t in 1:Int(removed_count)
                    row = CUDA.@allowscalar Int(removed_rows[t])
                    append_deleted_row_record!(
                        plan.tape,
                        row,
                        CUDA.@allowscalar(Float64(AL[row])),
                        CUDA.@allowscalar(Float64(AU[row]));
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                    )
                end
            end

            if any(drop_lower)
                _, changed_rows, changed_count = build_maps_from_mask(drop_lower)
                for t in 1:Int(changed_count)
                    row = CUDA.@allowscalar Int(changed_rows[t])
                    old_al = CUDA.@allowscalar(Float64(AL[row]))
                    old_au = CUDA.@allowscalar(Float64(AU[row]))
                    append_lhs_change_record!(
                        plan.tape,
                        row,
                        old_al,
                        old_au,
                        -Inf,
                        old_au;
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                    )
                end
            end

            if any(drop_upper)
                _, changed_rows, changed_count = build_maps_from_mask(drop_upper)
                for t in 1:Int(changed_count)
                    row = CUDA.@allowscalar Int(changed_rows[t])
                    old_al = CUDA.@allowscalar(Float64(AL[row]))
                    old_au = CUDA.@allowscalar(Float64(AU[row]))
                    append_rhs_change_record!(
                        plan.tape,
                        row,
                        old_al,
                        old_au,
                        old_al,
                        Inf;
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                    )
                end
            end
        end

        copyto!(plan.keep_row_mask, UInt8.(keep .& .!full_redundant))
        copyto!(plan.new_AL, ifelse.(drop_lower, -Inf, AL))
        copyto!(plan.new_AU, ifelse.(drop_upper, Inf, AU))
    end

    if changed_any
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
