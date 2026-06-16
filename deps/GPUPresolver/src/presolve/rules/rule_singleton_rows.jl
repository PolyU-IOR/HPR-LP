"""
Layer-2 rule: singleton rows.

Rules read `lp` and `stats`, and only update `plan`.
"""

@inline function _mark_infeasible_singleton_rows!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

function _kernel_singleton_row_bounds!(
    candidate_l,
    candidate_u,
    row_remove,
    keep_row,
    singleton_row_mask,
    support_col,
    support_val,
    AL,
    AU,
    n,
    zero_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m &&
       keep_row[i] != UInt8(0) &&
       singleton_row_mask[i] != UInt8(0)
        @inbounds col = support_col[i]
        @inbounds a = support_val[i]
        if col >= Int32(1) && col <= n && abs(a) > zero_tol
            @inbounds lower = AL[i]
            @inbounds upper = AU[i]
            implied_l = a > 0.0 ? (lower / a) : (upper / a)
            implied_u = a > 0.0 ? (upper / a) : (lower / a)
            CUDA.@atomic candidate_l[col] = max(candidate_l[col], implied_l)
            CUDA.@atomic candidate_u[col] = min(candidate_u[col], implied_u)
            @inbounds row_remove[i] = UInt8(1)
        end
    end
    return
end

"""
Rule: singleton rows induce variable-bound tightenings and can be removed.
"""
function apply_rule_singleton_rows!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    m = length(plan.keep_row_mask)
    tol = pparams.bound_tol
    zero_tol = pparams.zero_tol
    if m == 0
        return nothing
    end
    candidate_l = copy(plan.new_l)
    candidate_u = copy(plan.new_u)
    row_remove = CUDA.zeros(UInt8, m)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_row_bounds!(
        candidate_l,
        candidate_u,
        row_remove,
        plan.keep_row_mask,
        stats.singleton_row_mask,
        stats.singleton_row_col,
        stats.singleton_row_val,
        plan.new_AL,
        plan.new_AU,
        Int32(length(plan.new_l)),
        zero_tol,
        Int32(m),
    )

    if any(candidate_l .> (candidate_u .+ tol))
        _mark_infeasible_singleton_rows!(
            plan,
            "Singleton-row infeasibility: tightened lower bound exceeds upper bound.",
        )
        return nothing
    end

    row_removed_any = Int(sum(Int32.(row_remove))) > 0
    bound_changed_any = any(candidate_l .!= plan.new_l) || any(candidate_u .!= plan.new_u)
    if pparams.record_postsolve_tape && row_removed_any
        _, removed_rows, removed_count = build_maps_from_mask(row_remove)
        if Int(removed_count) > 0
            removed_al = gather_by_red2org(plan.new_AL, removed_rows)
            removed_au = gather_by_red2org(plan.new_AU, removed_rows)
            removed_col = gather_by_red2org(stats.singleton_row_col, removed_rows)
            removed_val = gather_by_red2org(stats.singleton_row_val, removed_rows)

            append_deleted_singleton_row_records_gpu!(
                plan.tape_gpu,
                removed_rows,
                removed_col,
                removed_val,
                removed_al,
                removed_au;
                dual_mode=POSTSOLVE_DUAL_MINIMAL,
            )

            if pparams.record_postsolve_tape_cpu
                removed_rows_h = _copy_vector_to_host(removed_rows)
                removed_al_h = _copy_vector_to_host(removed_al)
                removed_au_h = _copy_vector_to_host(removed_au)
                removed_col_h = _copy_vector_to_host(removed_col)
                removed_val_h = _copy_vector_to_host(removed_val)

                for t in eachindex(removed_rows_h)
                    append_deleted_singleton_row_record!(
                        plan.tape,
                        removed_rows_h[t],
                        removed_col_h[t],
                        removed_val_h[t],
                        removed_al_h[t],
                        removed_au_h[t];
                        dual_mode=POSTSOLVE_DUAL_MINIMAL,
                    )
                end
            end
        end
    end
    if row_removed_any
        copyto!(plan.keep_row_mask, UInt8.((plan.keep_row_mask .!= UInt8(0)) .& .!(row_remove .!= UInt8(0))))
    end
    if bound_changed_any
        copyto!(plan.new_l, candidate_l)
        copyto!(plan.new_u, candidate_u)
    end

    if row_removed_any || bound_changed_any
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
