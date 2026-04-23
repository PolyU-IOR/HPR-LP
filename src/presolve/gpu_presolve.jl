"""
GPU-first presolve pipeline (Layer 1 scheduler).

Architecture:
- Iterative loop
- Fixed phase order: row phase then col phase
- Each phase: stats -> plan -> apply
"""

using CUDA
using CUDA: CuVector

const PRESOLVE_CLEANUP_TRIGGER_RULES = (
    :close_bounds,
    :singleton_cols,
    :doubleton_eq,
    :primal_propagation,
    :dual_fix,
    :parallel_cols,
)
const PRESOLVE_TRIVIAL_ROW_RULES = (:singleton_rows, :empty_rows)
const PRESOLVE_TRIVIAL_COL_PREFIX_RULES = (:close_bounds,)
const PRESOLVE_TRIVIAL_COL_SUFFIX_RULES = (:empty_cols,)
const VALID_ROW_PHASE_RULES = (
    :empty_rows,
    :singleton_rows,
    :activity_checks,
    :primal_propagation,
    :parallel_rows,
)
const VALID_COL_PHASE_RULES = (
    :close_bounds,
    :empty_cols,
    :singleton_cols,
    :doubleton_eq,
    :dual_fix,
    :parallel_cols,
    :redundant_bounds,
)

@inline function _is_rule_enabled(pparams::PresolveParams, rule_name::Symbol)
    if rule_name == :close_bounds
        return pparams.enable_close_bounds
    elseif rule_name == :empty_rows
        return pparams.enable_empty_rows && pparams.enable_remove_empty_rows
    elseif rule_name == :singleton_rows
        return pparams.enable_singleton_rows
    elseif rule_name == :activity_checks
        return pparams.enable_activity_checks
    elseif rule_name == :primal_propagation
        return pparams.enable_primal_propagation
    elseif rule_name == :parallel_rows
        return pparams.enable_parallel_rows
    elseif rule_name == :empty_cols
        return pparams.enable_empty_cols && pparams.enable_remove_empty_cols
    elseif rule_name == :singleton_cols
        return pparams.enable_singleton_cols
    elseif rule_name == :doubleton_eq
        return pparams.enable_doubleton_eq
    elseif rule_name == :dual_fix
        return pparams.enable_dual_fix
    elseif rule_name == :parallel_cols
        return pparams.enable_parallel_cols
    elseif rule_name == :redundant_bounds
        return pparams.enable_redundant_bounds
    end
    return false
end

function _append_cuvector(base::CuVector{T}, extra::CuVector{T}) where {T}
    if length(extra) == 0
        return copy(base)
    end
    if length(base) == 0
        return copy(extra)
    end
    return concat_cuvector(base, extra)
end

@inline function _mask_keeps_all(keep_mask::CuVector{UInt8})
    return isempty(keep_mask) || Int(sum(Int32.(keep_mask .== UInt8(0)))) == 0
end

@inline function _row_phase_has_structural_change(plan::PresolvePlan_gpu)
    return !_mask_keeps_all(plan.keep_row_mask) ||
           !_mask_keeps_all(plan.keep_col_mask) ||
           !isnothing(plan.new_A) ||
           !isnothing(plan.new_AT_leading_slack) ||
           !isnothing(plan.new_AT_slack_after)
end

@inline function _col_phase_has_structural_change(plan::PresolvePlan_gpu)
    return !_mask_keeps_all(plan.keep_row_mask) ||
           !_mask_keeps_all(plan.keep_col_mask) ||
           !isnothing(plan.new_A) ||
           !isnothing(plan.new_AT_leading_slack) ||
           !isnothing(plan.new_AT_slack_after)
end

function _copy_record_with_updates(
    rec::PresolveRecord_gpu;
    m1::Int32=rec.m1,
    n1::Int32=rec.n1,
    obj_constant_new::Float64=rec.obj_constant_new,
    tape::PostsolveTape=rec.tape,
    tape_gpu::Union{Nothing,PostsolveTape_gpu}=rec.tape_gpu,
    tape_gpu_parts::Vector{PostsolveTape_gpu}=rec.tape_gpu_parts,
)
    return PresolveRecord_gpu(
        rec.m0,
        rec.n0,
        m1,
        n1,
        rec.row_org2red,
        rec.row_red2org,
        rec.col_org2red,
        rec.col_red2org,
        rec.fixed_idx,
        rec.fixed_val,
        rec.removed_row_idx,
        rec.removed_col_idx,
        rec.singleton_col_row_idx,
        rec.singleton_col_col_idx,
        rec.merged_col_from,
        rec.merged_col_to,
        rec.merged_col_ratio,
        rec.merged_col_from_l,
        rec.merged_col_from_u,
        rec.merged_col_to_l,
        rec.merged_col_to_u,
        rec.obj_constant_old,
        obj_constant_new,
        copy(rec.rule_counters),
        tape,
        tape_gpu,
        tape_gpu_parts,
    )
end

function _merge_postsolve_tape(
    rec_tape::PostsolveTape,
    plan_tape::PostsolveTape,
    row_red2org,
    col_red2org,
)
    if postsolve_record_count(plan_tape) == 0
        return rec_tape
    end
    append_postsolve_tape!(rec_tape, globalize_postsolve_tape(plan_tape, row_red2org, col_red2org))
    return rec_tape
end

function _merge_postsolve_tape_gpu_parts(
    rec_tape_gpu_parts::Vector{PostsolveTape_gpu},
    plan_tape::PostsolveTape,
    plan_tape_gpu::PostsolveTape_gpu,
    row_red2org::CuVector{Int32},
    col_red2org::CuVector{Int32},
)
    if postsolve_record_count(plan_tape_gpu) == 0 &&
       postsolve_record_count(plan_tape) == 0
        return rec_tape_gpu_parts
    end

    merged = copy(rec_tape_gpu_parts)

    if postsolve_record_count(plan_tape_gpu) > 0
        push!(
            merged,
            globalize_postsolve_tape_gpu(plan_tape_gpu, row_red2org, col_red2org),
        )
    end

    if postsolve_record_count(plan_tape) > 0
        push!(
            merged,
            PostsolveTape_gpu(globalize_postsolve_tape(plan_tape, row_red2org, col_red2org)),
        )
    end

    return merged
end

function _csr_row_lengths_gpu(csr::CuSparseMatrixCSR{T,Int32}) where {T}
    row_ptr = csr.rowPtr
    return row_ptr[2:end] .- row_ptr[1:(end - 1)]
end

function _virtual_starts_from_lengths_and_slack(
    lengths::Vector{Int32},
    leading_slack::Int32,
    slack_after::Vector{Int32},
)
    n = length(lengths)
    starts = Vector{Int}(undef, n)
    pos = Int(leading_slack)
    for i in 1:n
        starts[i] = pos
        pos += Int(lengths[i])
        if i < n
            pos += Int(slack_after[i])
        end
    end
    return starts
end

function _update_virtual_slack_without_shifts(
    old_lengths::Vector{Int32},
    old_leading_slack::Int32,
    old_slack_after::Vector{Int32},
    new_lengths::Vector{Int32},
)
    n = length(old_lengths)
    @assert length(old_slack_after) == n
    @assert length(new_lengths) == n

    starts = _virtual_starts_from_lengths_and_slack(
        old_lengths,
        old_leading_slack,
        old_slack_after,
    )
    new_leading_slack = Int32(isempty(starts) ? 0 : starts[1])
    new_slack_after = zeros(Int32, n)
    for i in 1:(n - 1)
        gap = starts[i + 1] - (starts[i] + Int(new_lengths[i]))
        new_slack_after[i] = Int32(max(gap, 0))
    end
    return (new_leading_slack, new_slack_after)
end

function _update_virtual_slack_without_shifts_gpu(
    old_lengths::CuVector{Int32},
    old_leading_slack::Int32,
    old_slack_after::CuVector{Int32},
    new_lengths::CuVector{Int32},
)
    n = length(old_lengths)
    @assert length(old_slack_after) == n
    @assert length(new_lengths) == n

    n == 0 && return (Int32(0), CuVector{Int32}(undef, 0))

    new_slack_after = CUDA.zeros(Int32, n)
    if n > 1
        @views new_slack_after[1:(n - 1)] .= max.(
            old_lengths[1:(n - 1)] .+
            old_slack_after[1:(n - 1)] .-
            new_lengths[1:(n - 1)],
            Int32(0),
        )
    end
    return (old_leading_slack, new_slack_after)
end

function _compact_virtual_slack_after(
    source_lengths::Vector{Int32},
    source_leading_slack::Int32,
    source_slack_after::Vector{Int32},
    kept_oldidx::CuVector{Int32},
)
    kept = Array(kept_oldidx)
    isempty(kept) && return (Int32(0), Int32[])

    starts = _virtual_starts_from_lengths_and_slack(
        source_lengths,
        source_leading_slack,
        source_slack_after,
    )
    new_leading_slack = Int32(starts[first(kept)])
    new_slack_after = zeros(Int32, length(kept))
    for t in 1:(length(kept) - 1)
        cur = Int(kept[t])
        nxt = Int(kept[t + 1])
        gap = starts[nxt] - (starts[cur] + Int(source_lengths[cur]))
        new_slack_after[t] = Int32(max(gap, 0))
    end
    return (new_leading_slack, new_slack_after)
end

function _compact_virtual_slack_after_gpu(
    source_lengths::CuVector{Int32},
    source_leading_slack::Int32,
    source_slack_after::CuVector{Int32},
    kept_oldidx::CuVector{Int32},
)
    @assert length(source_lengths) == length(source_slack_after)
    isempty(kept_oldidx) && return (Int32(0), CuVector{Int32}(undef, 0))

    stride = source_lengths .+ source_slack_after
    starts = cumsum(stride) .- stride .+ source_leading_slack
    kept_starts = gather_by_red2org(starts, kept_oldidx)
    kept_lengths = gather_by_red2org(source_lengths, kept_oldidx)

    new_leading_slack = Int32(_copy_scalar_to_host(kept_starts, 1))
    new_slack_after = CUDA.zeros(Int32, length(kept_oldidx))
    if length(kept_oldidx) > 1
        @views new_slack_after[1:(end - 1)] .= max.(
            kept_starts[2:end] .-
            (kept_starts[1:(end - 1)] .+ kept_lengths[1:(end - 1)]),
            Int32(0),
        )
    end
    return (new_leading_slack, new_slack_after)
end

function _release_presolve_gpu_temps!()
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

function _collect_removed_global_indices(
    keep_mask_local::CuVector{UInt8},
    old_red2org_global::CuVector{Int32},
)
    @assert length(keep_mask_local) == length(old_red2org_global)
    keep_h = Array(keep_mask_local)
    old_h = Array(old_red2org_global)

    removed = Int32[]
    for i in eachindex(keep_h)
        if keep_h[i] == UInt8(0)
            push!(removed, old_h[i])
        end
    end
    return CuVector(removed)
end

function _collect_removed_global_indices_gpu(
    keep_mask_local::CuVector{UInt8},
    old_red2org_global::CuVector{Int32},
)
    @assert length(keep_mask_local) == length(old_red2org_global)
    removed_mask = UInt8.(keep_mask_local .== UInt8(0))
    _, removed_local, removed_count = build_maps_from_mask(removed_mask)
    removed_count == 0 && return CuVector{Int32}(undef, 0)
    return gather_by_red2org(old_red2org_global, removed_local)
end

function _map_local_to_global_indices(
    local_idx::CuVector{Int32},
    old_red2org_global::CuVector{Int32},
)
    if length(local_idx) == 0
        return CuVector{Int32}(undef, 0)
    end
    return gather_by_red2org(old_red2org_global, local_idx)
end

function _throw_terminal_status_if_needed!(plan::PresolvePlan_gpu, phase::Symbol)
    if plan.has_infeasible
        error("GPU presolve $(phase)-phase detected INFEASIBLE: $(plan.status_message)")
    end
    if plan.has_unbounded
        error("GPU presolve $(phase)-phase detected UNBOUNDED: $(plan.status_message)")
    end
    return nothing
end

@inline function _needs_cleanup_recirculation(rule_name::Symbol)
    return rule_name in PRESOLVE_CLEANUP_TRIGGER_RULES
end

@inline function _phase_has_action(plan::PresolvePlan_gpu, phase::Symbol)
    if phase == :row
        return plan.has_row_action || plan.has_col_action
    elseif phase == :col
        return plan.has_col_action
    end
    error("Unknown presolve phase: $phase")
end

@inline function _phase_rule_order(pparams::PresolveParams, phase::Symbol)
    if phase == :row
        return pparams.row_rule_order
    elseif phase == :col
        return pparams.col_rule_order
    end
    error("Unknown presolve phase: $phase")
end

function _validate_phase_rule_order(rule_order, valid_rules, phase::Symbol)
    for rule_name in rule_order
        rule_name in valid_rules || error("Unknown $(phase)-phase rule symbol: $rule_name")
    end
    return nothing
end

function _validate_presolve_rule_orders(pparams::PresolveParams)
    _validate_phase_rule_order(pparams.row_rule_order, VALID_ROW_PHASE_RULES, :row)
    _validate_phase_rule_order(pparams.col_rule_order, VALID_COL_PHASE_RULES, :col)
    return nothing
end

function _subset_presolve_params(
    pparams::PresolveParams;
    phase::Symbol,
    rule_order,
)
    subset = deepcopy(pparams)
    if phase == :row
        subset.row_rule_order = collect(rule_order)
        subset.col_rule_order = Symbol[]
        return subset
    elseif phase == :col
        subset.row_rule_order = Symbol[]
        subset.col_rule_order = collect(rule_order)
        return subset
    end
    error("Unknown presolve phase: $phase")
end

function _enabled_rule_subset(pparams::PresolveParams, rules)
    enabled = Symbol[]
    for rule_name in rules
        if _is_rule_enabled(pparams, rule_name)
            push!(enabled, rule_name)
        end
    end
    return enabled
end

function _run_phase_rule_subset(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    phase::Symbol,
    rule_order,
)
    isempty(rule_order) && return (lp, rec, false)

    subset_pparams = _subset_presolve_params(pparams; phase=phase, rule_order=rule_order)
    stats = presolve_compute_stats(lp, subset_pparams; phase=phase)
    plan = presolve_make_plan(lp, stats, subset_pparams; phase=phase)
    _throw_terminal_status_if_needed!(plan, phase)

    if !_phase_has_action(plan, phase)
        return (lp, rec, false)
    end

    return presolve_apply_plan(lp, plan, rec, subset_pparams; phase=phase)
end

function _run_trivial_cleanup_recirculation(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
)
    row_rules = _enabled_rule_subset(pparams, PRESOLVE_TRIVIAL_ROW_RULES)
    col_prefix_rules = _enabled_rule_subset(pparams, PRESOLVE_TRIVIAL_COL_PREFIX_RULES)
    col_suffix_rules = _enabled_rule_subset(pparams, PRESOLVE_TRIVIAL_COL_SUFFIX_RULES)

    isempty(row_rules) && isempty(col_prefix_rules) && isempty(col_suffix_rules) &&
        return (lp, rec, false)

    lp_cur = lp
    rec_cur = rec
    changed_any = false

    while true
        changed_pass = false

        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=col_prefix_rules,
        )
        changed_pass |= changed

        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:row,
            rule_order=row_rules,
        )
        changed_pass |= changed

        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=col_suffix_rules,
        )
        changed_pass |= changed

        changed_any |= changed_pass
        !changed_pass && break
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_phase_rule_sequence(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    phase::Symbol,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    for rule_name in _phase_rule_order(pparams, phase)
        _is_rule_enabled(pparams, rule_name) || continue

        repeat_to_exhaustion =
            phase == :col && (
                rule_name == :singleton_cols ||
                (
                    rule_name == :doubleton_eq &&
                    !pparams.doubleton_eq_single_batch_per_iter
                )
            )

        while true
            lp_cur, rec_cur, changed_rule = _run_phase_rule_subset(
                lp_cur,
                rec_cur,
                pparams;
                phase=phase,
                rule_order=(rule_name,),
            )
            changed_any |= changed_rule

            if changed_rule && _needs_cleanup_recirculation(rule_name)
                lp_cur, rec_cur, changed_cleanup = _run_trivial_cleanup_recirculation(
                    lp_cur,
                    rec_cur,
                    pparams,
                )
                changed_any |= changed_cleanup
            end

            if !repeat_to_exhaustion || !changed_rule
                break
            end
        end
    end

    return (lp_cur, rec_cur, changed_any)
end

"""
Reset phase-local plan for the current LP.
"""
function presolve_reset_plan(
    lp::LP_info_gpu,
    _pparams::PresolveParams;
    phase::Symbol,
)
    m, n = size(lp.A)
    plan = PresolvePlan_gpu(m, n, lp.c, lp.AL, lp.AU, lp.l, lp.u)
    if phase == :row || phase == :col
        return plan
    end
    error("Unknown presolve phase: $phase")
end

"""
Compute phase-local structural stats on GPU.
"""
function presolve_compute_stats(
    lp::LP_info_gpu,
    pparams::PresolveParams;
    phase::Symbol,
)
    m, n = size(lp.A)
    stats = PresolveStats_gpu(m, n)

    if phase == :row
        compute_row_nnz!(stats.row_nnz, lp.A)
        stats.empty_row_mask .= UInt8.(stats.row_nnz .== Int32(0))

        if _is_rule_enabled(pparams, :singleton_rows)
            stats.singleton_row_mask .= UInt8.(stats.row_nnz .== Int32(1))
            compute_singleton_row_support!(
                stats.singleton_row_col,
                stats.singleton_row_val,
                stats.row_nnz,
                lp.A,
            )
        end

        return stats
    end

    if phase == :col
        compute_col_nnz!(stats.col_nnz, lp.AT)
        stats.empty_col_mask .= UInt8.(stats.col_nnz .== Int32(0))

        if _is_rule_enabled(pparams, :singleton_cols)
            stats.singleton_col_mask .= UInt8.(stats.col_nnz .== Int32(1))
            compute_singleton_col_support!(
                stats.singleton_col_row,
                stats.singleton_col_val,
                stats.col_nnz,
                lp.AT,
            )
        end

        return stats
    end

    error("Unknown presolve phase: $phase")
end

"""
Build a phase-local plan by scheduling enabled rules in user order.
"""
function presolve_make_plan(
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams;
    phase::Symbol,
)
    plan = presolve_reset_plan(lp, pparams; phase=phase)

    if phase == :row
        for rule_name in pparams.row_rule_order
            if rule_name == :empty_rows
                if _is_rule_enabled(pparams, :empty_rows)
                    apply_rule_empty_rows!(plan, lp, stats, pparams)
                end
            elseif rule_name == :singleton_rows
                if _is_rule_enabled(pparams, :singleton_rows)
                    apply_rule_singleton_rows!(plan, lp, stats, pparams)
                end
            elseif rule_name == :activity_checks
                if _is_rule_enabled(pparams, :activity_checks)
                    apply_rule_activity_checks!(plan, lp, stats, pparams)
                end
            elseif rule_name == :primal_propagation
                if _is_rule_enabled(pparams, :primal_propagation)
                    apply_rule_primal_propagation!(plan, lp, stats, pparams)
                end
            elseif rule_name == :parallel_rows
                if _is_rule_enabled(pparams, :parallel_rows)
                    apply_rule_parallel_rows!(plan, lp, stats, pparams)
                end
            else
                error("Unknown row-phase rule symbol: $rule_name")
            end

            if plan.has_infeasible || plan.has_unbounded
                break
            end
        end
    elseif phase == :col
        for rule_name in pparams.col_rule_order
            if rule_name == :close_bounds
                if _is_rule_enabled(pparams, :close_bounds)
                    apply_rule_close_bounds!(plan, lp, pparams)
                end
            elseif rule_name == :empty_cols
                if _is_rule_enabled(pparams, :empty_cols)
                    apply_rule_empty_cols!(plan, lp, stats, pparams)
                end
            elseif rule_name == :singleton_cols
                if _is_rule_enabled(pparams, :singleton_cols)
                    apply_rule_singleton_cols!(plan, lp, stats, pparams)
                end
            elseif rule_name == :doubleton_eq
                if _is_rule_enabled(pparams, :doubleton_eq)
                    apply_rule_doubleton_eq!(plan, lp, stats, pparams)
                end
            elseif rule_name == :dual_fix
                if _is_rule_enabled(pparams, :dual_fix)
                    apply_rule_dual_fix!(plan, lp, stats, pparams)
                end
            elseif rule_name == :redundant_bounds
                # Deferred to the one-shot final cleanup pass after the main loop.
            elseif rule_name == :parallel_cols
                if _is_rule_enabled(pparams, :parallel_cols)
                    apply_rule_parallel_cols!(plan, lp, stats, pparams)
                end
            else
                error("Unknown col-phase rule symbol: $rule_name")
            end

            if plan.has_infeasible || plan.has_unbounded
                break
            end
        end
    else
        error("Unknown presolve phase: $phase")
    end

    if pparams.debug_checks
        presolve_phase_basic_checks!(lp, stats, plan)
    end

    return plan
end

"""
Apply a phase-local plan and update cumulative record mappings.
"""
function presolve_apply_plan(
    lp::LP_info_gpu,
    plan::PresolvePlan_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    phase::Symbol,
)
    m_old, n_old = size(lp.A)

    if phase == :row
        A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
        obj_new = lp.obj_constant + plan.obj_constant_delta
        tape_new = _merge_postsolve_tape(rec.tape, plan.tape, rec.row_red2org, rec.col_red2org)
        tape_gpu_parts_new = _merge_postsolve_tape_gpu_parts(rec.tape_gpu_parts, plan.tape, plan.tape_gpu, rec.row_red2org, rec.col_red2org)
        tape_gpu_new = tape_gpu_parts_new === rec.tape_gpu_parts ? rec.tape_gpu : nothing

        if !_row_phase_has_structural_change(plan)
            lp_new = LP_info_gpu(
                lp.A,
                lp.AT,
                copy(plan.new_c),
                copy(plan.new_AL),
                copy(plan.new_AU),
                copy(plan.new_l),
                copy(plan.new_u),
                obj_new,
                lp.AT_leading_slack,
                copy(lp.AT_slack_after),
            )
            rec_new = _copy_record_with_updates(
                rec;
                obj_constant_new=obj_new,
                tape=tape_new,
                tape_gpu=tape_gpu_new,
                tape_gpu_parts=tape_gpu_parts_new,
            )
            changed = plan.has_change
            return (lp_new, rec_new, changed)
        end

        row_org2red_local, row_red2org_local, m_new = build_maps_from_mask(plan.keep_row_mask)
        col_org2red_local, col_red2org_local, n_new = build_maps_from_mask(plan.keep_col_mask)

        A_rows = compact_csr_by_rows(A_source, row_red2org_local)
        AT_rows = transpose_csr(A_rows)
        A_new = compact_csr_by_cols(A_rows, col_red2org_local)

        AL_new = gather_by_red2org(plan.new_AL, row_red2org_local)
        AU_new = gather_by_red2org(plan.new_AU, row_red2org_local)
        c_new = gather_by_red2org(plan.new_c, col_red2org_local)
        l_new = gather_by_red2org(plan.new_l, col_red2org_local)
        u_new = gather_by_red2org(plan.new_u, col_red2org_local)
        old_col_lengths = _csr_row_lengths_gpu(lp.AT)
        source_col_lengths = _csr_row_lengths_gpu(AT_rows)
        source_leading_slack, source_slack = isnothing(plan.new_AT_slack_after) ?
                       _update_virtual_slack_without_shifts_gpu(
            old_col_lengths,
            lp.AT_leading_slack,
            lp.AT_slack_after,
            source_col_lengths,
        ) :
                       (something(plan.new_AT_leading_slack, Int32(0)), something(plan.new_AT_slack_after, CuVector{Int32}(undef, 0)))
        leading_slack_new, slack_new = _compact_virtual_slack_after_gpu(
            source_col_lengths,
            source_leading_slack,
            source_slack,
            col_red2org_local,
        )

        A_rows = nothing
        AT_rows = nothing
        _release_presolve_gpu_temps!()

        AT_new = transpose_csr(A_new)

        lp_new = LP_info_gpu(
            A_new,
            AT_new,
            c_new,
            AL_new,
            AU_new,
            l_new,
            u_new,
            obj_new,
            leading_slack_new,
            slack_new,
        )

        row_red2org_global = compose_red2org(rec.row_red2org, row_red2org_local)
        row_org2red_global = build_org2red_from_red2org(row_red2org_global, Int(rec.m0))
        col_red2org_global = compose_red2org(rec.col_red2org, col_red2org_local)
        col_org2red_global = build_org2red_from_red2org(col_red2org_global, Int(rec.n0))

        removed_rows_global = _collect_removed_global_indices_gpu(plan.keep_row_mask, rec.row_red2org)
        removed_cols_global = _collect_removed_global_indices_gpu(plan.keep_col_mask, rec.col_red2org)
        fixed_idx_global = _map_local_to_global_indices(plan.fixed_idx, rec.col_red2org)
        singleton_row_global = _map_local_to_global_indices(plan.singleton_col_row_idx, rec.row_red2org)
        singleton_col_global = _map_local_to_global_indices(plan.singleton_col_col_idx, rec.col_red2org)
        merged_from_global = _map_local_to_global_indices(plan.merged_col_from, rec.col_red2org)
        merged_to_global = _map_local_to_global_indices(plan.merged_col_to, rec.col_red2org)

        rec_new = PresolveRecord_gpu(
            rec.m0,
            rec.n0,
            Int32(m_new),
            Int32(n_new),
            row_org2red_global,
            row_red2org_global,
            col_org2red_global,
            col_red2org_global,
            _append_cuvector(rec.fixed_idx, fixed_idx_global),
            _append_cuvector(rec.fixed_val, plan.fixed_val),
            _append_cuvector(rec.removed_row_idx, removed_rows_global),
            _append_cuvector(rec.removed_col_idx, removed_cols_global),
            _append_cuvector(rec.singleton_col_row_idx, singleton_row_global),
            _append_cuvector(rec.singleton_col_col_idx, singleton_col_global),
            _append_cuvector(rec.merged_col_from, merged_from_global),
            _append_cuvector(rec.merged_col_to, merged_to_global),
            _append_cuvector(rec.merged_col_ratio, plan.merged_col_ratio),
            _append_cuvector(rec.merged_col_from_l, plan.merged_col_from_l),
            _append_cuvector(rec.merged_col_from_u, plan.merged_col_from_u),
            _append_cuvector(rec.merged_col_to_l, plan.merged_col_to_l),
            _append_cuvector(rec.merged_col_to_u, plan.merged_col_to_u),
            rec.obj_constant_old,
            obj_new,
            copy(rec.rule_counters),
            tape_new,
            tape_gpu_new,
            tape_gpu_parts_new,
        )

        if pparams.debug_checks
            debug_assert_maps!("row", rec_new.row_org2red, rec_new.row_red2org, Int(rec_new.m0), Int(rec_new.m1))
            debug_assert_maps!("col", rec_new.col_org2red, rec_new.col_red2org, Int(rec_new.n0), Int(rec_new.n1))
        end

        changed = plan.has_change || (Int(m_new) != m_old) || (Int(n_new) != n_old)
        return (lp_new, rec_new, changed)
    end

    if phase == :col
        A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
        obj_new = lp.obj_constant + plan.obj_constant_delta
        tape_new = _merge_postsolve_tape(rec.tape, plan.tape, rec.row_red2org, rec.col_red2org)
        tape_gpu_parts_new = _merge_postsolve_tape_gpu_parts(rec.tape_gpu_parts, plan.tape, plan.tape_gpu, rec.row_red2org, rec.col_red2org)
        tape_gpu_new = tape_gpu_parts_new === rec.tape_gpu_parts ? rec.tape_gpu : nothing

        if !_col_phase_has_structural_change(plan)
            lp_new = LP_info_gpu(
                lp.A,
                lp.AT,
                copy(plan.new_c),
                copy(plan.new_AL),
                copy(plan.new_AU),
                copy(plan.new_l),
                copy(plan.new_u),
                obj_new,
                lp.AT_leading_slack,
                copy(lp.AT_slack_after),
            )
            rec_new = _copy_record_with_updates(
                rec;
                obj_constant_new=obj_new,
                tape=tape_new,
                tape_gpu=tape_gpu_new,
                tape_gpu_parts=tape_gpu_parts_new,
            )
            changed = plan.has_change
            return (lp_new, rec_new, changed)
        end

        row_org2red_local, row_red2org_local, m_new = build_maps_from_mask(plan.keep_row_mask)
        col_org2red_local, col_red2org_local, n_new = build_maps_from_mask(plan.keep_col_mask)

        A_rows = compact_csr_by_rows(A_source, row_red2org_local)
        AT_rows = transpose_csr(A_rows)
        A_new = compact_csr_by_cols(A_rows, col_red2org_local)

        AL_new = gather_by_red2org(plan.new_AL, row_red2org_local)
        AU_new = gather_by_red2org(plan.new_AU, row_red2org_local)
        c_new = gather_by_red2org(plan.new_c, col_red2org_local)
        l_new = gather_by_red2org(plan.new_l, col_red2org_local)
        u_new = gather_by_red2org(plan.new_u, col_red2org_local)
        old_col_lengths = _csr_row_lengths_gpu(lp.AT)
        source_col_lengths = _csr_row_lengths_gpu(AT_rows)
        source_leading_slack, source_slack = isnothing(plan.new_AT_slack_after) ?
                       _update_virtual_slack_without_shifts_gpu(
            old_col_lengths,
            lp.AT_leading_slack,
            lp.AT_slack_after,
            source_col_lengths,
        ) :
                       (something(plan.new_AT_leading_slack, Int32(0)), something(plan.new_AT_slack_after, CuVector{Int32}(undef, 0)))
        leading_slack_new, slack_new = _compact_virtual_slack_after_gpu(
            source_col_lengths,
            source_leading_slack,
            source_slack,
            col_red2org_local,
        )

        A_rows = nothing
        AT_rows = nothing
        _release_presolve_gpu_temps!()

        AT_new = transpose_csr(A_new)

        lp_new = LP_info_gpu(
            A_new,
            AT_new,
            c_new,
            AL_new,
            AU_new,
            l_new,
            u_new,
            obj_new,
            leading_slack_new,
            slack_new,
        )

        row_red2org_global = compose_red2org(rec.row_red2org, row_red2org_local)
        row_org2red_global = build_org2red_from_red2org(row_red2org_global, Int(rec.m0))
        col_red2org_global = compose_red2org(rec.col_red2org, col_red2org_local)
        col_org2red_global = build_org2red_from_red2org(col_red2org_global, Int(rec.n0))

        removed_rows_global = _collect_removed_global_indices_gpu(plan.keep_row_mask, rec.row_red2org)
        removed_cols_global = _collect_removed_global_indices_gpu(plan.keep_col_mask, rec.col_red2org)
        fixed_idx_global = _map_local_to_global_indices(plan.fixed_idx, rec.col_red2org)
        singleton_row_global = _map_local_to_global_indices(plan.singleton_col_row_idx, rec.row_red2org)
        singleton_col_global = _map_local_to_global_indices(plan.singleton_col_col_idx, rec.col_red2org)
        merged_from_global = _map_local_to_global_indices(plan.merged_col_from, rec.col_red2org)
        merged_to_global = _map_local_to_global_indices(plan.merged_col_to, rec.col_red2org)

        rec_new = PresolveRecord_gpu(
            rec.m0,
            rec.n0,
            Int32(m_new),
            Int32(n_new),
            row_org2red_global,
            row_red2org_global,
            col_org2red_global,
            col_red2org_global,
            _append_cuvector(rec.fixed_idx, fixed_idx_global),
            _append_cuvector(rec.fixed_val, plan.fixed_val),
            _append_cuvector(rec.removed_row_idx, removed_rows_global),
            _append_cuvector(rec.removed_col_idx, removed_cols_global),
            _append_cuvector(rec.singleton_col_row_idx, singleton_row_global),
            _append_cuvector(rec.singleton_col_col_idx, singleton_col_global),
            _append_cuvector(rec.merged_col_from, merged_from_global),
            _append_cuvector(rec.merged_col_to, merged_to_global),
            _append_cuvector(rec.merged_col_ratio, plan.merged_col_ratio),
            _append_cuvector(rec.merged_col_from_l, plan.merged_col_from_l),
            _append_cuvector(rec.merged_col_from_u, plan.merged_col_from_u),
            _append_cuvector(rec.merged_col_to_l, plan.merged_col_to_l),
            _append_cuvector(rec.merged_col_to_u, plan.merged_col_to_u),
            rec.obj_constant_old,
            obj_new,
            copy(rec.rule_counters),
            tape_new,
            tape_gpu_new,
            tape_gpu_parts_new,
        )

        if pparams.debug_checks
            debug_assert_maps!("row", rec_new.row_org2red, rec_new.row_red2org, Int(rec_new.m0), Int(rec_new.m1))
            debug_assert_maps!("col", rec_new.col_org2red, rec_new.col_red2org, Int(rec_new.n0), Int(rec_new.n1))
        end

        changed = plan.has_change || (Int(m_new) != m_old) || (Int(n_new) != n_old)
        return (lp_new, rec_new, changed)
    end

    error("Unknown presolve phase: $phase")
end

"""
Public GPU presolve entrypoint.

Calling order per iteration:
1. row phase: stats -> plan -> apply
2. col phase: stats -> plan -> apply
"""
function presolve_gpu(
    lp::LP_info_gpu,
    params::HPRLP_parameters;
    presolve_params::PresolveParams=PresolveParams(),
)
    m0, n0 = size(lp.A)
    presolve_params.record_postsolve_tape = presolve_params.record_postsolve_tape
    _validate_presolve_rule_orders(presolve_params)
    lp_cur = lp
    rec = presolve_identity_record(m0, n0, lp.obj_constant)

    if presolve_params.verbose || params.verbose
        println(">>> [GPU Presolve] start (m=$m0, n=$n0, max_iters=$(presolve_params.max_iters))")
    end
    t_start = time()

    for iter in 1:presolve_params.max_iters
        changed_iter = false

        lp_cur, rec, changed_row = _run_phase_rule_sequence(
            lp_cur,
            rec,
            presolve_params;
            phase=:row,
        )
        changed_iter |= changed_row

        lp_cur, rec, changed_col = _run_phase_rule_sequence(
            lp_cur,
            rec,
            presolve_params;
            phase=:col,
        )
        changed_iter |= changed_col

        if presolve_params.verbose || params.verbose
            m1, n1 = size(lp_cur.A)
            println(">>> [GPU Presolve] iter=$iter, changed=$changed_iter, dims=($m1, $n1)")
        end

        if !changed_iter
            break
        end
    end

    if _is_rule_enabled(presolve_params, :redundant_bounds)
        stats_cleanup = presolve_compute_stats(lp_cur, presolve_params; phase=:col)
        plan_cleanup = presolve_reset_plan(lp_cur, presolve_params; phase=:col)
        apply_rule_redundant_bounds!(plan_cleanup, lp_cur, stats_cleanup, presolve_params)
        _throw_terminal_status_if_needed!(plan_cleanup, :col)

        if plan_cleanup.has_col_action
            lp_cur, rec, _ = presolve_apply_plan(
                lp_cur,
                plan_cleanup,
                rec,
                presolve_params;
                phase=:col,
            )
        end
    end

    if presolve_params.verbose || params.verbose
        m1, n1 = size(lp_cur.A)
        println(
            ">>> [GPU Presolve] done (m=$m0->$m1, n=$n0->$n1) in ",
            round(time() - t_start; digits=4),
            "s",
        )
    end

    return (lp_cur, rec)
end
