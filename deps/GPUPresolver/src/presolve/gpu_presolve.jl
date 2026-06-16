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
    :structural_l1_substitution,
    :singleton_cols_dual_infer,
    :singleton_cols_eq,
    :primal_propagation,
    :dual_fix,
    :parallel_cols,
    :fme_projection,
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
    :structural_l1_substitution,
    :empty_cols,
    :singleton_cols_dual_infer,
    :singleton_cols_eq,
    :doubleton_eq,
    :linear_eq_agg,
    :dual_fix,
    :parallel_cols,
    :fme_projection,
    :redundant_bounds,
)

@inline function _is_rule_enabled(pparams::PresolveParams, rule_name::Symbol)
    if rule_name == :close_bounds
        return pparams.enable_close_bounds
    elseif rule_name == :empty_rows
        return pparams.enable_empty_rows
    elseif rule_name == :singleton_rows
        return pparams.enable_singleton_rows
    elseif rule_name == :activity_checks
        return pparams.enable_activity_checks
    elseif rule_name == :primal_propagation
        return pparams.enable_primal_propagation
    elseif rule_name == :parallel_rows
        return pparams.enable_parallel_rows
    elseif rule_name == :empty_cols
        return pparams.enable_empty_cols
    elseif rule_name == :structural_l1_substitution
        return pparams.enable_structural_l1_substitution
    elseif rule_name == :singleton_cols_eq
        return pparams.enable_singleton_cols_eq
    elseif rule_name == :singleton_cols_dual_infer
        return pparams.enable_singleton_cols_dual_infer
    elseif rule_name == :doubleton_eq
        return pparams.enable_doubleton_eq
    elseif rule_name == :linear_eq_agg
        return pparams.enable_linear_eq_agg
    elseif rule_name == :dual_fix
        return pparams.enable_dual_fix
    elseif rule_name == :parallel_cols
        return pparams.enable_parallel_cols
    elseif rule_name == :fme_projection
        return pparams.enable_fme_projection
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

@inline _bytes_to_gib(bytes::Integer) = Float64(bytes) / 2.0^30

function _gpu_memory_status_string()
    available = try
        CUDA.available_memory()
    catch
        nothing
    end

    total = try
        CUDA.total_memory()
    catch
        nothing
    end

    if isnothing(available) || isnothing(total)
        return "gpu_mem=unavailable"
    end

    used = max(total - available, 0)
    return string(
        "gpu_used_gib=",
        round(_bytes_to_gib(used); digits=3),
        "/",
        round(_bytes_to_gib(total); digits=3),
        ", gpu_free_gib=",
        round(_bytes_to_gib(available); digits=3),
    )
end

function _matrix_shape_string(matrix)
    m, n = size(matrix)
    return "dims=($m,$n), nnz=$(length(matrix.nzVal))"
end

function _log_presolve_memory!(
    pparams::PresolveParams,
    phase::Symbol,
    stage::AbstractString;
    matrix=nothing,
    extra::AbstractString="",
)
    # Memory logging is intentionally disabled by default.
    _ = pparams
    _ = phase
    _ = stage
    _ = matrix
    _ = extra
    return nothing
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
        copy(rec.structural_primal_recoveries),
    )
end

@inline function _map_local_col_to_global(local_col::Integer, col_red2org::AbstractVector{Int32})
    return Int32(col_red2org[Int(local_col)])
end

function _globalize_structural_primal_recovery(
    step::StructuralL1PrimalRecoveryStep,
    col_red2org::AbstractVector{Int32},
)
    splits = StructuralL1SplitRecovery[
        StructuralL1SplitRecovery(
            _map_local_col_to_global(split.t_col, col_red2org),
            _map_local_col_to_global(split.e_col, col_red2org),
            split.rho,
        ) for split in step.splits
    ]
    outer_pairs = StructuralOuterPairRecovery[
        StructuralOuterPairRecovery(
            _map_local_col_to_global(pair.bound_col, col_red2org),
            _map_local_col_to_global(pair.free_col, col_red2org),
        ) for pair in step.outer_pairs
    ]
    linked_slacks = StructuralLinkedSlackRecovery[
        StructuralLinkedSlackRecovery(
            _map_local_col_to_global(slack.slack_col, col_red2org),
            _map_local_col_to_global(slack.t_col, col_red2org),
            slack.factor,
        ) for slack in step.linked_slacks
    ]
    max_slacks = StructuralMaxSlackRecovery[
        StructuralMaxSlackRecovery(
            _map_local_col_to_global(slack.slack_col, col_red2org),
            Int32[_map_local_col_to_global(t_col, col_red2org) for t_col in slack.t_cols],
            copy(slack.factors),
        ) for slack in step.max_slacks
    ]
    return StructuralL1PrimalRecoveryStep(
        step.pattern,
        splits,
        outer_pairs,
        linked_slacks,
        max_slacks,
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

function _identity_red2org_gpu(n::Integer)
    n <= 0 && return CuVector{Int32}(undef, 0)
    return CuVector(Int32.(1:Int(n)))
end

"""
Pick a "sink" original row index for FME-appended (projected) rows. Projected
rows have no original-model counterpart; mapping them to a deleted original row
keeps every `row_red2org` entry a valid original index (so the shared scatter /
map kernels need no changes), and the FME tape replay later zeroes that row's
dual. Returns the global index of the first deleted original row, or 1 as a
fallback when nothing was deleted.
"""
function _fme_sink_global_row(
    keep_mask_local::CuVector{UInt8},
    old_red2org_global::CuVector{Int32},
    n_orig_reduced::Integer,
)
    n_orig_reduced <= 0 && return Int32(1)
    mask = keep_mask_local[1:Int(n_orig_reduced)]
    removed_mask = UInt8.(mask .== UInt8(0))
    _, removed_local, removed_count = build_maps_from_mask(removed_mask)
    Int(removed_count) == 0 && return Int32(1)
    first_local = Int(_copy_scalar_to_host(removed_local, 1))
    return Int32(_copy_scalar_to_host(old_red2org_global, first_local))
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
        # Some col-phase rules only tighten row sides (e.g. singleton_cols_dual_infer).
        return plan.has_row_action || plan.has_col_action
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
    pparams.gpu_presolve_scheduler in (:fixed, :tiered) ||
        error("gpu_presolve_scheduler must be one of :fixed or :tiered")
    pparams.max_time >= 0.0 ||
        error("max_time must be non-negative")
    pparams.tiered_cleanup_max_rounds >= 0 ||
        error("tiered_cleanup_max_rounds must be non-negative")
    pparams.tiered_max_light_streak >= 0 ||
        error("tiered_max_light_streak must be non-negative")
    pparams.tiered_global_period >= 1 ||
        error("tiered_global_period must be at least 1")
    0.0 < pparams.tiered_light_continue_ratio <= 1.0 ||
        error("tiered_light_continue_ratio must be in (0, 1]")
    0.0 < pparams.tiered_cycle_stop_ratio <= 1.0 ||
        error("tiered_cycle_stop_ratio must be in (0, 1]")
    return nothing
end

@inline function _presolve_time_exceeded(start_time, pparams::PresolveParams)
    start_time === nothing && return false
    return isfinite(pparams.max_time) && time() - start_time >= pparams.max_time
end

function _subset_presolve_params(
    pparams::PresolveParams;
    phase::Symbol,
    rule_order,
)
    subset = deepcopy(pparams)
    rules = Set(Symbol.(rule_order))
    subset.enable_empty_rows = :empty_rows in rules
    subset.enable_singleton_rows = :singleton_rows in rules
    subset.enable_activity_checks = :activity_checks in rules
    subset.enable_primal_propagation = :primal_propagation in rules
    subset.enable_parallel_rows = :parallel_rows in rules

    subset.enable_close_bounds = :close_bounds in rules
    subset.enable_structural_l1_substitution = :structural_l1_substitution in rules
    subset.enable_empty_cols = :empty_cols in rules
    subset.enable_singleton_cols_dual_infer = :singleton_cols_dual_infer in rules
    subset.enable_singleton_cols_eq = :singleton_cols_eq in rules
    subset.enable_doubleton_eq = :doubleton_eq in rules
    subset.enable_linear_eq_agg = :linear_eq_agg in rules
    subset.enable_dual_fix = :dual_fix in rules
    subset.enable_parallel_cols = :parallel_cols in rules
    subset.enable_fme_projection = :fme_projection in rules
    subset.enable_redundant_bounds = :redundant_bounds in rules

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

    profile_rules = _gpu_presolver_env_enabled("GPUPRESOLVER_PRESOLVE_RULE_PROFILE")
    t_rule = 0.0
    m_before = 0
    n_before = 0
    if profile_rules
        CUDA.synchronize()
        t_rule = time()
        m_before, n_before = size(lp.A)
    end

    if phase == :col &&
       length(rule_order) == 1 &&
       first(rule_order) == :structural_l1_substitution &&
       _is_rule_enabled(pparams, :structural_l1_substitution) &&
       !_sls_gpu_prefix_screen(lp, pparams.structural_l1_pattern)
        if profile_rules
            CUDA.synchronize()
            elapsed = time() - t_rule
            println(">>> [GPU Presolve profile] phase=$phase rules=$(collect(rule_order)) changed=false dims=($m_before,$n_before)->($m_before,$n_before) time=$(round(elapsed; digits=6))s stats=0.0s plan=$(round(elapsed; digits=6))s apply=0.0s")
        end
        return (lp, rec, false)
    end

    subset_pparams = _subset_presolve_params(pparams; phase=phase, rule_order=rule_order)
    stats = presolve_compute_stats(lp, subset_pparams; phase=phase)
    t_after_stats = 0.0
    if profile_rules
        CUDA.synchronize()
        t_after_stats = time()
    end
    plan = presolve_make_plan(lp, stats, subset_pparams; phase=phase)
    _throw_terminal_status_if_needed!(plan, phase)
    t_after_plan = 0.0
    if profile_rules
        CUDA.synchronize()
        t_after_plan = time()
    end

    if !_phase_has_action(plan, phase)
        if profile_rules
            elapsed = t_after_plan - t_rule
            stats_time = t_after_stats - t_rule
            plan_time = t_after_plan - t_after_stats
            println(">>> [GPU Presolve profile] phase=$phase rules=$(collect(rule_order)) changed=false dims=($m_before,$n_before)->($m_before,$n_before) time=$(round(elapsed; digits=6))s stats=$(round(stats_time; digits=6))s plan=$(round(plan_time; digits=6))s apply=0.0s")
        end
        return (lp, rec, false)
    end

    lp_new, rec_new, changed = presolve_apply_plan(lp, plan, rec, subset_pparams; phase=phase)
    if profile_rules
        CUDA.synchronize()
        t_after_apply = time()
        elapsed = t_after_apply - t_rule
        stats_time = t_after_stats - t_rule
        plan_time = t_after_plan - t_after_stats
        apply_time = t_after_apply - t_after_plan
        m_after, n_after = size(lp_new.A)
        println(">>> [GPU Presolve profile] phase=$phase rules=$(collect(rule_order)) changed=$changed dims=($m_before,$n_before)->($m_after,$n_after) time=$(round(elapsed; digits=6))s stats=$(round(stats_time; digits=6))s plan=$(round(plan_time; digits=6))s apply=$(round(apply_time; digits=6))s")
    end
    return (lp_new, rec_new, changed)
end

function _run_trivial_cleanup_recirculation(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
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
        _presolve_time_exceeded(start_time, pparams) && break
        changed_pass = false

        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=col_prefix_rules,
        )
        changed_pass |= changed
        _presolve_time_exceeded(start_time, pparams) && break

        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:row,
            rule_order=row_rules,
        )
        changed_pass |= changed
        _presolve_time_exceeded(start_time, pparams) && break

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
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    for rule_name in _phase_rule_order(pparams, phase)
        _presolve_time_exceeded(start_time, pparams) && break
        _is_rule_enabled(pparams, rule_name) || continue

        repeat_to_exhaustion =
            phase == :col && (
                rule_name == :singleton_cols_eq ||
                rule_name == :singleton_cols_dual_infer ||
                (
                    rule_name == :doubleton_eq &&
                    !pparams.doubleton_eq_single_batch_per_iter
                )
            )

        while true
            _presolve_time_exceeded(start_time, pparams) && break
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
                    start_time=start_time,
                )
                changed_any |= changed_cleanup
            end

            if !repeat_to_exhaustion || !changed_rule ||
               _presolve_time_exceeded(start_time, pparams)
                break
            end
        end
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_host_iteration_callback(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    iter::Int,
    start_time=nothing,
)
    _presolve_time_exceeded(start_time, pparams) && return (lp, rec, false)
    callback = pparams.host_iteration_callback
    isnothing(callback) && return (lp, rec, false)
    return callback(lp, rec, pparams; iter=iter, start_time=start_time)
end

@inline function _presolve_nnz(lp::LP_info_gpu)
    return length(lp.A.nzVal)
end

@inline function _has_good_nnz_progress(
    nnz_before::Integer,
    nnz_after::Integer,
    ratio::Float64,
)
    nnz_before <= 0 && return false
    return Float64(nnz_after) < ratio * Float64(nnz_before)
end

const TIERED_LIGHT_ROW_RULES = (:singleton_rows,)
const TIERED_LIGHT_COL_RULES = (:singleton_cols_dual_infer, :singleton_cols_eq)
const TIERED_MEDIUM_ROW_RULES = (:activity_checks, :primal_propagation, :parallel_rows)
const TIERED_MEDIUM_COL_RULES = (:parallel_cols, :fme_projection)
const TIERED_HEAVY_ROW_RULES = ()
const TIERED_HEAVY_COL_RULES = (:doubleton_eq,)

const TIERED_CLEANUP_COL_PREFIX_RULES = (:close_bounds, :empty_cols, :dual_fix)
const TIERED_CLEANUP_ROW_SUFFIX_RULES = (:empty_rows,)
const TIERED_CLEANUP_COL_SUFFIX_RULES = (:empty_cols,)
const TIERED_MEDIUM_PROPAGATION_ROW_RULES = (:activity_checks, :primal_propagation)
const TIERED_MEDIUM_PARALLEL_ROW_RULES = (:parallel_rows,)

function _run_tiered_rule_list(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    phase::Symbol,
    rule_order,
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    for rule_name in rule_order
        _presolve_time_exceeded(start_time, pparams) && break
        _is_rule_enabled(pparams, rule_name) || continue
        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=phase,
            rule_order=(rule_name,),
        )
        changed_any |= changed
        _presolve_time_exceeded(start_time, pparams) && break
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_repeated_col_rule(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    rule_name::Symbol,
    ;
    start_time=nothing,
)
    _is_rule_enabled(pparams, rule_name) || return (lp, rec, false)

    lp_cur = lp
    rec_cur = rec
    changed_any = false

    while true
        _presolve_time_exceeded(start_time, pparams) && break
        lp_cur, rec_cur, changed = _run_phase_rule_subset(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=(rule_name,),
        )
        changed_any |= changed
        (!changed || _presolve_time_exceeded(start_time, pparams)) && break
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_singleton_rows_to_exhaustion(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    while true
        _presolve_time_exceeded(start_time, pparams) && break
        lp_cur, rec_cur, changed = _run_tiered_rule_list(
            lp_cur,
            rec_cur,
            pparams;
            phase=:row,
            rule_order=TIERED_LIGHT_ROW_RULES,
            start_time=start_time,
        )
        changed_any |= changed
        (!changed || _presolve_time_exceeded(start_time, pparams)) && break
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_cleanup(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    lp_cur, rec_cur, changed = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:col,
        rule_order=TIERED_CLEANUP_COL_PREFIX_RULES,
        start_time=start_time,
    )
    changed_any |= changed
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed =
        _run_tiered_singleton_rows_to_exhaustion(lp_cur, rec_cur, pparams; start_time=start_time)
    changed_any |= changed
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:row,
        rule_order=TIERED_CLEANUP_ROW_SUFFIX_RULES,
        start_time=start_time,
    )
    changed_any |= changed
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:col,
        rule_order=TIERED_CLEANUP_COL_SUFFIX_RULES,
        start_time=start_time,
    )
    changed_any |= changed

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_doubleton_rule(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    if !pparams.doubleton_eq_single_batch_per_iter
        return _run_tiered_repeated_col_rule(lp, rec, pparams, :doubleton_eq; start_time=start_time)
    end

    return _run_tiered_rule_list(
        lp,
        rec,
        pparams;
        phase=:col,
        rule_order=TIERED_HEAVY_COL_RULES,
        start_time=start_time,
    )
end

function _run_tiered_fast_phase(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    changed_singleton = false
    while true
        _presolve_time_exceeded(start_time, pparams) && break
        lp_cur, rec_cur, changed = _run_tiered_rule_list(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=TIERED_LIGHT_COL_RULES,
            start_time=start_time,
        )
        changed_any |= changed
        changed_singleton |= changed
        if changed
            lp_cur, rec_cur, changed_cleanup =
                _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
            changed_any |= changed_cleanup
        end
        (!changed || _presolve_time_exceeded(start_time, pparams)) && break
    end
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed =
        _run_tiered_doubleton_rule(lp_cur, rec_cur, pparams; start_time=start_time)
    changed_any |= changed
    if changed || changed_singleton
        lp_cur, rec_cur, changed_cleanup =
            _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
        changed_any |= changed_cleanup
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_medium_phase(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    changed_any = false

    lp_cur, rec_cur, changed = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:row,
        rule_order=TIERED_MEDIUM_PROPAGATION_ROW_RULES,
        start_time=start_time,
    )
    changed_any |= changed
    if changed
        lp_cur, rec_cur, changed_cleanup =
            _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
        changed_any |= changed_cleanup
    end
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:row,
        rule_order=TIERED_MEDIUM_PARALLEL_ROW_RULES,
        start_time=start_time,
    )
    changed_any |= changed
    _presolve_time_exceeded(start_time, pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed_col = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:col,
        rule_order=TIERED_MEDIUM_COL_RULES,
        start_time=start_time,
    )
    changed_any |= changed_col

    if changed || changed_col
        lp_cur, rec_cur, changed_cleanup =
            _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
        changed_any |= changed_cleanup
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_bootstrap_phase(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    # Run structural L1 substitution before any bootstrap row work so
    # structure-sensitive patterns are not disrupted by generic reductions.
    bootstrap_pparams = deepcopy(pparams)
    bootstrap_pparams.col_rule_order = Symbol[
        rule for rule in bootstrap_pparams.col_rule_order
        if rule != :doubleton_eq && rule != :structural_l1_substitution
    ]
    bootstrap_pparams.row_rule_order = Symbol[
        rule for rule in bootstrap_pparams.row_rule_order
        if rule != :primal_propagation
    ]

    lp_cur = lp
    rec_cur = rec
    changed_any = false

    lp_cur, rec_cur, changed_structural = _run_tiered_rule_list(
        lp_cur,
        rec_cur,
        pparams;
        phase=:col,
        rule_order=(:structural_l1_substitution,),
        start_time=start_time,
    )
    changed_any |= changed_structural
    _presolve_time_exceeded(start_time, bootstrap_pparams) && return (lp_cur, rec_cur, changed_any)

    if changed_structural
        lp_cur, rec_cur, changed_cleanup =
            _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
        changed_any |= changed_cleanup
        _presolve_time_exceeded(start_time, bootstrap_pparams) && return (lp_cur, rec_cur, changed_any)
    end

    # Continue with one fixed-style bootstrap pass so early row/col
    # interactions are still harvested before switching to staged mode.
    lp_cur, rec_cur, changed_row = _run_phase_rule_sequence(
        lp_cur,
        rec_cur,
        bootstrap_pparams;
        phase=:row,
        start_time=start_time,
    )
    changed_any |= changed_row
    _presolve_time_exceeded(start_time, bootstrap_pparams) && return (lp_cur, rec_cur, changed_any)

    lp_cur, rec_cur, changed_col = _run_phase_rule_sequence(
        lp_cur,
        rec_cur,
        bootstrap_pparams;
        phase=:col,
        start_time=start_time,
    )
    changed_any |= changed_col
    _presolve_time_exceeded(start_time, bootstrap_pparams) && return (lp_cur, rec_cur, changed_any)

    if changed_row || changed_col
        # Generic bootstrap reductions can expose structural L1 blocks that were
        # hidden by removable rows/columns in the initial matrix.
        lp_cur, rec_cur, changed_structural_after = _run_tiered_rule_list(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            rule_order=(:structural_l1_substitution,),
            start_time=start_time,
        )
        changed_any |= changed_structural_after
        _presolve_time_exceeded(start_time, bootstrap_pparams) && return (lp_cur, rec_cur, changed_any)

        if changed_structural_after
            lp_cur, rec_cur, changed_cleanup =
                _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
            changed_any |= changed_cleanup
        end
    end

    return (lp_cur, rec_cur, changed_any)
end

function _run_tiered_presolve_loop(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec
    phase = :fast
    cycle_nnz_before = _presolve_nnz(lp_cur)
    progress_ratio = 0.95

    for iter in 1:pparams.max_iters
        _presolve_time_exceeded(start_time, pparams) && break
        changed_iter = false

        if iter == 1 && pparams.enable_tiered_bootstrap
            nnz_before_bootstrap = _presolve_nnz(lp_cur)
            lp_cur, rec_cur, changed_bootstrap = _run_tiered_bootstrap_phase(
                lp_cur,
                rec_cur,
                pparams;
                start_time=start_time,
            )
            changed_iter |= changed_bootstrap
            _presolve_time_exceeded(start_time, pparams) && break

            if pparams.verbose
                m_now, n_now = size(lp_cur.A)
                nnz_after_bootstrap = _presolve_nnz(lp_cur)
                println(">>> [GPU Presolve tiered] iter=1, phase=bootstrap, changed=$changed_bootstrap, dims=($m_now, $n_now), nnz=$nnz_before_bootstrap->$nnz_after_bootstrap")
            end

            lp_cur, rec_cur, changed_host = _run_host_iteration_callback(
                lp_cur,
                rec_cur,
                pparams;
                iter=iter,
                start_time=start_time,
            )
            changed_iter |= changed_host
            _presolve_time_exceeded(start_time, pparams) && break

            cycle_nnz_before = _presolve_nnz(lp_cur)
            phase = :fast

            if !changed_iter
                break
            end

            continue
        end

        lp_cur, rec_cur, changed_cleanup =
            _run_tiered_cleanup(lp_cur, rec_cur, pparams; start_time=start_time)
        changed_iter |= changed_cleanup
        _presolve_time_exceeded(start_time, pparams) && break

        nnz_before_phase = _presolve_nnz(lp_cur)

        if phase == :fast
            lp_cur, rec_cur, changed_phase =
                _run_tiered_fast_phase(lp_cur, rec_cur, pparams; start_time=start_time)
            changed_iter |= changed_phase
            _presolve_time_exceeded(start_time, pparams) && break

            nnz_after_phase = _presolve_nnz(lp_cur)
            fast_productive = _has_good_nnz_progress(
                nnz_before_phase,
                nnz_after_phase,
                progress_ratio,
            )
            phase = fast_productive ? :fast : :medium

            if pparams.verbose
                m_now, n_now = size(lp_cur.A)
                println(">>> [GPU Presolve tiered] iter=$iter, phase=fast, changed=$changed_iter, productive=$fast_productive, dims=($m_now, $n_now), nnz=$nnz_before_phase->$nnz_after_phase")
            end
        else
            lp_cur, rec_cur, changed_phase =
                _run_tiered_medium_phase(lp_cur, rec_cur, pparams; start_time=start_time)
            changed_iter |= changed_phase
            _presolve_time_exceeded(start_time, pparams) && break

            nnz_after_cycle = _presolve_nnz(lp_cur)
            cycle_productive = _has_good_nnz_progress(
                cycle_nnz_before,
                nnz_after_cycle,
                progress_ratio,
            )

            if pparams.verbose
                m_now, n_now = size(lp_cur.A)
                println(">>> [GPU Presolve tiered] iter=$iter, phase=medium, changed=$changed_iter, productive=$cycle_productive, dims=($m_now, $n_now), nnz=$cycle_nnz_before->$nnz_after_cycle")
            end

            lp_cur, rec_cur, changed_host = _run_host_iteration_callback(
                lp_cur,
                rec_cur,
                pparams;
                iter=iter,
                start_time=start_time,
            )
            changed_iter |= changed_host
            _presolve_time_exceeded(start_time, pparams) && break

            if !changed_iter || !cycle_productive
                break
            end

            cycle_nnz_before = nnz_after_cycle
            phase = :fast
            continue
        end

        lp_cur, rec_cur, changed_host = _run_host_iteration_callback(
            lp_cur,
            rec_cur,
            pparams;
            iter=iter,
            start_time=start_time,
        )
        changed_iter |= changed_host
        _presolve_time_exceeded(start_time, pparams) && break

        if !changed_iter && phase != :medium
            break
        end
    end

    return (lp_cur, rec_cur)
end

function _run_fixed_presolve_loop(
    lp::LP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams,
    ;
    start_time=nothing,
)
    lp_cur = lp
    rec_cur = rec

    for iter in 1:pparams.max_iters
        _presolve_time_exceeded(start_time, pparams) && break
        changed_iter = false

        lp_cur, rec_cur, changed_row = _run_phase_rule_sequence(
            lp_cur,
            rec_cur,
            pparams;
            phase=:row,
            start_time=start_time,
        )
        changed_iter |= changed_row
        _presolve_time_exceeded(start_time, pparams) && break

        lp_cur, rec_cur, changed_col = _run_phase_rule_sequence(
            lp_cur,
            rec_cur,
            pparams;
            phase=:col,
            start_time=start_time,
        )
        changed_iter |= changed_col
        _presolve_time_exceeded(start_time, pparams) && break

        lp_cur, rec_cur, changed_host = _run_host_iteration_callback(
            lp_cur,
            rec_cur,
            pparams;
            iter=iter,
            start_time=start_time,
        )
        changed_iter |= changed_host
        _presolve_time_exceeded(start_time, pparams) && break

        if pparams.verbose
            m1, n1 = size(lp_cur.A)
            println(">>> [GPU Presolve] iter=$iter, changed=$changed_iter, dims=($m1, $n1)")
        end
        _log_presolve_memory!(pparams, :global, "iter=$iter"; matrix=lp_cur.A, extra="changed=$changed_iter")

        changed_iter || break
    end

    return (lp_cur, rec_cur)
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
        stats.row_nnz_valid = true

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
        needs_col_nnz =
            _is_rule_enabled(pparams, :empty_cols) ||
            _is_rule_enabled(pparams, :singleton_cols_eq) ||
            _is_rule_enabled(pparams, :singleton_cols_dual_infer) ||
            _is_rule_enabled(pparams, :doubleton_eq) ||
            _is_rule_enabled(pparams, :linear_eq_agg) ||
            _is_rule_enabled(pparams, :dual_fix) ||
            _is_rule_enabled(pparams, :parallel_cols) ||
            _is_rule_enabled(pparams, :fme_projection) ||
            _is_rule_enabled(pparams, :redundant_bounds)

        if needs_col_nnz
            compute_col_nnz!(stats.col_nnz, lp.AT)
            stats.col_nnz_valid = true
        end

        if _is_rule_enabled(pparams, :empty_cols)
            stats.empty_col_mask .= UInt8.(stats.col_nnz .== Int32(0))
        end

        if _is_rule_enabled(pparams, :structural_l1_substitution)
            _sls_gpu_screen_flag(lp, stats, pparams.structural_l1_pattern)
        end

        if _is_rule_enabled(pparams, :singleton_cols_eq) ||
           _is_rule_enabled(pparams, :singleton_cols_dual_infer)
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
            elseif rule_name == :structural_l1_substitution
                if _is_rule_enabled(pparams, :structural_l1_substitution)
                    apply_rule_structural_l1_substitution!(plan, lp, stats, pparams)
                end
            elseif rule_name == :empty_cols
                if _is_rule_enabled(pparams, :empty_cols)
                    apply_rule_empty_cols!(plan, lp, stats, pparams)
                end
            elseif rule_name == :singleton_cols_dual_infer
                if _is_rule_enabled(pparams, :singleton_cols_dual_infer)
                    apply_rule_singleton_cols_dual_infer!(plan, lp, stats, pparams)
                end
            elseif rule_name == :singleton_cols_eq
                if _is_rule_enabled(pparams, :singleton_cols_eq)
                    apply_rule_singleton_cols_eq!(plan, lp, stats, pparams)
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
            elseif rule_name == :fme_projection
                if _is_rule_enabled(pparams, :fme_projection)
                    apply_rule_fme_projection!(plan, lp, stats, pparams)
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

        _log_presolve_memory!(
            pparams,
            phase,
            "rebuild:start";
            matrix=A_source,
            extra="source_override=$(!isnothing(plan.new_A)), target_dims=($(Int(m_new)),$(Int(n_new))), removed_rows=$(m_old - Int(m_new)), removed_cols=$(n_old - Int(n_new))",
        )

        profile_rebuild = _gpu_presolver_env_enabled("GPUPRESOLVER_PRESOLVE_REBUILD_PROFILE")
        t_rebuild = 0.0
        t_last = 0.0
        if profile_rebuild && phase == :col
            CUDA.synchronize()
            t_rebuild = time()
            t_last = t_rebuild
        end

        A_new = compact_csr_by_rows_and_cols(A_source, row_red2org_local, col_red2org_local)
        _log_presolve_memory!(pparams, phase, "rebuild:A_new"; matrix=A_new)
        if profile_rebuild && phase == :col
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase target_dims=($(Int(m_new)),$(Int(n_new))) removed=($(m_old - Int(m_new)),$(n_old - Int(n_new))) compact=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

        AT_new = transpose_csr(A_new)
        _log_presolve_memory!(pparams, phase, "rebuild:AT_new"; matrix=AT_new)
        if profile_rebuild && phase == :col
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase transpose=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

        AL_new = gather_by_red2org(plan.new_AL, row_red2org_local)
        AU_new = gather_by_red2org(plan.new_AU, row_red2org_local)
        c_new = gather_by_red2org(plan.new_c, col_red2org_local)
        l_new = gather_by_red2org(plan.new_l, col_red2org_local)
        u_new = gather_by_red2org(plan.new_u, col_red2org_local)
        leading_slack_new = Int32(0)
        slack_new = CUDA.zeros(Int32, Int(n_new))
        if profile_rebuild && phase == :col
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase gather=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

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

        # FME can grow the row space with appended projected rows. Appended rows
        # have no original-model counterpart, so map them to a deleted "sink"
        # original row: `m0` stays the original row dimension and the shared
        # scatter/map kernels keep receiving valid original indices.
        fme_appended_rows = length(plan.keep_row_mask) > length(rec.row_red2org)

        row_red2org_global = if fme_appended_rows
            n_orig_reduced = length(rec.row_red2org)
            n_appended = length(plan.keep_row_mask) - n_orig_reduced
            sink_global = _fme_sink_global_row(plan.keep_row_mask, rec.row_red2org, n_orig_reduced)
            extended_src = vcat(rec.row_red2org, CUDA.fill(Int32(sink_global), n_appended))
            gather_by_red2org(extended_src, row_red2org_local)
        else
            compose_red2org(rec.row_red2org, row_red2org_local)
        end
        row_org2red_global = build_org2red_from_red2org(
            row_red2org_global,
            Int(rec.m0),
        )
        col_red2org_global = compose_red2org(rec.col_red2org, col_red2org_local)
        col_org2red_global = build_org2red_from_red2org(col_red2org_global, Int(rec.n0))
        removed_rows_global = if fme_appended_rows
            _collect_removed_global_indices_gpu(
                plan.keep_row_mask[1:length(rec.row_red2org)],
                rec.row_red2org,
            )
        else
            _collect_removed_global_indices_gpu(plan.keep_row_mask, rec.row_red2org)
        end
        removed_cols_global = _collect_removed_global_indices_gpu(plan.keep_col_mask, rec.col_red2org)
        fixed_idx_global = _map_local_to_global_indices(plan.fixed_idx, rec.col_red2org)
        singleton_row_global = _map_local_to_global_indices(plan.singleton_col_row_idx, rec.row_red2org)
        singleton_col_global = _map_local_to_global_indices(plan.singleton_col_col_idx, rec.col_red2org)
        merged_from_global = _map_local_to_global_indices(plan.merged_col_from, rec.col_red2org)
        merged_to_global = _map_local_to_global_indices(plan.merged_col_to, rec.col_red2org)
        structural_primal_recoveries = copy(rec.structural_primal_recoveries)
        if !isnothing(plan.structural_primal_recovery)
            push!(
                structural_primal_recoveries,
                _globalize_structural_primal_recovery(
                    plan.structural_primal_recovery,
                    Array(rec.col_red2org),
                ),
            )
        end

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
            structural_primal_recoveries,
        )

        if pparams.debug_checks && !fme_appended_rows
            debug_assert_maps!("row", rec_new.row_org2red, rec_new.row_red2org, Int(rec_new.m0), Int(rec_new.m1))
            debug_assert_maps!("col", rec_new.col_org2red, rec_new.col_red2org, Int(rec_new.n0), Int(rec_new.n1))
        end

        _log_presolve_memory!(pparams, phase, "rebuild:done"; matrix=lp_new.A)
        if profile_rebuild && phase == :col
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase record=$(round(t_now - t_last; digits=6))s total=$(round(t_now - t_rebuild; digits=6))s")
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

        _log_presolve_memory!(
            pparams,
            phase,
            "rebuild:start";
            matrix=A_source,
            extra="source_override=$(!isnothing(plan.new_A)), target_dims=($(Int(m_new)),$(Int(n_new))), removed_rows=$(m_old - Int(m_new)), removed_cols=$(n_old - Int(n_new))",
        )

        profile_rebuild = _gpu_presolver_env_enabled("GPUPRESOLVER_PRESOLVE_REBUILD_PROFILE")
        t_rebuild = 0.0
        t_last = 0.0
        if profile_rebuild
            CUDA.synchronize()
            t_rebuild = time()
            t_last = t_rebuild
        end

        A_new = compact_csr_by_rows_and_cols(A_source, row_red2org_local, col_red2org_local)
        _log_presolve_memory!(pparams, phase, "rebuild:A_new"; matrix=A_new)
        if profile_rebuild
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase target_dims=($(Int(m_new)),$(Int(n_new))) removed=($(m_old - Int(m_new)),$(n_old - Int(n_new))) compact=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

        AT_new = transpose_csr(A_new)
        _log_presolve_memory!(pparams, phase, "rebuild:AT_new"; matrix=AT_new)
        if profile_rebuild
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase transpose=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

        AL_new = gather_by_red2org(plan.new_AL, row_red2org_local)
        AU_new = gather_by_red2org(plan.new_AU, row_red2org_local)
        c_new = gather_by_red2org(plan.new_c, col_red2org_local)
        l_new = gather_by_red2org(plan.new_l, col_red2org_local)
        u_new = gather_by_red2org(plan.new_u, col_red2org_local)
        leading_slack_new = Int32(0)
        slack_new = CUDA.zeros(Int32, Int(n_new))
        if profile_rebuild
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase gather=$(round(t_now - t_last; digits=6))s")
            t_last = t_now
        end

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

        # FME can grow the row space with appended projected rows. Appended rows
        # have no original-model counterpart, so map them to a deleted "sink"
        # original row: `m0` stays the original row dimension and the shared
        # scatter/map kernels keep receiving valid original indices.
        fme_appended_rows = length(plan.keep_row_mask) > length(rec.row_red2org)

        row_red2org_global = if fme_appended_rows
            n_orig_reduced = length(rec.row_red2org)
            n_appended = length(plan.keep_row_mask) - n_orig_reduced
            sink_global = _fme_sink_global_row(plan.keep_row_mask, rec.row_red2org, n_orig_reduced)
            extended_src = vcat(rec.row_red2org, CUDA.fill(Int32(sink_global), n_appended))
            gather_by_red2org(extended_src, row_red2org_local)
        else
            compose_red2org(rec.row_red2org, row_red2org_local)
        end
        row_org2red_global = build_org2red_from_red2org(
            row_red2org_global,
            Int(rec.m0),
        )
        col_red2org_global = compose_red2org(rec.col_red2org, col_red2org_local)
        col_org2red_global = build_org2red_from_red2org(col_red2org_global, Int(rec.n0))
        removed_rows_global = if fme_appended_rows
            _collect_removed_global_indices_gpu(
                plan.keep_row_mask[1:length(rec.row_red2org)],
                rec.row_red2org,
            )
        else
            _collect_removed_global_indices_gpu(plan.keep_row_mask, rec.row_red2org)
        end
        removed_cols_global = _collect_removed_global_indices_gpu(plan.keep_col_mask, rec.col_red2org)
        fixed_idx_global = _map_local_to_global_indices(plan.fixed_idx, rec.col_red2org)
        singleton_row_global = _map_local_to_global_indices(plan.singleton_col_row_idx, rec.row_red2org)
        singleton_col_global = _map_local_to_global_indices(plan.singleton_col_col_idx, rec.col_red2org)
        merged_from_global = _map_local_to_global_indices(plan.merged_col_from, rec.col_red2org)
        merged_to_global = _map_local_to_global_indices(plan.merged_col_to, rec.col_red2org)
        structural_primal_recoveries = copy(rec.structural_primal_recoveries)
        if !isnothing(plan.structural_primal_recovery)
            push!(
                structural_primal_recoveries,
                _globalize_structural_primal_recovery(
                    plan.structural_primal_recovery,
                    Array(rec.col_red2org),
                ),
            )
        end

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
            structural_primal_recoveries,
        )

        if pparams.debug_checks && !fme_appended_rows
            debug_assert_maps!("row", rec_new.row_org2red, rec_new.row_red2org, Int(rec_new.m0), Int(rec_new.m1))
            debug_assert_maps!("col", rec_new.col_org2red, rec_new.col_red2org, Int(rec_new.n0), Int(rec_new.n1))
        end

        _log_presolve_memory!(pparams, phase, "rebuild:done"; matrix=lp_new.A)
        if profile_rebuild
            CUDA.synchronize()
            t_now = time()
            println(">>> [GPU Presolve rebuild profile] phase=$phase record=$(round(t_now - t_last; digits=6))s total=$(round(t_now - t_rebuild; digits=6))s")
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
    lp::LP_info_gpu;
    presolve_params::PresolveParams=PresolveParams(),
    verbose::Bool=false,
)
    m0, n0 = size(lp.A)
    presolve_params.verbose = presolve_params.verbose || verbose
    _validate_presolve_rule_orders(presolve_params)
    lp_cur = lp
    rec = presolve_identity_record(m0, n0, lp.obj_constant)

    if presolve_params.verbose
        println(">>> [GPU Presolve] start (m=$m0, n=$n0, max_iters=$(presolve_params.max_iters))")
    end
    _log_presolve_memory!(presolve_params, :global, "start"; matrix=lp.A)
    CUDA.synchronize()
    t_start = time()

    _validate_presolve_rule_orders(presolve_params)
    scheduler = presolve_params.gpu_presolve_scheduler
    if scheduler == :tiered
        lp_cur, rec = _run_tiered_presolve_loop(lp_cur, rec, presolve_params; start_time=t_start)
    else
        lp_cur, rec = _run_fixed_presolve_loop(lp_cur, rec, presolve_params; start_time=t_start)
    end

    if _is_rule_enabled(presolve_params, :redundant_bounds) &&
       !_presolve_time_exceeded(t_start, presolve_params)
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

    CUDA.synchronize()
    if presolve_params.verbose
        m1, n1 = size(lp_cur.A)
        println(
            ">>> [GPU Presolve] done (m=$m0->$m1, n=$n0->$n1) in ",
            round(time() - t_start; digits=4),
            "s",
        )
    end
    _log_presolve_memory!(presolve_params, :global, "done"; matrix=lp_cur.A)

    return (lp_cur, rec)
end

function presolve_gpu(
    lp::LP_info_gpu,
    params::GPUPresolverParameters;
    presolve_params::PresolveParams=PresolveParams(),
)
    return presolve_gpu(lp; presolve_params=presolve_params, verbose=params.verbose)
end
