"""
Helpers for isolated presolve-rule verification.
"""

const CURRENT_GPU_RULES = (
    :close_bounds,
    :empty_rows,
    :singleton_rows,
    :activity_checks,
    :primal_propagation,
    :parallel_rows,
    :empty_cols,
    :singleton_cols,
    :doubleton_eq,
    :dual_fix,
    :parallel_cols,
    :redundant_bounds,
)

function build_isolated_presolve_params(
    rule::Symbol;
    enabled::Bool=true,
    max_iters::Int=20,
    verbose::Bool=false,
    debug_checks::Bool=false,
)
    rule in CURRENT_GPU_RULES || error("Unsupported presolve rule: $rule")

    pparams = PresolveParams()
    pparams.max_iters = max_iters
    pparams.verbose = verbose
    pparams.debug_checks = debug_checks

    pparams.enable_empty_rows = false
    pparams.enable_singleton_rows = false
    pparams.enable_activity_checks = false
    pparams.enable_primal_propagation = false
    pparams.enable_parallel_rows = false
    pparams.enable_empty_cols = false
    pparams.enable_singleton_cols = false
    pparams.enable_doubleton_eq = false
    pparams.enable_dual_fix = false
    pparams.enable_parallel_cols = false
    pparams.enable_redundant_bounds = false
    pparams.enable_close_bounds = false
    pparams.enable_remove_empty_rows = false
    pparams.enable_remove_empty_cols = false
    pparams.row_rule_order = Symbol[]
    pparams.col_rule_order = Symbol[]

    if enabled
        if rule == :close_bounds
            pparams.enable_close_bounds = true
            pparams.col_rule_order = [:close_bounds]
        elseif rule == :empty_rows
            pparams.enable_empty_rows = true
            pparams.enable_remove_empty_rows = true
            pparams.row_rule_order = [:empty_rows]
        elseif rule == :singleton_rows
            pparams.enable_singleton_rows = true
            pparams.row_rule_order = [:singleton_rows]
        elseif rule == :activity_checks
            pparams.enable_activity_checks = true
            pparams.row_rule_order = [:activity_checks]
        elseif rule == :primal_propagation
            pparams.enable_primal_propagation = true
            pparams.row_rule_order = [:primal_propagation]
        elseif rule == :parallel_rows
            pparams.enable_parallel_rows = true
            pparams.row_rule_order = [:parallel_rows]
        elseif rule == :empty_cols
            pparams.enable_empty_cols = true
            pparams.enable_remove_empty_cols = true
            pparams.col_rule_order = [:empty_cols]
        elseif rule == :singleton_cols
            pparams.enable_singleton_cols = true
            pparams.col_rule_order = [:singleton_cols]
        elseif rule == :doubleton_eq
            pparams.enable_doubleton_eq = true
            pparams.col_rule_order = [:doubleton_eq]
        elseif rule == :dual_fix
            pparams.enable_dual_fix = true
            pparams.col_rule_order = [:dual_fix]
        elseif rule == :parallel_cols
            pparams.enable_parallel_cols = true
            pparams.col_rule_order = [:parallel_cols]
        elseif rule == :redundant_bounds
            pparams.enable_redundant_bounds = true
        end
    end

    return pparams
end

const _SINGLETON_DIAG_ELIMINATED = Int32(1)
const _SINGLETON_DIAG_ROW_OWNER_CONFLICT = Int32(2)
const _SINGLETON_DIAG_NOT_IMPLIED_FREE_EQ = Int32(3)
const _SINGLETON_DIAG_NOT_IMPLIED_FREE_INEQ = Int32(4)
const _SINGLETON_DIAG_ACTIVE_SIDE_INFINITE = Int32(5)
const _SINGLETON_DIAG_ROW_TOO_SMALL = Int32(6)
const _SINGLETON_DIAG_INACTIVE_TARGET = Int32(7)
const _SINGLETON_DIAG_ZERO_COEFF = Int32(8)
const _SINGLETON_DIAG_NONFINITE_ACTIVITY = Int32(9)
const _SINGLETON_DIAG_TIGHTEN_LHS_TO_RHS = Int32(10)
const _SINGLETON_DIAG_TIGHTEN_RHS_TO_LHS = Int32(11)

function _kernel_singleton_col_row_owner_diag!(
    row_owner,
    row_candidate_count,
    keep_row,
    keep_col,
    singleton_mask,
    support_row,
    support_val,
    c_cur,
    l_cur,
    u_cur,
    AL_cur,
    AU_cur,
    row_ptr,
    col_val,
    nz_val,
    tol,
    zero_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if keep_col[j] == UInt8(0) || singleton_mask[j] == UInt8(0)
            return
        end

        @inbounds row = support_row[j]
        if row < 1 || row > length(keep_row) || keep_row[row] == UInt8(0)
            return
        end

        @inbounds a = support_val[j]
        abs(a) > zero_tol || return

        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
        live_row_nnz = Int32(0)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                if keep_col[col] != UInt8(0)
                    live_row_nnz += Int32(1)
                    live_row_nnz > Int32(1) && break
                end
            end
        end
        live_row_nnz > Int32(1) || return

        @inbounds lhs = AL_cur[row]
        @inbounds rhs = AU_cur[row]
        rest_min, rest_max, rest_finite = _singleton_col_activity_bounds(
            row_start,
            row_stop,
            j,
            keep_col,
            col_val,
            nz_val,
            l_cur,
            u_cur,
        )
        rest_finite || return

        CUDA.@atomic row_candidate_count[row] += Int32(1)

        if isfinite(lhs) && isfinite(rhs) && abs(lhs - rhs) <= tol
            x1 = (rhs - rest_min) / a
            x2 = (rhs - rest_max) / a
            implied_lb = min(x1, x2)
            implied_ub = max(x1, x2)

            @inbounds impl_free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol)
            @inbounds impl_free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol)
            (impl_free_from_above || impl_free_from_below) || return
        else
            impl_free_from_above = _singleton_col_implied_free_from_above(
                a,
                lhs,
                rhs,
                u_cur[j],
                rest_min,
                rest_max,
                tol,
            )
            impl_free_from_below = _singleton_col_implied_free_from_below(
                a,
                lhs,
                rhs,
                l_cur[j],
                rest_min,
                rest_max,
                tol,
            )
            @inbounds ineq_action, _ = _singleton_col_ineq_action(
                c_cur[j],
                a,
                lhs,
                rhs,
                impl_free_from_above,
                impl_free_from_below,
                zero_tol,
            )
            ineq_action != _SINGLETON_COL_INEQ_NO_ACTION || return
        end

        CUDA.@atomic row_owner[row] = min(row_owner[row], Int32(j))
    end
    return
end

function _kernel_diagnose_singleton_cols!(
    decision_code,
    won_row_owner,
    impl_free_from_above_mask,
    impl_free_from_below_mask,
    is_equality_row_mask,
    row_owner,
    keep_row,
    keep_col,
    singleton_mask,
    support_row,
    support_val,
    c_cur,
    l_cur,
    u_cur,
    AL_cur,
    AU_cur,
    row_ptr,
    col_val,
    nz_val,
    tol,
    zero_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if keep_col[j] == UInt8(0) || singleton_mask[j] == UInt8(0)
            return
        end

        @inbounds row = support_row[j]
        if row < 1 || row > length(keep_row) || keep_row[row] == UInt8(0)
            @inbounds decision_code[j] = _SINGLETON_DIAG_INACTIVE_TARGET
            return
        end

        @inbounds a = support_val[j]
        if !(abs(a) > zero_tol)
            @inbounds decision_code[j] = _SINGLETON_DIAG_ZERO_COEFF
            return
        end

        @inbounds row_start = row_ptr[row]
        @inbounds row_stop = row_ptr[row + 1] - Int32(1)
        live_row_nnz = Int32(0)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                if keep_col[col] != UInt8(0)
                    live_row_nnz += Int32(1)
                    live_row_nnz > Int32(1) && break
                end
            end
        end
        if !(live_row_nnz > Int32(1))
            @inbounds decision_code[j] = _SINGLETON_DIAG_ROW_TOO_SMALL
            return
        end

        @inbounds won = row_owner[row] == Int32(j)
        @inbounds won_row_owner[j] = won ? UInt8(1) : UInt8(0)

        @inbounds lhs = AL_cur[row]
        @inbounds rhs = AU_cur[row]
        is_eq = isfinite(lhs) && isfinite(rhs) && abs(lhs - rhs) <= tol
        @inbounds is_equality_row_mask[j] = is_eq ? UInt8(1) : UInt8(0)

        rest_min, rest_max, rest_finite = _singleton_col_activity_bounds(
            row_start,
            row_stop,
            j,
            keep_col,
            col_val,
            nz_val,
            l_cur,
            u_cur,
        )
        if !rest_finite
            @inbounds decision_code[j] = _SINGLETON_DIAG_NONFINITE_ACTIVITY
            return
        end

        if is_eq
            x1 = (rhs - rest_min) / a
            x2 = (rhs - rest_max) / a
            implied_lb = min(x1, x2)
            implied_ub = max(x1, x2)

            free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol)
            free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol)
            @inbounds impl_free_from_above_mask[j] = free_from_above ? UInt8(1) : UInt8(0)
            @inbounds impl_free_from_below_mask[j] = free_from_below ? UInt8(1) : UInt8(0)

            if free_from_above || free_from_below
                if won
                    @inbounds decision_code[j] = _SINGLETON_DIAG_ELIMINATED
                else
                    @inbounds decision_code[j] = _SINGLETON_DIAG_ROW_OWNER_CONFLICT
                end
            else
                @inbounds decision_code[j] = _SINGLETON_DIAG_NOT_IMPLIED_FREE_EQ
            end
            return
        end

        free_from_above = _singleton_col_implied_free_from_above(
            a,
            lhs,
            rhs,
            u_cur[j],
            rest_min,
            rest_max,
            tol,
        )
        free_from_below = _singleton_col_implied_free_from_below(
            a,
            lhs,
            rhs,
            l_cur[j],
            rest_min,
            rest_max,
            tol,
        )
        @inbounds impl_free_from_above_mask[j] = free_from_above ? UInt8(1) : UInt8(0)
        @inbounds impl_free_from_below_mask[j] = free_from_below ? UInt8(1) : UInt8(0)

        @inbounds ineq_action, _ = _singleton_col_ineq_action(
            c_cur[j],
            a,
            lhs,
            rhs,
            free_from_above,
            free_from_below,
            zero_tol,
        )

        if ineq_action == _SINGLETON_COL_INEQ_NO_ACTION
            @inbounds decision_code[j] = _SINGLETON_DIAG_NOT_IMPLIED_FREE_INEQ
            return
        end

        if ineq_action == _SINGLETON_COL_INEQ_ELIMINATE
            if won
                @inbounds decision_code[j] = _SINGLETON_DIAG_ELIMINATED
            else
                @inbounds decision_code[j] = _SINGLETON_DIAG_ROW_OWNER_CONFLICT
            end
        elseif ineq_action == _SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS
            if won
                @inbounds decision_code[j] = _SINGLETON_DIAG_TIGHTEN_LHS_TO_RHS
            else
                @inbounds decision_code[j] = _SINGLETON_DIAG_ROW_OWNER_CONFLICT
            end
        else
            if won
                @inbounds decision_code[j] = _SINGLETON_DIAG_TIGHTEN_RHS_TO_LHS
            else
                @inbounds decision_code[j] = _SINGLETON_DIAG_ROW_OWNER_CONFLICT
            end
        end
    end
    return
end

@inline function _singleton_diag_decision_symbol(code::Integer)
    if code == _SINGLETON_DIAG_ELIMINATED
        return :eliminated
    elseif code == _SINGLETON_DIAG_ROW_OWNER_CONFLICT
        return :row_owner_conflict
    elseif code == _SINGLETON_DIAG_NOT_IMPLIED_FREE_EQ
        return :not_implied_free_eq
    elseif code == _SINGLETON_DIAG_NOT_IMPLIED_FREE_INEQ
        return :not_implied_free_ineq
    elseif code == _SINGLETON_DIAG_ACTIVE_SIDE_INFINITE
        return :active_side_infinite
    elseif code == _SINGLETON_DIAG_ROW_TOO_SMALL
        return :row_too_small
    elseif code == _SINGLETON_DIAG_INACTIVE_TARGET
        return :inactive_target
    elseif code == _SINGLETON_DIAG_ZERO_COEFF
        return :zero_coeff
    elseif code == _SINGLETON_DIAG_NONFINITE_ACTIVITY
        return :nonfinite_activity
    elseif code == _SINGLETON_DIAG_TIGHTEN_LHS_TO_RHS
        return :tighten_lhs_to_rhs
    elseif code == _SINGLETON_DIAG_TIGHTEN_RHS_TO_LHS
        return :tighten_rhs_to_lhs
    end
    return :unknown
end

function diagnose_singleton_cols(
    lp::LP_info_gpu,
    pparams::PresolveParams=build_isolated_presolve_params(:singleton_cols);
    trace_limit::Int=1000,
)
    m, n = size(lp.A)
    if m == 0 || n == 0
        return (
            summary=(
                m=m,
                n=n,
                candidate_count=0,
                equality_candidate_count=0,
                inequality_candidate_count=0,
                row_owner_win_count=0,
                row_owner_conflict_count=0,
                implied_free_reject_count=0,
                active_side_infinite_count=0,
                row_too_small_count=0,
                eliminated_count=0,
                row_tighten_eq_count=0,
                trace_sample_count=0,
                trace_limit=trace_limit,
                trace_truncated=false,
            ),
            trace=NamedTuple[],
        )
    end

    stats = presolve_compute_stats(lp, pparams; phase=:col)
    keep_row = CUDA.fill(UInt8(1), m)
    keep_col = CUDA.fill(UInt8(1), n)
    row_owner = CUDA.fill(typemax(Int32), m)
    row_candidate_count = CUDA.zeros(Int32, m)
    blocks = cld(n, GPU_PRESOLVE_THREADS)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_col_row_owner_diag!(
        row_owner,
        row_candidate_count,
        keep_row,
        keep_col,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        lp.c,
        lp.l,
        lp.u,
        lp.AL,
        lp.AU,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        Int32(n),
    )

    decision_code = CUDA.zeros(Int32, n)
    won_row_owner = CUDA.zeros(UInt8, n)
    impl_free_from_above_mask = CUDA.zeros(UInt8, n)
    impl_free_from_below_mask = CUDA.zeros(UInt8, n)
    is_equality_row_mask = CUDA.zeros(UInt8, n)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_diagnose_singleton_cols!(
        decision_code,
        won_row_owner,
        impl_free_from_above_mask,
        impl_free_from_below_mask,
        is_equality_row_mask,
        row_owner,
        keep_row,
        keep_col,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        lp.c,
        lp.l,
        lp.u,
        lp.AL,
        lp.AU,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        Int32(n),
    )

    singleton_mask_cpu = Array(stats.singleton_col_mask)
    support_row_cpu = Array(stats.singleton_col_row)
    support_val_cpu = Array(stats.singleton_col_val)
    row_owner_cpu = Array(row_owner)
    row_candidate_count_cpu = Array(row_candidate_count)
    decision_code_cpu = Array(decision_code)
    won_row_owner_cpu = Array(won_row_owner)
    impl_free_from_above_cpu = Array(impl_free_from_above_mask)
    impl_free_from_below_cpu = Array(impl_free_from_below_mask)
    is_equality_row_cpu = Array(is_equality_row_mask)

    candidate_idx = findall(==(UInt8(1)), singleton_mask_cpu)
    trace_cap = trace_limit < 0 ? typemax(Int) : trace_limit
    trace = NamedTuple[]

    equality_candidate_count = 0
    row_owner_win_count = 0
    row_owner_conflict_count = 0
    implied_free_reject_count = 0
    active_side_infinite_count = 0
    row_too_small_count = 0
    eliminated_count = 0
    row_tighten_eq_count = 0

    for j in candidate_idx
        decision = _singleton_diag_decision_symbol(decision_code_cpu[j])
        is_eq = is_equality_row_cpu[j] != UInt8(0)
        won = won_row_owner_cpu[j] != UInt8(0)

        equality_candidate_count += is_eq ? 1 : 0
        row_owner_win_count += won ? 1 : 0
        row_owner_conflict_count += decision == :row_owner_conflict ? 1 : 0
        implied_free_reject_count +=
            (decision == :not_implied_free_eq || decision == :not_implied_free_ineq) ? 1 : 0
        active_side_infinite_count += decision == :active_side_infinite ? 1 : 0
        row_too_small_count += decision == :row_too_small ? 1 : 0
        eliminated_count += decision == :eliminated ? 1 : 0
        row_tighten_eq_count +=
            (decision == :tighten_lhs_to_rhs || decision == :tighten_rhs_to_lhs) ? 1 : 0

        if length(trace) < trace_cap
            row = Int(support_row_cpu[j])
            row_owner_col =
                (1 <= row <= m && row_owner_cpu[row] != typemax(Int32)) ? Int(row_owner_cpu[row]) : 0
            row_candidate_total =
                (1 <= row <= m) ? Int(row_candidate_count_cpu[row]) : 0

            push!(
                trace,
                (
                    col=Int(j),
                    row=row,
                    coeff=support_val_cpu[j],
                    row_singleton_candidate_count=row_candidate_total,
                    row_owner_col=row_owner_col,
                    won_row_owner=won,
                    is_equality_row=is_eq,
                    impl_free_from_above=impl_free_from_above_cpu[j] != UInt8(0),
                    impl_free_from_below=impl_free_from_below_cpu[j] != UInt8(0),
                    decision=decision,
                ),
            )
        end
    end

    candidate_count = length(candidate_idx)
    inequality_candidate_count = candidate_count - equality_candidate_count
    summary = (
        m=m,
        n=n,
        candidate_count=candidate_count,
        equality_candidate_count=equality_candidate_count,
        inequality_candidate_count=inequality_candidate_count,
        row_owner_win_count=row_owner_win_count,
        row_owner_conflict_count=row_owner_conflict_count,
        implied_free_reject_count=implied_free_reject_count,
        active_side_infinite_count=active_side_infinite_count,
        row_too_small_count=row_too_small_count,
        eliminated_count=eliminated_count,
        row_tighten_eq_count=row_tighten_eq_count,
        trace_sample_count=length(trace),
        trace_limit=trace_limit,
        trace_truncated=candidate_count > length(trace),
    )

    return (summary=summary, trace=trace)
end

function diagnose_singleton_cols(
    file::AbstractString;
    device_number::Int=0,
    verbose::Bool=false,
    debug_checks::Bool=false,
    max_iters::Int=20,
    trace_limit::Int=1000,
)
    isfile(file) || error("MPS file not found: $file")

    params = HPRLP_parameters()
    params.use_gpu = true
    params.device_number = device_number
    params.warm_up = false
    params.verbose = verbose
    params.presolve = "NONE"
    params.use_resolve = false
    params.use_postsolve = false

    model, _ = build_from_mps(String(file), params)
    lp_gpu = setup_gpu_model(model, params)
    pparams = build_isolated_presolve_params(
        :singleton_cols;
        max_iters=max_iters,
        verbose=verbose,
        debug_checks=debug_checks,
    )
    report = diagnose_singleton_cols(lp_gpu, pparams; trace_limit=trace_limit)
    return merge((file=abspath(file),), report)
end

function _has_nonfinite_entries(v::AbstractVector{<:Real})
    return any(x -> !isfinite(x), v)
end

function _cleanup_rule_validation_gpu_memory!()
    GC.gc(true)
    try
        CUDA.reclaim()
    catch
    end
    return nothing
end

function validate_isolated_rule_postsolve(
    file::AbstractString,
    rule::Symbol;
    time_limit::Float64=20.0,
    presolve_time_limit::Float64=time_limit,
    max_iter::Int=20_000,
    stoptol::Float64=1.0e-4,
    device_number::Int=0,
    verbose::Bool=false,
    presolve_max_iters::Int=20,
    debug_checks::Bool=false,
)
    rule in CURRENT_GPU_RULES || error("Unsupported presolve rule: $rule")
    isfile(file) || error("MPS file not found: $file")

    _cleanup_rule_validation_gpu_memory!()
    report = nothing
    try
        params = HPRLP_parameters()
        params.use_gpu = true
        params.device_number = device_number
        params.warm_up = false
        params.verbose = verbose
        params.presolve = "GPU"
        params.use_resolve = false
        params.use_postsolve = true
        params.time_limit = time_limit
        params.presolve_time_limit = presolve_time_limit
        params.max_iter = max_iter
        params.stoptol = stoptol
        params.presolve_stoptol = stoptol

        model, original_lp = build_from_mps(String(file), params)
        pparams = build_isolated_presolve_params(
            rule;
            max_iters=presolve_max_iters,
            verbose=verbose,
            debug_checks=debug_checks,
        )

        lp_gpu = setup_gpu_model(model, params)
        lp_red, rec = presolve_gpu(lp_gpu, params; presolve_params=pparams)
        fired = isolated_rule_fired(model, lp_red, rec)

        results = optimize(model, params, nothing; presolve_params=pparams)
        x_cpu = Array(results.x)
        y_cpu = Array(results.y)
        z_cpu = Array(results.z)
        postsolve_refine_duals_from_original!(
            x_cpu,
            y_cpu,
            z_cpu,
            rec,
            original_lp,
        )
        p_obj, d_obj, p_feas, d_feas, gap, delta_y, delta_z =
            compute_original_kkt_metrics(original_lp, x_cpu, y_cpu, z_cpu)

        report = (
            file=abspath(file),
            rule=rule,
            fired=fired,
            status=results.status,
            m0=Int(rec.m0),
            n0=Int(rec.n0),
            m1=Int(rec.m1),
            n1=Int(rec.n1),
            presolve_time=results.presolve_time,
            postsolve_time=results.postsolve_time,
            solve_time=results.time,
            reduced_primal_obj=results.primal_obj,
            original_primal_obj=p_obj,
            original_dual_obj=d_obj,
            original_p_feas=p_feas,
            original_d_feas=d_feas,
            original_gap=gap,
            delta_y=delta_y,
            delta_z=delta_z,
            has_nonfinite=_has_nonfinite_entries(x_cpu) ||
                          _has_nonfinite_entries(y_cpu) ||
                          _has_nonfinite_entries(z_cpu),
        )
    finally
        _cleanup_rule_validation_gpu_memory!()
    end
    return report
end

function diagnose_isolated_bound_change_postsolve(
    file::AbstractString,
    rule::Symbol;
    time_limit::Float64=20.0,
    presolve_time_limit::Float64=time_limit,
    max_iter::Int=20_000,
    stoptol::Float64=1.0e-4,
    device_number::Int=0,
    verbose::Bool=false,
    presolve_max_iters::Int=20,
    debug_checks::Bool=false,
    tol::Float64=1.0e-7,
)
    rule in (:primal_propagation, :redundant_bounds, :activity_checks, :singleton_rows) ||
        error("Unsupported bound-change diagnostic rule: $rule")
    isfile(file) || error("MPS file not found: $file")

    _cleanup_rule_validation_gpu_memory!()
    report = nothing
    try
        params = HPRLP_parameters()
    params.use_gpu = true
    params.device_number = device_number
    params.warm_up = false
    params.verbose = verbose
    params.presolve = "GPU"
    params.use_resolve = false
    params.use_postsolve = true
    params.time_limit = time_limit
    params.presolve_time_limit = presolve_time_limit
    params.max_iter = max_iter
    params.stoptol = stoptol
    params.presolve_stoptol = stoptol

    model, original_lp = build_from_mps(String(file), params)
    pparams = build_isolated_presolve_params(
        rule;
        max_iters=presolve_max_iters,
        verbose=verbose,
        debug_checks=debug_checks,
    )

    lp_gpu = setup_gpu_model(model, params)
    lp_red, rec = presolve_gpu(lp_gpu, params; presolve_params=pparams)
    fired = isolated_rule_fired(model, lp_red, rec)
    results = optimize(model, params, nothing; presolve_params=pparams)

    x_org = Array(results.x)
    y_org = Array(results.y)
    z_org = Array(results.z)
    ATy = original_lp.AT * y_org
    row_activity = original_lp.A * x_org

    ensure_postsolve_tape_cpu!(rec)
    bound_change_rows = Int[]
    record_count = postsolve_record_count(rec.tape)
    for k in record_count:-1:1
        reduction_type = rec.tape.types[k]
        reduction_type == BOUND_CHANGE_THE_ROW || continue
        idx = postsolve_record_indices(rec.tape, k)
        length(idx) >= 2 || continue
        row = Int(idx[2])
        push!(bound_change_rows, row)
        _refine_live_row_dual_from_original_cached!(
            x_org,
            y_org,
            z_org,
            ATy,
            row_activity,
            original_lp,
            row;
            tol=tol,
        )
    end

    affected_rows = unique(bound_change_rows)
    if !isempty(affected_rows)
        for _ in 1:10
            _project_column_duals_from_ATy!(
                x_org,
                z_org,
                ATy,
                original_lp,
                tol=tol,
            )
            for row in affected_rows
                _refine_live_row_dual_from_original_cached!(
                    x_org,
                    y_org,
                    z_org,
                    ATy,
                    row_activity,
                    original_lp,
                    row;
                    tol=tol,
                )
            end
        end
    end

    stats = _targeted_bound_change_dual_reconstruction_with_stats!(
        x_org,
        y_org,
        z_org,
        original_lp,
        affected_rows;
        tol=tol,
    )

    _cleanup_unbounded_dual_slacks!(
        z_org,
        original_lp.l,
        original_lp.u;
        tol=tol,
    )

    p_obj, d_obj, p_feas, d_feas, gap, delta_y, delta_z =
        compute_original_kkt_metrics(original_lp, x_org, y_org, z_org)

        report = (
            file=abspath(file),
            rule=rule,
            fired=fired,
            status=results.status,
            m0=Int(rec.m0),
            n0=Int(rec.n0),
            m1=Int(rec.m1),
            n1=Int(rec.n1),
            presolve_time=results.presolve_time,
            postsolve_time=results.postsolve_time,
            solve_time=results.time,
            affected_rows=length(affected_rows),
            targeted_stats=stats,
            original_primal_obj=p_obj,
            original_dual_obj=d_obj,
            original_p_feas=p_feas,
            original_d_feas=d_feas,
            original_gap=gap,
            delta_y=delta_y,
            delta_z=delta_z,
            has_nonfinite=_has_nonfinite_entries(x_org) ||
                          _has_nonfinite_entries(y_org) ||
                          _has_nonfinite_entries(z_org),
        )
    finally
        _cleanup_rule_validation_gpu_memory!()
    end
    return report
end

function rule_toggle_match(
    status_on::AbstractString,
    status_off::AbstractString,
    obj_on::Real,
    obj_off::Real;
    atol::Float64=1e-4,
    rtol::Float64=1e-6,
)
    return status_on == status_off && isapprox(obj_on, obj_off; atol=atol, rtol=rtol)
end

function _values_changed(a, b; atol::Float64=1e-12, rtol::Float64=1e-9)
    length(a) == length(b) || return true
    isempty(a) && return false
    return !all(isapprox.(a, b; atol=atol, rtol=rtol))
end

function isolated_rule_fired(
    lp0::LP_info_cpu,
    lp_red::LP_info_gpu,
    rec::PresolveRecord_gpu;
    atol::Float64=1e-12,
    rtol::Float64=1e-9,
)
    if Int(rec.m0) != Int(rec.m1) || Int(rec.n0) != Int(rec.n1)
        return true
    end

    return _values_changed(Array(lp_red.l), lp0.l; atol=atol, rtol=rtol) ||
           _values_changed(Array(lp_red.u), lp0.u; atol=atol, rtol=rtol) ||
           _values_changed(Array(lp_red.AL), lp0.AL; atol=atol, rtol=rtol) ||
           _values_changed(Array(lp_red.AU), lp0.AU; atol=atol, rtol=rtol) ||
           _values_changed(Array(lp_red.c), lp0.c; atol=atol, rtol=rtol) ||
           !isapprox(lp_red.obj_constant, lp0.obj_constant; atol=atol, rtol=rtol)
end

function rule_toggle_fired(res)
    if hasproperty(res, :fired)
        return Bool(getproperty(res, :fired))
    end
    return res.m0_on != res.m1_on ||
           res.n0_on != res.n1_on ||
           res.m1_on != res.m1_off ||
           res.n1_on != res.n1_off
end

function rule_validation_report_path(rule::Symbol)
    return abspath(joinpath(@__DIR__, "..", "..", "docs", "validation", "$(String(rule))-validation.md"))
end

function write_rule_validation_report(rule::Symbol, results)
    report_path = rule_validation_report_path(rule)
    csv_path = replace(report_path, ".md" => ".csv")
    mkpath(dirname(report_path))

    passed = count(r -> r.passed, results)
    fired = count(rule_toggle_fired, results)
    total = length(results)

    open(report_path, "w") do io
        println(io, "# Rule Validation Report: $(String(rule))")
        println(io)
        println(io, "- Total cases: $total")
        println(io, "- Passed cases: $passed")
        println(io, "- Fired cases: $fired")
        println(io)
        println(io, "| file | passed | fired | status_on | status_off | primal_obj_on | primal_obj_off | m_on | n_on | m_off | n_off |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for res in results
            println(io, "| $(basename(res.file)) | $(res.passed) | $(rule_toggle_fired(res)) | $(res.status_on) | $(res.status_off) | $(res.primal_obj_on) | $(res.primal_obj_off) | $(res.m0_on)->$(res.m1_on) | $(res.n0_on)->$(res.n1_on) | $(res.m0_off)->$(res.m1_off) | $(res.n0_off)->$(res.n1_off) |")
        end
    end

    CSV.write(csv_path, DataFrame(results))
    return (markdown=report_path, csv=csv_path)
end
