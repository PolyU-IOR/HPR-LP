"""
Dedicated QP GPU presolve skeleton.

This file will host the QP-native scheduler and rules so that LP and QP
presolve can evolve independently while still sharing low-level sparse helpers.
"""

using CUDA.CUSPARSE: sort_csr

mutable struct QPresolveStats_gpu
    row_nnz::CuVector{Int32}
    col_nnz::CuVector{Int32}
    empty_col_mask::CuVector{UInt8}
    singleton_col_mask::CuVector{UInt8}
    singleton_col_row::CuVector{Int32}
    singleton_col_val::CuVector{Float64}
    q_offdiag_nnz::CuVector{Int32}
end

function QPresolveStats_gpu(m::Integer, n::Integer)
    return QPresolveStats_gpu(
        CUDA.zeros(Int32, Int(m)),
        CUDA.zeros(Int32, Int(n)),
        CUDA.zeros(UInt8, Int(n)),
        CUDA.zeros(UInt8, Int(n)),
        CUDA.fill(Int32(-1), Int(n)),
        CUDA.zeros(Float64, Int(n)),
        CUDA.zeros(Int32, Int(n)),
    )
end

mutable struct QPresolvePlan_gpu
    keep_row_mask::CuVector{UInt8}
    keep_col_mask::CuVector{UInt8}
    new_A::Union{Nothing,CuSparseMatrixCSR{Float64,Int32}}
    new_Q::Union{Nothing,CuSparseMatrixCSR{Float64,Int32}}
    new_c::CuVector{Float64}
    new_AL::CuVector{Float64}
    new_AU::CuVector{Float64}
    new_l::CuVector{Float64}
    new_u::CuVector{Float64}
    new_q_diag::CuVector{Float64}
    obj_constant_delta::Float64
    fixed_idx::CuVector{Int32}
    fixed_val::CuVector{Float64}
    singleton_col_row_idx::CuVector{Int32}
    singleton_col_col_idx::CuVector{Int32}
    merged_col_from::CuVector{Int32}
    merged_col_to::CuVector{Int32}
    merged_col_ratio::CuVector{Float64}
    merged_col_from_l::CuVector{Float64}
    merged_col_from_u::CuVector{Float64}
    merged_col_to_l::CuVector{Float64}
    merged_col_to_u::CuVector{Float64}
    tape::PostsolveTape
    tape_gpu::PostsolveTape_gpu
    has_change::Bool
    has_infeasible::Bool
    has_unbounded::Bool
    status_message::String
end

function QPresolvePlan_gpu(qp::QP_info_gpu)
    m, n = size(qp.A)
    return QPresolvePlan_gpu(
        CUDA.fill(UInt8(1), Int(m)),
        CUDA.fill(UInt8(1), Int(n)),
        nothing,
        nothing,
        copy(qp.c),
        copy(qp.AL),
        copy(qp.AU),
        copy(qp.l),
        copy(qp.u),
        copy(qp.q_diag),
        0.0,
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        CuVector{Float64}(undef, 0),
        PostsolveTape(),
        PostsolveTape_gpu(),
        false,
        false,
        false,
        "",
    )
end

function append_plan_fixed_from_mask!(
    plan::QPresolvePlan_gpu,
    fixed_mask::CuVector{UInt8},
    fixed_val::CuVector{Float64},
)
    _, fixed_idx, fixed_count = build_maps_from_mask(fixed_mask)
    if Int(fixed_count) == 0
        return false
    end

    fixed_sel = gather_by_red2org(fixed_val, fixed_idx)
    plan.fixed_idx = concat_cuvector(plan.fixed_idx, fixed_idx)
    plan.fixed_val = concat_cuvector(plan.fixed_val, fixed_sel)
    return true
end

function _kernel_compute_q_offdiag_nnz!(
    q_offdiag_nnz,
    row_ptr,
    col_val,
    nz_val,
    zero_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        start_ptr = row_ptr[j]
        stop_ptr = row_ptr[j + 1] - Int32(1)
        count = Int32(0)
        for p in start_ptr:stop_ptr
            col = col_val[p]
            val = nz_val[p]
            if col != j && abs(val) > zero_tol
                count += Int32(1)
            end
        end
        @inbounds q_offdiag_nnz[j] = count
    end
    return
end

function compute_q_offdiag_nnz!(
    q_offdiag_nnz::CuVector{Int32},
    QT_csr::CuSparseMatrixCSR{Float64,Int32},
    zero_tol::Float64,
)
    n, _ = size(QT_csr)
    @assert length(q_offdiag_nnz) == n
    n == 0 && return nothing
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_q_offdiag_nnz!(
        q_offdiag_nnz,
        QT_csr.rowPtr,
        QT_csr.colVal,
        QT_csr.nzVal,
        zero_tol,
        Int32(n),
    )
    return nothing
end

function _kernel_extract_q_diag!(
    q_diag,
    row_ptr,
    col_val,
    nz_val,
    n,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        diag_val = 0.0
        @inbounds start_ptr = row_ptr[i]
        @inbounds stop_ptr = row_ptr[i + Int32(1)] - Int32(1)
        if start_ptr <= stop_ptr
            for p in start_ptr:stop_ptr
                @inbounds col = col_val[p]
                if col == i
                    @inbounds diag_val += nz_val[p]
                end
            end
        end
        @inbounds q_diag[i] = diag_val
    end
    return
end

"""
Extract the diagonal from the actual GPU CSR `Q`.

QP-native rules can update `Q` structurally through sparse delta matrices.  The
derived `q_diag` cache must therefore be rebuilt from the post-update matrix
before the next rule computes Q-aware statistics or one-dimensional objectives.
"""
function extract_q_diag(Q_csr::CuSparseMatrixCSR{Float64,Int32})
    n, ncols = size(Q_csr)
    @assert n == ncols
    q_diag = CUDA.zeros(Float64, n)
    n == 0 && return q_diag
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_extract_q_diag!(
        q_diag,
        Q_csr.rowPtr,
        Q_csr.colVal,
        Q_csr.nzVal,
        Int32(n),
    )
    return q_diag
end

function presolve_compute_qp_stats(
    qp::QP_info_gpu,
    pparams::PresolveParams,
)
    m, n = size(qp.A)
    stats = QPresolveStats_gpu(m, n)
    compute_row_nnz!(stats.row_nnz, qp.A)
    compute_col_nnz!(stats.col_nnz, qp.AT)
    stats.empty_col_mask .= UInt8.(stats.col_nnz .== Int32(0))
    stats.singleton_col_mask .= UInt8.(stats.col_nnz .== Int32(1))
    compute_singleton_col_support!(
        stats.singleton_col_row,
        stats.singleton_col_val,
        stats.col_nnz,
        qp.AT,
    )
    compute_q_offdiag_nnz!(stats.q_offdiag_nnz, qp.QT, pparams.zero_tol)
    return stats
end

function presolve_reset_qp_plan(qp::QP_info_gpu, _pparams::PresolveParams)
    return QPresolvePlan_gpu(qp)
end

function _kernel_qp_fill_identity_map!(out, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds out[i] = Int32(i)
    end
    return
end

function _qp_identity_map_gpu(len::Integer)
    n = Int(len)
    out = CuVector{Int32}(undef, n)
    n == 0 && return out
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_fill_identity_map!(out, Int32(n))
    return out
end

function _kernel_qp_delta_count_rows!(
    row_counts,
    delta_rows,
    nnz,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= nnz
        @inbounds row = delta_rows[t]
        CUDA.@atomic row_counts[row] += Int32(1)
    end
    return
end

function _kernel_qp_delta_scatter_csr!(
    csr_cols,
    csr_vals,
    row_write_ptr,
    delta_rows,
    delta_cols,
    delta_vals,
    nnz,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= nnz
        @inbounds row = delta_rows[t]
        @inbounds slot = CUDA.@atomic row_write_ptr[row] += Int32(1)
        @inbounds csr_cols[slot] = delta_cols[t]
        @inbounds csr_vals[slot] = delta_vals[t]
    end
    return
end

"""
Build a CSR delta matrix directly from GPU triplet buffers.

This mirrors the LP doubleton-equality direct CSR path: count entries per row,
prefix-sum row offsets, scatter entries into CSR storage, then sort CSR rows.
It intentionally avoids the slower COO-to-CSR conversion path.
"""
function _qp_build_delta_csr_direct(
    delta_rows::CuVector{Int32},
    delta_cols::CuVector{Int32},
    delta_vals::CuVector{Float64},
    dims::Tuple{Int,Int},
)
    nnz = length(delta_rows)
    nnz == length(delta_cols) == length(delta_vals) || error("QP delta CSR payload length mismatch.")
    m, n = dims
    if nnz == 0
        row_ptr = CUDA.fill(Int32(1), m + 1)
        return CuSparseMatrixCSR(
            row_ptr,
            CuVector{Int32}(undef, 0),
            CuVector{Float64}(undef, 0),
            (m, n),
        )
    end

    blocks = cld(nnz, GPU_PRESOLVE_THREADS)
    row_counts = CUDA.zeros(Int32, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_delta_count_rows!(
        row_counts,
        delta_rows,
        Int32(nnz),
    )

    prefix = cumsum(row_counts)
    row_ptr = CUDA.fill(Int32(1), m + 1)
    row_ptr[2:end] .= prefix .+ Int32(1)

    csr_cols = CuVector{Int32}(undef, nnz)
    csr_vals = CuVector{Float64}(undef, nnz)
    row_write_ptr = copy(row_ptr[1:end-1])
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_delta_scatter_csr!(
        csr_cols,
        csr_vals,
        row_write_ptr,
        delta_rows,
        delta_cols,
        delta_vals,
        Int32(nnz),
    )

    return sort_csr(CuSparseMatrixCSR(row_ptr, csr_cols, csr_vals, (m, n)))
end

const _PRESOLVE_QP_RULES_DIR = joinpath(@__DIR__, "rules", "qp_rules")

include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_empty_cols.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_singleton_cols_dual_infer.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_singleton_cols_eq.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_parallel_cols.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_dual_fix.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_doubleton_eq.jl"))
include(joinpath(_PRESOLVE_QP_RULES_DIR, "qp_rule_linear_eq_agg.jl"))

function _qp_lp_view(qp::QP_info_gpu)
    return LP_info_gpu(
        qp.A,
        qp.AT,
        qp.c,
        qp.AL,
        qp.AU,
        qp.l,
        qp.u,
        qp.obj_constant,
        qp.AT_leading_slack,
        qp.AT_slack_after,
    )
end

function _qp_from_safe_lp_row_phase(qp_old::QP_info_gpu, lp_new::LP_info_gpu)
    n_old = size(qp_old.Q, 1)
    n_new = size(lp_new.A, 2)
    if n_new != n_old
        error(
            "QP row phase removed columns (n=$n_old->$n_new). " *
            "That requires a Q-aware column update and is not allowed in the shared row phase.",
        )
    end

    return QP_info_gpu(
        lp_new.A,
        lp_new.AT,
        qp_old.Q,
        qp_old.QT,
        lp_new.c,
        qp_old.q_diag,
        lp_new.AL,
        lp_new.AU,
        lp_new.l,
        lp_new.u,
        lp_new.obj_constant,
        lp_new.AT_leading_slack,
        lp_new.AT_slack_after,
    )
end

function _kernel_qp_scatter_fixed_values!(
    fixed_full,
    fixed_idx,
    fixed_val,
    fixed_count,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= fixed_count
        @inbounds col = fixed_idx[t]
        @inbounds fixed_full[col] = fixed_val[t]
    end
    return
end

function _qp_fixed_full_vector(n::Integer, fixed_idx::CuVector{Int32}, fixed_val::CuVector{Float64})
    fixed_full = CUDA.zeros(Float64, Int(n))
    fixed_count = length(fixed_idx)
    fixed_count == 0 && return fixed_full
    blocks = cld(fixed_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_scatter_fixed_values!(
        fixed_full,
        fixed_idx,
        fixed_val,
        Int32(fixed_count),
    )
    return fixed_full
end

function _qp_from_lp_phase_plan(
    qp_old::QP_info_gpu,
    lp_new::LP_info_gpu,
    rec_new::PresolveRecord_gpu,
    plan::PresolvePlan_gpu,
)
    n_old = size(qp_old.Q, 1)
    _, col_red2org_local, n_new = build_maps_from_mask(plan.keep_col_mask)
    fixed_count = length(plan.fixed_idx)

    if Int(n_new) == n_old
        return _qp_from_safe_lp_row_phase(qp_old, lp_new), rec_new
    end

    removed_cols = n_old - Int(n_new)
    if removed_cols != fixed_count
        error(
            "QP shared LP phase removed non-fixed columns (removed=$removed_cols, fixed=$fixed_count). " *
            "That rule needs a dedicated Q-aware implementation.",
        )
    end

    fixed_full = _qp_fixed_full_vector(n_old, plan.fixed_idx, plan.fixed_val)
    q_fixed_shift = qp_old.Q * fixed_full
    c_all = plan.new_c .+ q_fixed_shift
    c_new = gather_by_red2org(c_all, col_red2org_local)
    Q_new = compact_csr_by_rows_and_cols(qp_old.Q, col_red2org_local, col_red2org_local)
    QT_new = transpose_csr(Q_new)
    q_diag_new = extract_q_diag(Q_new)
    obj_new = qp_old.obj_constant + plan.obj_constant_delta + 0.5 * sum(fixed_full .* q_fixed_shift)

    qp_new = QP_info_gpu(
        lp_new.A,
        lp_new.AT,
        Q_new,
        QT_new,
        c_new,
        q_diag_new,
        lp_new.AL,
        lp_new.AU,
        lp_new.l,
        lp_new.u,
        obj_new,
        lp_new.AT_leading_slack,
        lp_new.AT_slack_after,
    )
    rec_fixed = _copy_record_with_updates(rec_new; obj_constant_new=obj_new)
    return qp_new, rec_fixed
end

function _run_qp_shared_lp_phase_subset(
    qp::QP_info_gpu,
    rec::PresolveRecord_gpu,
    pparams::PresolveParams;
    phase::Symbol,
    rule_order,
    start_time=nothing,
)
    isempty(rule_order) && return qp, rec, false
    _presolve_time_exceeded(start_time, pparams) && return (qp, rec, false)

    lp_view = _qp_lp_view(qp)
    subset_pparams = _subset_presolve_params(pparams; phase=phase, rule_order=rule_order)
    stats = presolve_compute_stats(lp_view, subset_pparams; phase=phase)
    plan = presolve_make_plan(lp_view, stats, subset_pparams; phase=phase)
    _throw_terminal_status_if_needed!(plan, phase)
    _phase_has_action(plan, phase) || return qp, rec, false

    lp_new, rec_new, changed = presolve_apply_plan(
        lp_view,
        plan,
        rec,
        subset_pparams;
        phase=phase,
    )
    changed || return qp, rec, false
    qp_new, rec_qp = _qp_from_lp_phase_plan(qp, lp_new, rec_new, plan)
    return qp_new, rec_qp, true
end

function _apply_one_qp_col_rule!(
    qp_cur::QP_info_gpu,
    rec_cur::PresolveRecord_gpu,
    presolve_params::PresolveParams,
    rule_name::Symbol,
    ;
    start_time=nothing,
)
    _presolve_time_exceeded(start_time, presolve_params) && return (qp_cur, rec_cur, false)
    if rule_name == :singleton_cols_dual_infer
        presolve_params.enable_singleton_cols_dual_infer || return qp_cur, rec_cur, false
        changed_any = false
        while true
            _presolve_time_exceeded(start_time, presolve_params) && break
            stats = presolve_compute_qp_stats(qp_cur, presolve_params)
            plan = presolve_reset_qp_plan(qp_cur, presolve_params)
            apply_rule_qp_singleton_cols_dual_infer!(plan, qp_cur, stats, presolve_params)
            if plan.has_infeasible
                error("GPU QP presolve col-phase detected INFEASIBLE: $(plan.status_message)")
            end
            if plan.has_unbounded
                error("GPU QP presolve col-phase detected UNBOUNDED: $(plan.status_message)")
            end
            plan.has_change || break
            qp_cur, rec_cur, _ = presolve_apply_qp_plan(
                qp_cur,
                plan,
                rec_cur,
                presolve_params;
                rule_name=:singleton_cols_dual_infer,
            )
            changed_any = true
        end
        return qp_cur, rec_cur, changed_any
    elseif rule_name == :singleton_cols_eq
        presolve_params.enable_singleton_cols_eq || return qp_cur, rec_cur, false
        changed_any = false
        while true
            _presolve_time_exceeded(start_time, presolve_params) && break
            stats = presolve_compute_qp_stats(qp_cur, presolve_params)
            plan = presolve_reset_qp_plan(qp_cur, presolve_params)
            apply_rule_qp_singleton_cols_eq!(plan, qp_cur, stats, presolve_params)
            if plan.has_infeasible
                error("GPU QP presolve col-phase detected INFEASIBLE: $(plan.status_message)")
            end
            if plan.has_unbounded
                error("GPU QP presolve col-phase detected UNBOUNDED: $(plan.status_message)")
            end
            plan.has_change || break
            qp_cur, rec_cur, _ = presolve_apply_qp_plan(
                qp_cur,
                plan,
                rec_cur,
                presolve_params;
                rule_name=:singleton_cols_eq,
            )
            changed_any = true
        end
        return qp_cur, rec_cur, changed_any
    end

    if rule_name == :close_bounds
        presolve_params.enable_close_bounds || return qp_cur, rec_cur, false
        return _run_qp_shared_lp_phase_subset(
            qp_cur,
            rec_cur,
            presolve_params;
            phase=:col,
            rule_order=(:close_bounds,),
            start_time=start_time,
        )
    elseif rule_name == :empty_cols
        presolve_params.enable_empty_cols || return qp_cur, rec_cur, false
        stats = presolve_compute_qp_stats(qp_cur, presolve_params)
        plan = presolve_reset_qp_plan(qp_cur, presolve_params)
        apply_rule_qp_empty_cols!(plan, qp_cur, stats, presolve_params)
    elseif rule_name == :dual_fix
        presolve_params.enable_dual_fix || return qp_cur, rec_cur, false
        stats = presolve_compute_qp_stats(qp_cur, presolve_params)
        plan = presolve_reset_qp_plan(qp_cur, presolve_params)
        apply_rule_qp_dual_fix!(plan, qp_cur, stats, presolve_params)
    elseif rule_name == :parallel_cols
        presolve_params.enable_parallel_cols || return qp_cur, rec_cur, false
        stats = presolve_compute_qp_stats(qp_cur, presolve_params)
        plan = presolve_reset_qp_plan(qp_cur, presolve_params)
        apply_rule_qp_parallel_cols!(plan, qp_cur, stats, presolve_params)
    elseif rule_name == :doubleton_eq
        presolve_params.enable_doubleton_eq || return qp_cur, rec_cur, false
        stats = presolve_compute_qp_stats(qp_cur, presolve_params)
        plan = presolve_reset_qp_plan(qp_cur, presolve_params)
        apply_rule_qp_doubleton_eq!(plan, qp_cur, stats, presolve_params)
    elseif rule_name == :linear_eq_agg
        presolve_params.enable_linear_eq_agg || return qp_cur, rec_cur, false
        stats = presolve_compute_qp_stats(qp_cur, presolve_params)
        plan = presolve_reset_qp_plan(qp_cur, presolve_params)
        apply_rule_qp_linear_eq_agg!(plan, qp_cur, stats, presolve_params)
    else
        return qp_cur, rec_cur, false
    end

    if plan.has_infeasible
        error("GPU QP presolve col-phase detected INFEASIBLE: $(plan.status_message)")
    end
    if plan.has_unbounded
        error("GPU QP presolve col-phase detected UNBOUNDED: $(plan.status_message)")
    end
    plan.has_change || return qp_cur, rec_cur, false
    return presolve_apply_qp_plan(
        qp_cur,
        plan,
        rec_cur,
        presolve_params;
        rule_name=rule_name,
    )
end

function presolve_apply_qp_plan(
    qp::QP_info_gpu,
    plan::QPresolvePlan_gpu,
    rec::PresolveRecord_gpu,
    _pparams::PresolveParams;
    rule_name::Symbol=:empty_cols,
)
    n_old = size(qp.A, 2)
    m_old = size(qp.A, 1)
    keep_col = plan.keep_col_mask
    _, col_red2org_local, n_new = build_maps_from_mask(keep_col)
    removed = n_old - Int(n_new)
    if removed == 0
        A_keep = isnothing(plan.new_A) ? qp.A : plan.new_A
        AT_keep = isnothing(plan.new_A) ? qp.AT : transpose_csr(A_keep)
        Q_keep = isnothing(plan.new_Q) ? qp.Q : plan.new_Q
        QT_keep = isnothing(plan.new_Q) ? qp.QT : transpose_csr(Q_keep)
        q_diag_keep = isnothing(plan.new_Q) ? copy(plan.new_q_diag) : extract_q_diag(Q_keep)
        qp_new = QP_info_gpu(
            A_keep,
            AT_keep,
            Q_keep,
            QT_keep,
            copy(plan.new_c),
            q_diag_keep,
            copy(plan.new_AL),
            copy(plan.new_AU),
            copy(plan.new_l),
            copy(plan.new_u),
            qp.obj_constant + plan.obj_constant_delta,
            qp.AT_leading_slack,
            copy(qp.AT_slack_after),
        )
        rec_new = _copy_record_with_updates(
            rec;
            obj_constant_new=qp_new.obj_constant,
            tape=_merge_postsolve_tape(rec.tape, plan.tape, rec.row_red2org, rec.col_red2org),
            tape_gpu=nothing,
            tape_gpu_parts=_merge_postsolve_tape_gpu_parts(rec.tape_gpu_parts, plan.tape, plan.tape_gpu, rec.row_red2org, rec.col_red2org),
        )
        return qp_new, rec_new, plan.has_change
    end

    row_red2org_local, m_new = if rule_name in (:doubleton_eq, :singleton_cols_eq, :linear_eq_agg)
        _, rows, row_count = build_maps_from_mask(plan.keep_row_mask)
        rows, row_count
    else
        _qp_identity_map_gpu(m_old), Int32(m_old)
    end
    A_source = isnothing(plan.new_A) ? qp.A : plan.new_A
    Q_source = isnothing(plan.new_Q) ? qp.Q : plan.new_Q
    A_new = compact_csr_by_rows_and_cols(A_source, row_red2org_local, col_red2org_local)
    AT_new = transpose_csr(A_new)
    Q_new = compact_csr_by_rows_and_cols(Q_source, col_red2org_local, col_red2org_local)
    QT_new = transpose_csr(Q_new)

    c_new = gather_by_red2org(plan.new_c, col_red2org_local)
    q_diag_new = extract_q_diag(Q_new)
    AL_new = gather_by_red2org(plan.new_AL, row_red2org_local)
    AU_new = gather_by_red2org(plan.new_AU, row_red2org_local)
    l_new = gather_by_red2org(plan.new_l, col_red2org_local)
    u_new = gather_by_red2org(plan.new_u, col_red2org_local)
    obj_new = qp.obj_constant + plan.obj_constant_delta

    qp_new = QP_info_gpu(
        A_new,
        AT_new,
        Q_new,
        QT_new,
        c_new,
        q_diag_new,
        AL_new,
        AU_new,
        l_new,
        u_new,
        obj_new,
        Int32(0),
        CUDA.zeros(Int32, Int(n_new)),
    )

    row_red2org_global, row_org2red_global, removed_rows_global = if rule_name in (:doubleton_eq, :singleton_cols_eq, :linear_eq_agg)
        rows_global = compose_red2org(rec.row_red2org, row_red2org_local)
        org_global = build_org2red_from_red2org(rows_global, Int(rec.m0))
        removed_global = _collect_removed_global_indices_gpu(plan.keep_row_mask, rec.row_red2org)
        rows_global, org_global, removed_global
    else
        rec.row_red2org, rec.row_org2red, CuVector{Int32}(undef, 0)
    end
    col_red2org_global = compose_red2org(rec.col_red2org, col_red2org_local)
    col_org2red_global = build_org2red_from_red2org(col_red2org_global, Int(rec.n0))
    removed_cols_global = _collect_removed_global_indices_gpu(plan.keep_col_mask, rec.col_red2org)
    fixed_idx_global = _map_local_to_global_indices(plan.fixed_idx, rec.col_red2org)
    merged_from_global = _map_local_to_global_indices(plan.merged_col_from, rec.col_red2org)
    merged_to_global = _map_local_to_global_indices(plan.merged_col_to, rec.col_red2org)
    tape_new = _merge_postsolve_tape(rec.tape, plan.tape, rec.row_red2org, rec.col_red2org)
    tape_gpu_parts_new = _merge_postsolve_tape_gpu_parts(rec.tape_gpu_parts, plan.tape, plan.tape_gpu, rec.row_red2org, rec.col_red2org)

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
        _append_cuvector(rec.singleton_col_row_idx, _map_local_to_global_indices(plan.singleton_col_row_idx, rec.row_red2org)),
        _append_cuvector(rec.singleton_col_col_idx, _map_local_to_global_indices(plan.singleton_col_col_idx, rec.col_red2org)),
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
        nothing,
        tape_gpu_parts_new,
        copy(rec.structural_primal_recoveries),
    )

    change_count = if !isempty(plan.merged_col_from)
        Int(length(plan.merged_col_from))
    elseif !isempty(plan.singleton_col_col_idx)
        Int(length(plan.singleton_col_col_idx))
    elseif rule_name in (:doubleton_eq, :linear_eq_agg)
        Int(length(removed_rows_global))
    else
        Int(length(plan.fixed_idx))
    end
    rec_new.rule_counters[rule_name] = get(rec_new.rule_counters, rule_name, 0) + change_count
    return qp_new, rec_new, true
end

const QP_TIERED_CLEANUP_ROW_RULES = (
    :empty_rows,
    :singleton_rows,
    :activity_checks,
    :primal_propagation,
    :parallel_rows,
)
const QP_TIERED_CLEANUP_COL_RULES = (
    :close_bounds,
    :empty_cols,
    :singleton_cols_dual_infer,
    :dual_fix,
    :parallel_cols,
)
const QP_TIERED_FAST_COL_RULES = (
    :doubleton_eq,
    :singleton_cols_eq,
    :linear_eq_agg,
)

@inline function _qp_filter_enabled_rules(
    pparams::PresolveParams,
    phase::Symbol,
    candidate_rules,
)
    active_order = phase == :row ? pparams.row_rule_order : pparams.col_rule_order
    return Tuple(rule for rule in active_order if rule in candidate_rules && _is_rule_enabled(pparams, rule))
end

@inline function _qp_stats_is_diagonal(stats::QPresolveStats_gpu)
    return length(stats.q_offdiag_nnz) == 0 || Int(sum(stats.q_offdiag_nnz)) == 0
end

function _run_qp_col_rule_list!(
    qp_cur::QP_info_gpu,
    rec_cur::PresolveRecord_gpu,
    presolve_params::PresolveParams,
    rule_order;
    start_time::Float64,
)
    changed_any = false
    for rule_name in rule_order
        _presolve_time_exceeded(start_time, presolve_params) && break
        qp_cur, rec_cur, changed = _apply_one_qp_col_rule!(
            qp_cur,
            rec_cur,
            presolve_params,
            rule_name,
            start_time=start_time,
        )
        changed_any |= changed
    end
    return qp_cur, rec_cur, changed_any
end

function _run_qp_tiered_diagonal_loop(
    qp::QP_info_gpu,
    rec::PresolveRecord_gpu,
    presolve_params::PresolveParams;
    start_time::Float64,
)
    qp_cur = qp
    rec_cur = rec
    changed_iter = false

    cleanup_row_rules = _qp_filter_enabled_rules(presolve_params, :row, QP_TIERED_CLEANUP_ROW_RULES)
    cleanup_col_rules = _qp_filter_enabled_rules(presolve_params, :col, QP_TIERED_CLEANUP_COL_RULES)
    fast_col_rules = _qp_filter_enabled_rules(presolve_params, :col, QP_TIERED_FAST_COL_RULES)

    if !isempty(cleanup_row_rules)
        qp_cur, rec_cur, changed = _run_qp_shared_lp_phase_subset(
            qp_cur,
            rec_cur,
            presolve_params;
            phase=:row,
            rule_order=cleanup_row_rules,
            start_time=start_time,
        )
        changed_iter |= changed
    end
    _presolve_time_exceeded(start_time, presolve_params) && return qp_cur, rec_cur, changed_iter

    if !isempty(cleanup_col_rules)
        qp_cur, rec_cur, changed = _run_qp_col_rule_list!(
            qp_cur,
            rec_cur,
            presolve_params,
            cleanup_col_rules;
            start_time=start_time,
        )
        changed_iter |= changed
    end
    _presolve_time_exceeded(start_time, presolve_params) && return qp_cur, rec_cur, changed_iter

    while true
        _presolve_time_exceeded(start_time, presolve_params) && break
        changed_fast = false
        if !isempty(fast_col_rules)
            qp_cur, rec_cur, changed = _run_qp_col_rule_list!(
                qp_cur,
                rec_cur,
                presolve_params,
                fast_col_rules;
                start_time=start_time,
            )
            changed_fast |= changed
            changed_iter |= changed
        end

        changed_fast || break
        _presolve_time_exceeded(start_time, presolve_params) && break

        if !isempty(cleanup_row_rules)
            qp_cur, rec_cur, changed = _run_qp_shared_lp_phase_subset(
                qp_cur,
                rec_cur,
                presolve_params;
                phase=:row,
                rule_order=cleanup_row_rules,
                start_time=start_time,
            )
            changed_iter |= changed
        end
        _presolve_time_exceeded(start_time, presolve_params) && break

        if !isempty(cleanup_col_rules)
            qp_cur, rec_cur, changed = _run_qp_col_rule_list!(
                qp_cur,
                rec_cur,
                presolve_params,
                cleanup_col_rules;
                start_time=start_time,
            )
            changed_iter |= changed
        end
    end

    return qp_cur, rec_cur, changed_iter
end

"""
QP-native GPU presolve entrypoint.

This currently wires the native separable empty-column reduction into the
dedicated QP backend. Additional QP-native rules will be migrated here
incrementally.
"""
function presolve_gpu(
    qp::QP_info_gpu;
    presolve_params::PresolveParams=PresolveParams(),
    verbose::Bool=false,
)
    m0, n0 = size(qp.A)
    presolve_params.verbose = presolve_params.verbose || verbose
    rec = presolve_identity_record(m0, n0, qp.obj_constant)
    qp_cur = qp
    rec_cur = rec

    if presolve_params.verbose
        println(">>> [GPU QP Presolve] start (m=$m0, n=$n0)")
    end

    t_start = time()
    for iter in 1:presolve_params.max_iters
        _presolve_time_exceeded(t_start, presolve_params) && break
        changed_iter = false
        _validate_presolve_rule_orders(presolve_params)
        scheduler = presolve_params.gpu_presolve_scheduler
        stats0 = presolve_compute_qp_stats(qp_cur, presolve_params)
        diagonal_q = _qp_stats_is_diagonal(stats0)

        if scheduler == :tiered && diagonal_q
            if presolve_params.verbose
                println(">>> [GPU QP Presolve tiered] iter=$iter, q_structure=diagonal, mode=cleanup+fast")
            end
            qp_cur, rec_cur, changed_iter = _run_qp_tiered_diagonal_loop(
                qp_cur,
                rec_cur,
                presolve_params;
                start_time=t_start,
            )
        else
            qp_cur, rec_cur, changed = _run_qp_shared_lp_phase_subset(
                qp_cur,
                rec_cur,
                presolve_params;
                phase=:row,
                rule_order=presolve_params.row_rule_order,
                start_time=t_start,
            )
            changed_iter |= changed
            _presolve_time_exceeded(t_start, presolve_params) && break

            for rule_name in presolve_params.col_rule_order
                _presolve_time_exceeded(t_start, presolve_params) && break
                qp_cur, rec_cur, changed = _apply_one_qp_col_rule!(
                    qp_cur,
                    rec_cur,
                    presolve_params,
                    rule_name,
                    start_time=t_start,
                )
                changed_iter |= changed
            end
        end

        if presolve_params.verbose
            m_now, n_now = size(qp_cur.A)
            println(
                ">>> [GPU QP Presolve] iter=$iter, scheduler=$scheduler, diagonal_q=$diagonal_q, changed=$changed_iter, dims=($m_now, $n_now)"
            )
        end

        (!changed_iter || _presolve_time_exceeded(t_start, presolve_params)) && break
    end

    if presolve_params.verbose
        m1, n1 = size(qp_cur.A)
        println(">>> [GPU QP Presolve] done (m=$m0->$m1, n=$n0->$n1)")
    end

    return qp_cur, rec_cur
end
