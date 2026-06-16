"""
QP layer-2 rule: doubleton equality elimination.

This file contains the Q-aware substitution, sparse `Q` delta construction, and
fill guard used by the QP-native doubleton equality rule.
"""

@inline function _qp_csr_entry_exists_device(
    row_ptr,
    col_val,
    nz_val,
    row,
    col,
    zero_tol,
)
    @inbounds start_ptr = row_ptr[row]
    @inbounds stop_ptr = row_ptr[row + Int32(1)] - Int32(1)
    if start_ptr <= stop_ptr
        for p in start_ptr:stop_ptr
            @inbounds current_col = col_val[p]
            if current_col == col
                @inbounds return abs(nz_val[p]) > zero_tol
            end
        end
    end
    return false
end

@inline function _qp_doubleton_q_fill_guard_device(
    q_row_ptr,
    q_col_val,
    q_nz_val,
    qt_row_ptr,
    qt_col_val,
    qt_nz_val,
    q_diag,
    elim_col,
    keep_col,
    alpha,
    zero_tol,
    max_fill_abs,
    max_fill_ratio,
)
    fill_count = Int32(0)
    incident_count = Int32(0)
    q_keep_elim = 0.0
    q_elim_keep = 0.0
    @inbounds q_elim_elim = q_diag[elim_col]

    @inbounds col_start = qt_row_ptr[elim_col]
    @inbounds col_stop = qt_row_ptr[elim_col + Int32(1)] - Int32(1)
    if col_start <= col_stop
        for p in col_start:col_stop
            @inbounds row = qt_col_val[p]
            @inbounds val = qt_nz_val[p]
            if row == keep_col
                q_keep_elim = val
            elseif row != elim_col && abs(val) > zero_tol
                incident_count += Int32(1)
                if !_qp_csr_entry_exists_device(
                    q_row_ptr,
                    q_col_val,
                    q_nz_val,
                    row,
                    keep_col,
                    zero_tol,
                )
                    fill_count += Int32(1)
                end
            end
        end
    end

    @inbounds row_start = q_row_ptr[elim_col]
    @inbounds row_stop = q_row_ptr[elim_col + Int32(1)] - Int32(1)
    if row_start <= row_stop
        for p in row_start:row_stop
            @inbounds col = q_col_val[p]
            @inbounds val = q_nz_val[p]
            if col == keep_col
                q_elim_keep = val
            elseif col != elim_col && abs(val) > zero_tol
                incident_count += Int32(1)
                if !_qp_csr_entry_exists_device(
                    q_row_ptr,
                    q_col_val,
                    q_nz_val,
                    keep_col,
                    col,
                    zero_tol,
                )
                    fill_count += Int32(1)
                end
            end
        end
    end

    diag_delta = alpha * q_keep_elim + alpha * q_elim_keep + alpha * alpha * q_elim_elim
    if abs(diag_delta) > zero_tol
        incident_count += Int32(1)
        if !_qp_csr_entry_exists_device(
            q_row_ptr,
            q_col_val,
            q_nz_val,
            keep_col,
            keep_col,
            zero_tol,
        )
            fill_count += Int32(1)
        end
    end

    if max_fill_abs >= Int32(0) && fill_count > max_fill_abs
        return false
    end
    if isfinite(max_fill_ratio) && max_fill_ratio >= 0.0
        return Float64(fill_count) <= max_fill_ratio * Float64(max(incident_count, Int32(1)))
    end
    return true
end

function _kernel_mark_qp_doubleton_eq_batch_acceptable!(
    acceptable_mask,
    candidate_mask,
    candidate_keep_col,
    candidate_elim_col,
    candidate_keep_val,
    candidate_elim_val,
    candidate_elim_col_mask,
    col_nnz,
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    q_row_ptr,
    q_col_val,
    q_nz_val,
    qt_row_ptr,
    qt_col_val,
    qt_nz_val,
    q_diag,
    q_offdiag_nnz,
    require_qdiag_zero,
    zero_tol,
    max_fill_in_proxy,
    max_q_fill_abs,
    max_q_fill_ratio,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(candidate_mask[i] != UInt8(0))
        keep_col = @inbounds candidate_keep_col[i]
        elim_col = @inbounds candidate_elim_col[i]
        @inbounds q_offdiag_nnz[elim_col] == Int32(0) || return
        if require_qdiag_zero != UInt8(0)
            @inbounds abs(q_diag[elim_col]) <= zero_tol || return
        end
        keep_nnz = @inbounds col_nnz[keep_col]
        elim_nnz = @inbounds col_nnz[elim_col]
        if elim_nnz > keep_nnz + max_fill_in_proxy + Int32(1)
            return
        end

        fill_in = _doubleton_eq_fill_in_proxy_device_live(
            AT_row_ptr,
            AT_col_val,
            AT_nz_val,
            keep_row,
            zero_tol,
            keep_col,
            elim_col,
        )
        fill_in <= max_fill_in_proxy || return

        @inbounds col_start = qt_row_ptr[elim_col]
        @inbounds col_stop = qt_row_ptr[elim_col + Int32(1)] - Int32(1)
        if col_start <= col_stop
            for p in col_start:col_stop
                @inbounds row = qt_col_val[p]
                if row != elim_col && candidate_elim_col_mask[row] != UInt8(0)
                    return
                end
            end
        end

        @inbounds row_start = q_row_ptr[elim_col]
        @inbounds row_stop = q_row_ptr[elim_col + Int32(1)] - Int32(1)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = q_col_val[p]
                if col != elim_col && candidate_elim_col_mask[col] != UInt8(0)
                    return
                end
            end
        end

        @inbounds alpha = -candidate_keep_val[i] / candidate_elim_val[i]
        _qp_doubleton_q_fill_guard_device(
            q_row_ptr,
            q_col_val,
            q_nz_val,
            qt_row_ptr,
            qt_col_val,
            qt_nz_val,
            q_diag,
            elim_col,
            keep_col,
            alpha,
            zero_tol,
            max_q_fill_abs,
            max_q_fill_ratio,
        ) || return

        @inbounds acceptable_mask[i] = UInt8(1)
    end
    return
end

function _kernel_mark_qp_doubleton_candidate_elim_cols!(
    candidate_elim_col_mask,
    candidate_mask,
    candidate_elim_col,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(candidate_mask[i] != UInt8(0))
        @inbounds candidate_elim_col_mask[candidate_elim_col[i]] = UInt8(1)
    end
    return
end

function _kernel_apply_qp_doubleton_eq_batch_subst_decisions!(
    l,
    u,
    row_delete,
    col_delete,
    selected_rows,
    keep_col,
    elim_col,
    keep_l_new,
    keep_u_new,
    keep_fixed_mask,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k && @inbounds(keep_fixed_mask[t] == UInt8(0))
        @inbounds row = selected_rows[t]
        @inbounds keep = keep_col[t]
        @inbounds elim = elim_col[t]
        @inbounds l[keep] = keep_l_new[t]
        @inbounds u[keep] = keep_u_new[t]
        @inbounds row_delete[row] = UInt8(1)
        @inbounds col_delete[elim] = UInt8(1)
    end
    return
end

function _kernel_qp_doubleton_q_batch_delta_counts!(
    entry_counts,
    diag_delta,
    subst_elim_col,
    subst_keep_col,
    subst_alpha,
    q_row_ptr,
    q_col_val,
    q_nz_val,
    qt_row_ptr,
    qt_col_val,
    qt_nz_val,
    q_diag,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds elim = subst_elim_col[t]
        @inbounds keep = subst_keep_col[t]
        @inbounds alpha = subst_alpha[t]
        count = Int32(0)
        q_keep_elim = 0.0
        q_elim_keep = 0.0
        @inbounds qee = q_diag[elim]

        @inbounds col_start = qt_row_ptr[elim]
        @inbounds col_stop = qt_row_ptr[elim + Int32(1)] - Int32(1)
        if col_start <= col_stop
            for p in col_start:col_stop
                @inbounds row = qt_col_val[p]
                @inbounds val = qt_nz_val[p]
                if row == keep
                    q_keep_elim = val
                elseif row != elim && abs(val) > zero_tol
                    count += Int32(1)
                end
            end
        end

        @inbounds row_start = q_row_ptr[elim]
        @inbounds row_stop = q_row_ptr[elim + Int32(1)] - Int32(1)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = q_col_val[p]
                @inbounds val = q_nz_val[p]
                if col == keep
                    q_elim_keep = val
                elseif col != elim && abs(val) > zero_tol
                    count += Int32(1)
                end
            end
        end

        d = alpha * q_keep_elim + alpha * q_elim_keep + alpha * alpha * qee
        @inbounds diag_delta[t] = d
        if abs(d) > zero_tol
            count += Int32(1)
        end
        @inbounds entry_counts[t] = count
    end
    return
end

function _kernel_qp_doubleton_q_batch_delta_build!(
    delta_rows,
    delta_cols,
    delta_vals,
    entry_starts,
    diag_delta,
    subst_elim_col,
    subst_keep_col,
    subst_alpha,
    q_row_ptr,
    q_col_val,
    q_nz_val,
    qt_row_ptr,
    qt_col_val,
    qt_nz_val,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds elim = subst_elim_col[t]
        @inbounds keep = subst_keep_col[t]
        @inbounds alpha = subst_alpha[t]
        @inbounds write_pos = entry_starts[t]

        @inbounds col_start = qt_row_ptr[elim]
        @inbounds col_stop = qt_row_ptr[elim + Int32(1)] - Int32(1)
        if col_start <= col_stop
            for p in col_start:col_stop
                @inbounds row = qt_col_val[p]
                @inbounds val = qt_nz_val[p]
                if row != elim && row != keep && abs(val) > zero_tol
                    @inbounds delta_rows[write_pos] = row
                    @inbounds delta_cols[write_pos] = keep
                    @inbounds delta_vals[write_pos] = alpha * val
                    write_pos += Int32(1)
                end
            end
        end

        @inbounds row_start = q_row_ptr[elim]
        @inbounds row_stop = q_row_ptr[elim + Int32(1)] - Int32(1)
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = q_col_val[p]
                @inbounds val = q_nz_val[p]
                if col != elim && col != keep && abs(val) > zero_tol
                    @inbounds delta_rows[write_pos] = keep
                    @inbounds delta_cols[write_pos] = col
                    @inbounds delta_vals[write_pos] = alpha * val
                    write_pos += Int32(1)
                end
            end
        end

        @inbounds d = diag_delta[t]
        if abs(d) > zero_tol
            @inbounds delta_rows[write_pos] = keep
            @inbounds delta_cols[write_pos] = keep
            @inbounds delta_vals[write_pos] = d
        end
    end
    return
end

function _kernel_qp_doubleton_batch_c_update!(
    c,
    obj_contrib,
    subst_elim_col,
    subst_keep_col,
    subst_elim_obj,
    subst_alpha,
    subst_beta,
    q_diag,
    qt_row_ptr,
    qt_col_val,
    qt_nz_val,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds elim = subst_elim_col[t]
        @inbounds keep = subst_keep_col[t]
        @inbounds alpha = subst_alpha[t]
        @inbounds beta = subst_beta[t]
        @inbounds elim_obj = subst_elim_obj[t]
        @inbounds qee = q_diag[elim]

        @inbounds start_idx = qt_row_ptr[elim]
        @inbounds stop_idx = qt_row_ptr[elim + Int32(1)] - Int32(1)
        if start_idx <= stop_idx
            for p in start_idx:stop_idx
                @inbounds row = qt_col_val[p]
                @inbounds val = qt_nz_val[p]
                if row != elim && abs(val) > zero_tol
                    CUDA.@atomic c[row] += beta * val
                end
            end
        end
        CUDA.@atomic c[keep] += alpha * (elim_obj + beta * qee)
        @inbounds obj_contrib[t] = elim_obj * beta + 0.5 * qee * beta * beta
    end
    return
end

function _qp_build_doubleton_q_batch_delta_csr(
    qp::QP_info_gpu,
    subst_elim_col::CuVector{Int32},
    subst_keep_col::CuVector{Int32},
    subst_alpha::CuVector{Float64},
    zero_tol::Float64,
)
    k = length(subst_elim_col)
    if k == 0
        return _qp_build_delta_csr_direct(
            CuVector{Int32}(undef, 0),
            CuVector{Int32}(undef, 0),
            CuVector{Float64}(undef, 0),
            size(qp.Q),
        )
    end

    blocks = cld(k, GPU_PRESOLVE_THREADS)
    entry_counts = CUDA.zeros(Int32, k)
    diag_delta = CUDA.zeros(Float64, k)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_doubleton_q_batch_delta_counts!(
        entry_counts,
        diag_delta,
        subst_elim_col,
        subst_keep_col,
        subst_alpha,
        qp.Q.rowPtr,
        qp.Q.colVal,
        qp.Q.nzVal,
        qp.QT.rowPtr,
        qp.QT.colVal,
        qp.QT.nzVal,
        qp.q_diag,
        zero_tol,
        Int32(k),
    )

    entry_prefix = cumsum(entry_counts)
    total_entries = Int(_copy_scalar_to_host(entry_prefix, k))
    if total_entries == 0
        return _qp_build_delta_csr_direct(
            CuVector{Int32}(undef, 0),
            CuVector{Int32}(undef, 0),
            CuVector{Float64}(undef, 0),
            size(qp.Q),
        )
    end

    entry_starts = CUDA.fill(Int32(1), k)
    if k > 1
        entry_starts[2:end] .= entry_prefix[1:(end - 1)] .+ Int32(1)
    end

    delta_rows = CuVector{Int32}(undef, total_entries)
    delta_cols = CuVector{Int32}(undef, total_entries)
    delta_vals = CuVector{Float64}(undef, total_entries)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_doubleton_q_batch_delta_build!(
        delta_rows,
        delta_cols,
        delta_vals,
        entry_starts,
        diag_delta,
        subst_elim_col,
        subst_keep_col,
        subst_alpha,
        qp.Q.rowPtr,
        qp.Q.colVal,
        qp.Q.nzVal,
        qp.QT.rowPtr,
        qp.QT.colVal,
        qp.QT.nzVal,
        zero_tol,
        Int32(k),
    )
    return _qp_build_delta_csr_direct(delta_rows, delta_cols, delta_vals, size(qp.Q))
end

function _append_qp_doubleton_eq_batch_postsolve_records!(
    plan::QPresolvePlan_gpu,
    pparams::PresolveParams,
    AT_source::CuSparseMatrixCSR{Float64,Int32},
    keep_row::CuVector{UInt8},
    subst_elim_col::CuVector{Int32},
    subst_rows::CuVector{Int32},
    subst_keep_col::CuVector{Int32},
    subst_keep_val::CuVector{Float64},
    subst_elim_val::CuVector{Float64},
    subst_rhs::CuVector{Float64},
    subst_old_elim_l::CuVector{Float64},
    subst_old_elim_u::CuVector{Float64},
    subst_old_keep_l::CuVector{Float64},
    subst_old_keep_u::CuVector{Float64},
    subst_new_keep_l::CuVector{Float64},
    subst_new_keep_u::CuVector{Float64},
    subst_elim_obj::CuVector{Float64},
    support_counts::CuVector{Int32},
    subst_lower_source::CuVector{Float64},
    subst_upper_source::CuVector{Float64},
    zero_tol::Float64,
)
    pparams.record_postsolve_tape || return nothing

    subst_count = length(subst_elim_col)
    subst_count == 0 && return nothing

    idx_lengths = support_counts .+ Int32(4)
    val_lengths = support_counts .+ Int32(12)
    idx_prefix = cumsum(idx_lengths)
    val_prefix = cumsum(val_lengths)
    total_idx = Int(_copy_scalar_to_host(idx_prefix, subst_count))
    total_val = Int(_copy_scalar_to_host(val_prefix, subst_count))

    idx_starts = CUDA.fill(Int32(1), subst_count + 1)
    val_starts = CUDA.fill(Int32(1), subst_count + 1)
    idx_starts[2:end] .= idx_prefix .+ Int32(1)
    val_starts[2:end] .= val_prefix .+ Int32(1)

    indices = CuVector{Int32}(undef, total_idx)
    vals = CuVector{Float64}(undef, total_val)
    blocks = cld(subst_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_doubleton_eq_batch_records_gpu!(
        indices,
        vals,
        idx_starts,
        val_starts,
        support_counts,
        subst_rows,
        subst_keep_col,
        subst_elim_col,
        subst_keep_val,
        subst_elim_val,
        subst_rhs,
        subst_old_elim_l,
        subst_old_elim_u,
        subst_old_keep_l,
        subst_old_keep_u,
        subst_new_keep_l,
        subst_new_keep_u,
        subst_elim_obj,
        subst_lower_source,
        subst_upper_source,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        keep_row,
        zero_tol,
        Int32(subst_count),
    )

    record = PostsolveTape_gpu(
        CUDA.fill(Int32(DOUBLETON_EQ), subst_count),
        idx_starts,
        val_starts,
        CUDA.fill(UInt8(POSTSOLVE_DUAL_MINIMAL), subst_count),
        indices,
        vals,
    )
    append_postsolve_tape!(plan.tape_gpu, record)
    if pparams.record_postsolve_tape_cpu
        plan.tape = PostsolveTape(plan.tape_gpu)
    end
    return nothing
end

function _apply_rule_qp_doubleton_eq_batch!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    m = length(plan.keep_row_mask)
    n = length(plan.keep_col_mask)
    if m == 0 || n == 0
        return nothing
    end

    A_source = isnothing(plan.new_A) ? qp.A : plan.new_A
    AT_source = qp.AT
    ratio_tol = max(1e-9, pparams.bound_tol)
    blocks_rows = cld(m, GPU_PRESOLVE_THREADS)

    candidate_mask = CUDA.zeros(UInt8, m)
    candidate_elim_col = CUDA.fill(Int32(0), m)
    candidate_elim_val = CUDA.zeros(Float64, m)
    candidate_keep_col = CUDA.fill(Int32(0), m)
    candidate_keep_val = CUDA.zeros(Float64, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_doubleton_eq_candidates!(
        candidate_mask,
        candidate_elim_col,
        candidate_elim_val,
        candidate_keep_col,
        candidate_keep_val,
        A_source.rowPtr,
        A_source.colVal,
        A_source.nzVal,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.col_nnz,
        plan.new_AL,
        plan.new_AU,
        pparams.zero_tol,
        pparams.bound_tol,
        ratio_tol,
        Int32(m),
    )

    candidate_elim_col_mask = CUDA.zeros(UInt8, n)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_mark_qp_doubleton_candidate_elim_cols!(
        candidate_elim_col_mask,
        candidate_mask,
        candidate_elim_col,
        Int32(m),
    )

    acceptable_mask = CUDA.zeros(UInt8, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_mark_qp_doubleton_eq_batch_acceptable!(
        acceptable_mask,
        candidate_mask,
        candidate_keep_col,
        candidate_elim_col,
        candidate_keep_val,
        candidate_elim_val,
        candidate_elim_col_mask,
        stats.col_nnz,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        plan.keep_row_mask,
        qp.Q.rowPtr,
        qp.Q.colVal,
        qp.Q.nzVal,
        qp.QT.rowPtr,
        qp.QT.colVal,
        qp.QT.nzVal,
        qp.q_diag,
        stats.q_offdiag_nnz,
        pparams.qp_doubleton_eq_require_qdiag_zero ? UInt8(1) : UInt8(0),
        pparams.zero_tol,
        Int32(pparams.doubleton_eq_max_fill_in_proxy),
        Int32(pparams.qp_doubleton_max_q_fill_abs),
        pparams.qp_doubleton_max_q_fill_ratio,
        Int32(m),
    )

    selected_mask = CUDA.zeros(UInt8, m)
    active_mask = copy(acceptable_mask)
    blocked_col = CUDA.zeros(UInt8, n)
    col_owner = CUDA.fill(typemax(Int32), n)
    for _ in 1:_DOUBLETONEQ_BATCH_MATCHING_ROUNDS
        fill!(col_owner, typemax(Int32))
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_claim_doubleton_eq_batch_columns!(
            col_owner,
            active_mask,
            blocked_col,
            candidate_keep_col,
            candidate_elim_col,
            Int32(m),
        )
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_select_doubleton_eq_batch_rows!(
            selected_mask,
            active_mask,
            blocked_col,
            candidate_keep_col,
            candidate_elim_col,
            col_owner,
            Int32(m),
        )
    end

    _, selected_rows, selected_count = build_maps_from_mask(selected_mask)
    k = Int(selected_count)
    k == 0 && return nothing

    selected_keep_col = CuVector{Int32}(undef, k)
    selected_elim_col = CuVector{Int32}(undef, k)
    selected_keep_val = CuVector{Float64}(undef, k)
    selected_elim_val = CuVector{Float64}(undef, k)
    selected_rhs = CuVector{Float64}(undef, k)
    selected_old_elim_l = CuVector{Float64}(undef, k)
    selected_old_elim_u = CuVector{Float64}(undef, k)
    selected_elim_obj = CuVector{Float64}(undef, k)
    selected_old_keep_l = CuVector{Float64}(undef, k)
    selected_old_keep_u = CuVector{Float64}(undef, k)
    blocks_sel = cld(k, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_sel _kernel_pack_doubleton_eq_selected_metadata!(
        selected_keep_col,
        selected_elim_col,
        selected_keep_val,
        selected_elim_val,
        selected_rhs,
        selected_old_elim_l,
        selected_old_elim_u,
        selected_elim_obj,
        selected_old_keep_l,
        selected_old_keep_u,
        selected_rows,
        candidate_keep_col,
        candidate_elim_col,
        candidate_keep_val,
        candidate_elim_val,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        plan.new_c,
        Int32(k),
    )

    keep_l_new = CUDA.zeros(Float64, k)
    keep_u_new = CUDA.zeros(Float64, k)
    alpha = CUDA.zeros(Float64, k)
    beta = CUDA.zeros(Float64, k)
    fixed_at = CUDA.zeros(Float64, k)
    keep_fixed_mask = CUDA.zeros(UInt8, k)
    lower_source = CUDA.zeros(Float64, k)
    upper_source = CUDA.zeros(Float64, k)
    infeas_flag = CUDA.zeros(Int32, 1)
    infeas_row_ref = CUDA.fill(Int32(m + 1), 1)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_sel _kernel_doubleton_eq_batch_bound_transfer!(
        infeas_flag,
        infeas_row_ref,
        keep_l_new,
        keep_u_new,
        alpha,
        beta,
        fixed_at,
        keep_fixed_mask,
        lower_source,
        upper_source,
        selected_rows,
        selected_keep_val,
        selected_elim_val,
        selected_rhs,
        selected_old_elim_l,
        selected_old_elim_u,
        selected_old_keep_l,
        selected_old_keep_u,
        pparams.bound_tol,
        Int32(k),
    )
    if Int(_copy_scalar_to_host(infeas_flag, 1)) != 0
        bad_row = Int(_copy_scalar_to_host(infeas_row_ref, 1))
        plan.has_infeasible = true
        plan.status_message = "QP doubleton-equality infeasibility at row $(bad_row): transferred bounds made l > u."
        return nothing
    end

    _, subst_pair_idx, subst_count_i32 = build_maps_from_mask(UInt8.(keep_fixed_mask .== UInt8(0)))
    subst_count = Int(subst_count_i32)
    subst_count == 0 && return nothing

    subst_rows = CuVector{Int32}(undef, subst_count)
    subst_keep_col = CuVector{Int32}(undef, subst_count)
    subst_elim_col = CuVector{Int32}(undef, subst_count)
    subst_keep_val = CuVector{Float64}(undef, subst_count)
    subst_elim_val = CuVector{Float64}(undef, subst_count)
    subst_rhs = CuVector{Float64}(undef, subst_count)
    subst_old_elim_l = CuVector{Float64}(undef, subst_count)
    subst_old_elim_u = CuVector{Float64}(undef, subst_count)
    subst_old_keep_l = CuVector{Float64}(undef, subst_count)
    subst_old_keep_u = CuVector{Float64}(undef, subst_count)
    subst_new_keep_l = CuVector{Float64}(undef, subst_count)
    subst_new_keep_u = CuVector{Float64}(undef, subst_count)
    subst_elim_obj = CuVector{Float64}(undef, subst_count)
    subst_alpha = CuVector{Float64}(undef, subst_count)
    subst_beta = CuVector{Float64}(undef, subst_count)
    subst_lower_source = CuVector{Float64}(undef, subst_count)
    subst_upper_source = CuVector{Float64}(undef, subst_count)
    blocks_subst = cld(subst_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst _kernel_pack_doubleton_eq_subst_metadata!(
        subst_rows,
        subst_keep_col,
        subst_elim_col,
        subst_keep_val,
        subst_elim_val,
        subst_rhs,
        subst_old_elim_l,
        subst_old_elim_u,
        subst_old_keep_l,
        subst_old_keep_u,
        subst_new_keep_l,
        subst_new_keep_u,
        subst_elim_obj,
        subst_alpha,
        subst_beta,
        subst_lower_source,
        subst_upper_source,
        subst_pair_idx,
        selected_rows,
        selected_keep_col,
        selected_elim_col,
        selected_keep_val,
        selected_elim_val,
        selected_rhs,
        selected_old_elim_l,
        selected_old_elim_u,
        selected_old_keep_l,
        selected_old_keep_u,
        selected_elim_obj,
        keep_l_new,
        keep_u_new,
        alpha,
        beta,
        lower_source,
        upper_source,
        Int32(subst_count),
    )

    row_delete = CUDA.zeros(UInt8, m)
    col_delete = CUDA.zeros(UInt8, n)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_sel _kernel_apply_qp_doubleton_eq_batch_subst_decisions!(
        plan.new_l,
        plan.new_u,
        row_delete,
        col_delete,
        selected_rows,
        selected_keep_col,
        selected_elim_col,
        keep_l_new,
        keep_u_new,
        keep_fixed_mask,
        Int32(k),
    )
    keep_row_new = UInt8.((plan.keep_row_mask .!= UInt8(0)) .& (row_delete .== UInt8(0)))
    keep_col_new = UInt8.((plan.keep_col_mask .!= UInt8(0)) .& (col_delete .== UInt8(0)))

    row_shift = CUDA.zeros(Float64, m)
    subst_support_counts = CuVector{Int32}(undef, subst_count)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst _kernel_doubleton_eq_subst_entry_counts!(
        subst_support_counts,
        subst_rows,
        subst_elim_col,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        keep_row_new,
        pparams.zero_tol,
        Int32(subst_count),
    )
    entry_prefix = cumsum(subst_support_counts)
    total_entries = Int(_copy_scalar_to_host(entry_prefix, subst_count))
    A_new = A_source
    if total_entries > 0
        entry_starts = CUDA.fill(Int32(1), subst_count)
        if subst_count > 1
            entry_starts[2:end] .= entry_prefix[1:(end - 1)] .+ Int32(1)
        end
        delta_rows = CuVector{Int32}(undef, total_entries)
        delta_cols = CuVector{Int32}(undef, total_entries)
        delta_vals = CuVector{Float64}(undef, total_entries)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst _kernel_build_doubleton_eq_subst_entries!(
            delta_rows,
            delta_cols,
            delta_vals,
            row_shift,
            entry_starts,
            subst_rows,
            subst_keep_col,
            subst_elim_col,
            subst_alpha,
            subst_beta,
            AT_source.rowPtr,
            AT_source.colVal,
            AT_source.nzVal,
            keep_row_new,
            pparams.zero_tol,
            Int32(subst_count),
        )
        delta_A = _build_doubleton_eq_delta_csr(delta_rows, delta_cols, delta_vals, size(A_source))
        A_new = A_source + sort_csr(delta_A)
    end
    plan.new_A = A_new
    plan.new_AL .-= row_shift
    plan.new_AU .-= row_shift

    delta_Q = _qp_build_doubleton_q_batch_delta_csr(qp, subst_elim_col, subst_keep_col, subst_alpha, pparams.zero_tol)
    plan.new_Q = qp.Q + delta_Q

    obj_contrib = CUDA.zeros(Float64, subst_count)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst _kernel_qp_doubleton_batch_c_update!(
        plan.new_c,
        obj_contrib,
        subst_elim_col,
        subst_keep_col,
        subst_elim_obj,
        subst_alpha,
        subst_beta,
        qp.q_diag,
        qp.QT.rowPtr,
        qp.QT.colVal,
        qp.QT.nzVal,
        pparams.zero_tol,
        Int32(subst_count),
    )
    plan.obj_constant_delta += sum(obj_contrib)

    _append_qp_doubleton_eq_batch_postsolve_records!(
        plan,
        pparams,
        AT_source,
        keep_row_new,
        subst_elim_col,
        subst_rows,
        subst_keep_col,
        subst_keep_val,
        subst_elim_val,
        subst_rhs,
        subst_old_elim_l,
        subst_old_elim_u,
        subst_old_keep_l,
        subst_old_keep_u,
        subst_new_keep_l,
        subst_new_keep_u,
        subst_elim_obj,
        subst_support_counts,
        subst_lower_source,
        subst_upper_source,
        pparams.zero_tol,
    )

    copyto!(plan.keep_row_mask, keep_row_new)
    copyto!(plan.keep_col_mask, keep_col_new)
    plan.has_change = true
    return nothing
end

function apply_rule_qp_doubleton_eq!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    return _apply_rule_qp_doubleton_eq_batch!(plan, qp, stats, pparams)
end
