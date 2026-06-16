"""
QP layer-2 rule: exact objective-compatible parallel columns.

The QP variant keeps the LP GPU hash/group detection for the linear matrix and
adds a device-side objective check:

    A[:, k] = r A[:, j], Q[:, k] = r Q[:, j], c[k] = r c[j].

Only exact aggregate merges are applied here. Nonzero objective-gap fixing is
left to QP dual-fix style rules, because the quadratic part can make the
one-dimensional choice depend on the kept variables unless the column is
separable.
"""

@inline function _qp_next_live_csr_entry(
    ptr,
    stop,
    idx,
    val,
    keep,
    zero_tol,
)
    cur = ptr
    while cur <= stop
        @inbounds row = idx[cur]
        if keep[row] != UInt8(0)
            @inbounds a = val[cur]
            if abs(a) > zero_tol
                return cur
            end
        end
        cur += Int32(1)
    end
    return stop + Int32(1)
end

@inline function _qp_parallel_q_ratio_exact(
    j,
    k,
    ratio,
    q_row_ptr,
    q_col_idx,
    q_val,
    keep_col,
    zero_tol,
    obj_tol,
)
    @inbounds ptr_j = q_row_ptr[j]
    @inbounds stop_j = q_row_ptr[j + 1] - Int32(1)
    @inbounds ptr_k = q_row_ptr[k]
    @inbounds stop_k = q_row_ptr[k + 1] - Int32(1)

    while true
        ptr_j = _qp_next_live_csr_entry(ptr_j, stop_j, q_col_idx, q_val, keep_col, zero_tol)
        ptr_k = _qp_next_live_csr_entry(ptr_k, stop_k, q_col_idx, q_val, keep_col, zero_tol)

        if ptr_j > stop_j || ptr_k > stop_k
            break
        end

        @inbounds row_j = q_col_idx[ptr_j]
        @inbounds row_k = q_col_idx[ptr_k]
        row_j == row_k || return false

        @inbounds qj = q_val[ptr_j]
        @inbounds qk = q_val[ptr_k]
        abs(qk - ratio * qj) <= obj_tol || return false

        ptr_j += Int32(1)
        ptr_k += Int32(1)
    end

    ptr_j = _qp_next_live_csr_entry(ptr_j, stop_j, q_col_idx, q_val, keep_col, zero_tol)
    ptr_k = _qp_next_live_csr_entry(ptr_k, stop_k, q_col_idx, q_val, keep_col, zero_tol)
    return ptr_j > stop_j && ptr_k > stop_k
end

function _kernel_qp_parallel_col_groups!(
    col_delete,
    merge_to,
    merge_ratio,
    merge_from_l,
    merge_from_u,
    merge_to_l,
    merge_to_u,
    sorted_hash,
    sorted_cols,
    keep_col,
    keep_row,
    c,
    l,
    u,
    a_row_ptr,
    a_row_idx,
    a_row_val,
    q_row_ptr,
    q_col_idx,
    q_val,
    zero_tol,
    coeff_tol,
    obj_tol,
    n,
)
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= n
        @inbounds hs = sorted_hash[s]
        if s > 1
            @inbounds sorted_hash[s - 1] == hs && return
        end

        e = s
        while e < n
            @inbounds sorted_hash[e + 1] == hs || break
            e += Int32(1)
        end
        e > s || return

        for a in s:(e - Int32(1))
            @inbounds j = sorted_cols[a]
            if keep_col[j] == UInt8(0) || col_delete[j] != UInt8(0)
                continue
            end

            for b in (a + Int32(1)):e
                @inbounds k = sorted_cols[b]
                if keep_col[k] == UInt8(0) || col_delete[k] != UInt8(0)
                    continue
                end

                is_parallel, ratio = _parallel_col_ratio(
                    j,
                    k,
                    a_row_ptr,
                    a_row_idx,
                    a_row_val,
                    keep_row,
                    zero_tol,
                    coeff_tol,
                )
                is_parallel || continue

                @inbounds abs(c[k] - ratio * c[j]) <= obj_tol || continue
                _qp_parallel_q_ratio_exact(
                    j,
                    k,
                    ratio,
                    q_row_ptr,
                    q_col_idx,
                    q_val,
                    keep_col,
                    zero_tol,
                    obj_tol,
                ) || continue

                @inbounds target_l = l[j]
                @inbounds target_u = u[j]
                @inbounds source_l = l[k]
                @inbounds source_u = u[k]
                new_l = _merged_lower_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio)
                new_u = _merged_upper_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio)
                new_l <= new_u + obj_tol || continue

                @inbounds merge_to[k] = j
                @inbounds merge_ratio[k] = ratio
                @inbounds merge_from_l[k] = source_l
                @inbounds merge_from_u[k] = source_u
                @inbounds merge_to_l[k] = target_l
                @inbounds merge_to_u[k] = target_u
                @inbounds l[j] = new_l
                @inbounds u[j] = new_u
                @inbounds col_delete[k] = UInt8(1)
            end
        end
    end
    return
end

function apply_rule_qp_parallel_cols!(
    plan::QPresolvePlan_gpu,
    qp::QP_info_gpu,
    _stats::QPresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    _, n = size(qp.A)
    n <= 1 && return nothing

    A_source = isnothing(plan.new_A) ? qp.A : plan.new_A
    AT_source = isnothing(plan.new_A) ? qp.AT : transpose_csr(A_source)
    Q_source = isnothing(plan.new_Q) ? qp.Q : plan.new_Q
    QT_source = isnothing(plan.new_Q) ? qp.QT : transpose_csr(Q_source)

    col_hash = CUDA.zeros(UInt64, n)
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_parallel_col_hashes!(
        col_hash,
        plan.keep_col_mask,
        plan.keep_row_mask,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        pparams.zero_tol,
        Int32(n),
    )

    order64 = sortperm(col_hash)
    sorted_cols = Int32.(order64)
    sorted_hash = col_hash[order64]

    col_delete = CUDA.zeros(UInt8, n)
    merge_to = CUDA.fill(Int32(0), n)
    merge_ratio = CUDA.zeros(Float64, n)
    merge_from_l = CUDA.zeros(Float64, n)
    merge_from_u = CUDA.zeros(Float64, n)
    merge_to_l = CUDA.zeros(Float64, n)
    merge_to_u = CUDA.zeros(Float64, n)
    coeff_tol = max(pparams.zero_tol, pparams.bound_tol)
    obj_tol = max(pparams.zero_tol, pparams.bound_tol)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_qp_parallel_col_groups!(
        col_delete,
        merge_to,
        merge_ratio,
        merge_from_l,
        merge_from_u,
        merge_to_l,
        merge_to_u,
        sorted_hash,
        sorted_cols,
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        QT_source.rowPtr,
        QT_source.colVal,
        QT_source.nzVal,
        pparams.zero_tol,
        coeff_tol,
        obj_tol,
        Int32(n),
    )

    _, merged_from, merge_count = build_maps_from_mask(col_delete)
    Int(merge_count) == 0 && return nothing

    copyto!(
        plan.keep_col_mask,
        UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(col_delete .!= UInt8(0))),
    )
    plan.merged_col_from = merged_from
    plan.merged_col_to = gather_by_red2org(merge_to, merged_from)
    plan.merged_col_ratio = gather_by_red2org(merge_ratio, merged_from)
    plan.merged_col_from_l = gather_by_red2org(merge_from_l, merged_from)
    plan.merged_col_from_u = gather_by_red2org(merge_from_u, merged_from)
    plan.merged_col_to_l = gather_by_red2org(merge_to_l, merged_from)
    plan.merged_col_to_u = gather_by_red2org(merge_to_u, merged_from)

    append_parallel_col_records_gpu!(
        plan.tape_gpu,
        plan.merged_col_from,
        plan.merged_col_to,
        plan.merged_col_ratio,
        plan.merged_col_from_l,
        plan.merged_col_from_u,
        plan.merged_col_to_l,
        plan.merged_col_to_u;
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if pparams.record_postsolve_tape_cpu
        plan.tape = PostsolveTape(plan.tape_gpu)
    end
    plan.has_change = true
    return nothing
end
