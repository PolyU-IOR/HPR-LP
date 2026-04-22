"""
Layer-2 rule: parallel columns.

Current GPU scope:
- detect live columns through the CSR rows of `AT`
- accept exact merges for any nonzero parallel ratio
- apply the PSLP one-sided finite-fixing branches when `c_k != r c_j`
- merge `k -> j` by keeping column `j` and removing column `k`
- aggregate bounds onto the kept variable as `y_j = x_j + r * x_k`
"""

@inline function _mark_unbounded_parallel_cols!(plan::PresolvePlan_gpu, msg::String)
    plan.has_unbounded = true
    plan.status_message = msg
    return nothing
end

@inline function _parallel_col_hash_mix(h::UInt64, x::UInt64)
    return (h ⊻ x) * UInt64(0x100000001b3)
end

function _kernel_parallel_col_hashes!(
    col_hash,
    keep_col,
    keep_row,
    row_ptr,
    row_idx,
    row_val,
    coeff_tol,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        if keep_col[j] == UInt8(0)
            @inbounds col_hash[j] = UInt64(j)
            return
        end

        @inbounds start_j = row_ptr[j]
        @inbounds stop_j = row_ptr[j + 1] - 1

        live_len = Int32(0)
        pivot = 0.0
        pivot_set = false
        for p in start_j:stop_j
            @inbounds row = row_idx[p]
            if keep_row[row] != UInt8(0)
                @inbounds a = row_val[p]
                if abs(a) > coeff_tol
                    live_len += Int32(1)
                    if !pivot_set
                        pivot = a
                        pivot_set = true
                    end
                end
            end
        end

        if !pivot_set || live_len == Int32(0)
            @inbounds col_hash[j] = _parallel_col_hash_mix(UInt64(0xcbf29ce484222325), UInt64(j))
            return
        end

        h = _parallel_col_hash_mix(UInt64(0xcbf29ce484222325), UInt64(live_len))
        for p in start_j:stop_j
            @inbounds row = row_idx[p]
            if keep_row[row] != UInt8(0)
                @inbounds a = row_val[p]
                if abs(a) > coeff_tol
                    h = _parallel_col_hash_mix(h, UInt64(row))
                    norm = a / pivot
                    h = _parallel_col_hash_mix(h, reinterpret(UInt64, norm))
                end
            end
        end
        @inbounds col_hash[j] = h
    end
    return
end

@inline function _next_live_parallel_col_entry(
    ptr,
    stop,
    row_idx,
    row_val,
    keep_row,
    zero_tol,
)
    cur = ptr
    while cur <= stop
        @inbounds row = row_idx[cur]
        if keep_row[row] != UInt8(0)
            @inbounds a = row_val[cur]
            if abs(a) > zero_tol
                return cur
            end
        end
        cur += Int32(1)
    end
    return stop + Int32(1)
end

@inline function _parallel_col_ratio(
    j,
    k,
    row_ptr,
    row_idx,
    row_val,
    keep_row,
    zero_tol,
    coeff_tol,
)
    @inbounds ptr_j = row_ptr[j]
    @inbounds stop_j = row_ptr[j + 1] - Int32(1)
    @inbounds ptr_k = row_ptr[k]
    @inbounds stop_k = row_ptr[k + 1] - Int32(1)

    ratio = 0.0
    ratio_set = false

    while true
        ptr_j = _next_live_parallel_col_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol)
        ptr_k = _next_live_parallel_col_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol)

        if ptr_j > stop_j || ptr_k > stop_k
            break
        end

        @inbounds row_j = row_idx[ptr_j]
        @inbounds row_k = row_idx[ptr_k]
        row_j == row_k || return (false, 0.0)

        @inbounds a_j = row_val[ptr_j]
        @inbounds a_k = row_val[ptr_k]

        if !ratio_set
            abs(a_j) > zero_tol || return (false, 0.0)
            ratio = a_k / a_j
            abs(ratio) > zero_tol || return (false, 0.0)
            isfinite(ratio) || return (false, 0.0)
            ratio_set = true
        end

        abs(a_k - ratio * a_j) <= coeff_tol || return (false, 0.0)

        ptr_j += Int32(1)
        ptr_k += Int32(1)
    end

    ptr_j = _next_live_parallel_col_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol)
    ptr_k = _next_live_parallel_col_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol)

    if ptr_j <= stop_j || ptr_k <= stop_k || !ratio_set
        return (false, 0.0)
    end

    return (true, ratio)
end

@inline function _merged_lower_bound_parallel_cols(
    target_l,
    target_u,
    source_l,
    source_u,
    ratio,
)
    if ratio > 0.0
        if isfinite(target_l) && isfinite(source_l)
            return target_l + ratio * source_l
        end
    else
        if isfinite(target_l) && isfinite(source_u)
            return target_l + ratio * source_u
        end
    end
    return -Inf
end

@inline function _merged_upper_bound_parallel_cols(
    target_l,
    target_u,
    source_l,
    source_u,
    ratio,
)
    if ratio > 0.0
        if isfinite(target_u) && isfinite(source_u)
            return target_u + ratio * source_u
        end
    else
        if isfinite(target_u) && isfinite(source_l)
            return target_u + ratio * source_l
        end
    end
    return Inf
end

function _kernel_parallel_col_groups!(
    status_flag,
    col_delete,
    fixed_mask,
    fixed_val,
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
    row_ptr,
    row_idx,
    row_val,
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
            if keep_col[j] == UInt8(0) || col_delete[j] != UInt8(0) || fixed_mask[j] != UInt8(0)
                continue
            end

            for b in (a + Int32(1)):e
                @inbounds k = sorted_cols[b]
                if keep_col[k] == UInt8(0) || col_delete[k] != UInt8(0) || fixed_mask[k] != UInt8(0)
                    continue
                end

                is_parallel, ratio = _parallel_col_ratio(
                    j,
                    k,
                    row_ptr,
                    row_idx,
                    row_val,
                    keep_row,
                    zero_tol,
                    coeff_tol,
                )
                if !is_parallel
                    continue
                end

                @inbounds obj_gap = c[k] - ratio * c[j]
                @inbounds target_l = l[j]
                @inbounds target_u = u[j]
                @inbounds source_l = l[k]
                @inbounds source_u = u[k]

                if abs(obj_gap) <= obj_tol
                    @inbounds merge_to[k] = j
                    @inbounds merge_ratio[k] = ratio
                    @inbounds merge_from_l[k] = source_l
                    @inbounds merge_from_u[k] = source_u
                    @inbounds merge_to_l[k] = target_l
                    @inbounds merge_to_u[k] = target_u
                    @inbounds l[j] = _merged_lower_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio)
                    @inbounds u[j] = _merged_upper_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio)
                    @inbounds col_delete[k] = UInt8(1)
                    continue
                end

                fix_xk_to_lower = false
                fix_xk_to_upper = false
                fix_xj_to_lower = false
                fix_xj_to_upper = false

                if obj_gap > obj_tol
                    if ratio > 0.0
                        fix_xk_to_lower = !isfinite(target_u)
                        fix_xj_to_upper = !isfinite(source_l)
                    else
                        fix_xk_to_lower = !isfinite(target_l)
                        fix_xj_to_lower = !isfinite(source_l)
                    end
                else
                    if ratio > 0.0
                        fix_xk_to_upper = !isfinite(target_l)
                        fix_xj_to_lower = !isfinite(source_u)
                    else
                        fix_xk_to_upper = !isfinite(target_u)
                        fix_xj_to_upper = !isfinite(source_u)
                    end
                end

                if fix_xk_to_lower
                    if !isfinite(source_l)
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                        return
                    end
                    @inbounds fixed_mask[k] = UInt8(1)
                    @inbounds fixed_val[k] = source_l
                    continue
                elseif fix_xk_to_upper
                    if !isfinite(source_u)
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                        return
                    end
                    @inbounds fixed_mask[k] = UInt8(1)
                    @inbounds fixed_val[k] = source_u
                    continue
                end

                if fix_xj_to_lower
                    if !isfinite(target_l)
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                        return
                    end
                    @inbounds fixed_mask[j] = UInt8(1)
                    @inbounds fixed_val[j] = target_l
                    break
                elseif fix_xj_to_upper
                    if !isfinite(target_u)
                        CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                        return
                    end
                    @inbounds fixed_mask[j] = UInt8(1)
                    @inbounds fixed_val[j] = target_u
                    break
                end
            end
        end
    end
    return
end

function _kernel_parallel_col_fixed_row_shift!(
    row_shift,
    fixed_mask,
    fixed_val,
    keep_row,
    row_ptr,
    row_idx,
    row_val,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n && fixed_mask[j] != UInt8(0)
        @inbounds vj = fixed_val[j]
        @inbounds start_j = row_ptr[j]
        @inbounds stop_j = row_ptr[j + 1] - Int32(1)
        for p in start_j:stop_j
            @inbounds row = row_idx[p]
            if keep_row[row] != UInt8(0)
                @inbounds CUDA.@atomic row_shift[row] += row_val[p] * vj
            end
        end
    end
    return
end

function _append_parallel_col_postsolve_records!(
    plan::PresolvePlan_gpu,
    merged_from,
    merged_to,
    merged_ratio,
    merged_from_l,
    merged_from_u,
    merged_to_l,
    merged_to_u,
    record_cpu::Bool,
)
    merge_count = length(merged_from)
    merge_count == 0 && return nothing

    append_parallel_col_records_gpu!(
        plan.tape_gpu,
        merged_from,
        merged_to,
        merged_ratio,
        merged_from_l,
        merged_from_u,
        merged_to_l,
        merged_to_u;
        dual_mode=POSTSOLVE_DUAL_MINIMAL,
    )
    if record_cpu
        append_parallel_col_records!(
            plan.tape,
            merged_from,
            merged_to,
            merged_ratio,
            merged_from_l,
            merged_from_u,
            merged_to_l,
            merged_to_u;
            dual_mode=POSTSOLVE_DUAL_MINIMAL,
        )
    end
    return nothing
end

"""
Rule: merge exact objective-compatible parallel columns using the CSR rows of `AT`.
"""
function apply_rule_parallel_cols!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    _stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    _, n = size(lp.A)
    if n <= 1
        return nothing
    end

    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
    AT_source = isnothing(plan.new_A) ? lp.AT : transpose_csr(A_source)

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

    status_flag = CUDA.zeros(Int32, 1)
    col_delete = CUDA.zeros(UInt8, n)
    fixed_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    merge_to = CUDA.fill(Int32(0), n)
    merge_ratio = CUDA.zeros(Float64, n)
    merge_from_l = CUDA.zeros(Float64, n)
    merge_from_u = CUDA.zeros(Float64, n)
    merge_to_l = CUDA.zeros(Float64, n)
    merge_to_u = CUDA.zeros(Float64, n)
    coeff_tol = max(pparams.zero_tol, pparams.bound_tol)
    obj_tol = max(pparams.zero_tol, pparams.bound_tol)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_parallel_col_groups!(
        status_flag,
        col_delete,
        fixed_mask,
        fixed_val,
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
        pparams.zero_tol,
        coeff_tol,
        obj_tol,
        Int32(n),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status != 0
        _mark_unbounded_parallel_cols!(
            plan,
            "Parallel-column unboundedness: some improving fixing bound is infinite.",
        )
        return nothing
    end

    _, merged_from, merge_count = build_maps_from_mask(col_delete)
    fixed_count = Int(sum(Int32.(fixed_mask .!= UInt8(0))))
    if Int(merge_count) == 0 && fixed_count == 0
        return nothing
    end

    if fixed_count > 0
        row_shift = CUDA.zeros(Float64, size(lp.A, 1))
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_parallel_col_fixed_row_shift!(
            row_shift,
            fixed_mask,
            fixed_val,
            plan.keep_row_mask,
            AT_source.rowPtr,
            AT_source.colVal,
            AT_source.nzVal,
            Int32(n),
        )

        append_fixed_col_postsolve_records!(
            plan,
            lp,
            pparams,
            fixed_mask,
            fixed_val,
            plan.new_c,
            plan.keep_row_mask,
        )
        append_plan_fixed_from_mask!(plan, fixed_mask, fixed_val)
        plan.obj_constant_delta += _stable_presolve_masked_product_sum(plan.new_c, fixed_val, fixed_mask)
        copyto!(plan.new_AL, plan.new_AL .- row_shift)
        copyto!(plan.new_AU, plan.new_AU .- row_shift)
        plan.has_row_action = true
    end

    col_remove = UInt8.((col_delete .!= UInt8(0)) .| (fixed_mask .!= UInt8(0)))
    copyto!(
        plan.keep_col_mask,
        UInt8.((plan.keep_col_mask .!= UInt8(0)) .& .!(col_remove .!= UInt8(0))),
    )
    plan.merged_col_from = merged_from
    plan.merged_col_to = gather_by_red2org(merge_to, merged_from)
    plan.merged_col_ratio = gather_by_red2org(merge_ratio, merged_from)
    plan.merged_col_from_l = gather_by_red2org(merge_from_l, merged_from)
    plan.merged_col_from_u = gather_by_red2org(merge_from_u, merged_from)
    plan.merged_col_to_l = gather_by_red2org(merge_to_l, merged_from)
    plan.merged_col_to_u = gather_by_red2org(merge_to_u, merged_from)
    if pparams.record_postsolve_tape
        _append_parallel_col_postsolve_records!(
            plan,
            plan.merged_col_from,
            plan.merged_col_to,
            plan.merged_col_ratio,
            plan.merged_col_from_l,
            plan.merged_col_from_u,
            plan.merged_col_to_l,
            plan.merged_col_to_u,
            pparams.record_postsolve_tape_cpu,
        )
    end
    plan.has_col_action = true
    plan.has_change = true
    return nothing
end
