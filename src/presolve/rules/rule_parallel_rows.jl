"""
Layer-2 rule: parallel rows.

Current scope:
- detect scalar-multiple live rows on the GPU
- intersect feasible activity intervals across each parallel-row group
- keep one representative row and delete the others
"""

@inline function _mark_infeasible_parallel_rows!(plan::PresolvePlan_gpu, msg::String)
    plan.has_infeasible = true
    plan.status_message = msg
    return nothing
end

@inline function _parallel_row_hash_mix(h::UInt64, x::UInt64)
    return (h ⊻ x) * UInt64(0x100000001b3)
end

function _kernel_parallel_row_hashes!(
    row_hash,
    keep_row,
    row_ptr,
    col_val,
    nz_val,
    coeff_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        if keep_row[i] == UInt8(0)
            @inbounds row_hash[i] = UInt64(i)
            return
        end

        @inbounds start_i = row_ptr[i]
        @inbounds stop_i = row_ptr[i + 1] - 1
        len_i = stop_i - start_i + 1
        if len_i <= 0
            @inbounds row_hash[i] = _parallel_row_hash_mix(UInt64(0xcbf29ce484222325), UInt64(i))
            return
        end

        @inbounds pivot = nz_val[start_i]
        if abs(pivot) <= coeff_tol
            @inbounds row_hash[i] = _parallel_row_hash_mix(UInt64(0xcbf29ce484222325), UInt64(i))
            return
        end

        h = _parallel_row_hash_mix(UInt64(0xcbf29ce484222325), UInt64(len_i))
        for p in start_i:stop_i
            @inbounds h = _parallel_row_hash_mix(h, UInt64(col_val[p]))
            @inbounds norm = nz_val[p] / pivot
            h = _parallel_row_hash_mix(h, reinterpret(UInt64, norm))
        end
        @inbounds row_hash[i] = h
    end
    return
end

@inline function _parallel_row_ratio(
    i,
    k,
    row_ptr,
    col_val,
    nz_val,
    coeff_tol,
)
    @inbounds start_i = row_ptr[i]
    @inbounds stop_i = row_ptr[i + 1] - 1
    @inbounds start_k = row_ptr[k]
    @inbounds stop_k = row_ptr[k + 1] - 1

    len_i = stop_i - start_i + 1
    len_k = stop_k - start_k + 1
    len_i == len_k || return (false, 0.0)

    if len_i <= 0
        return (true, 1.0)
    end

    ratio = 0.0
    ratio_set = false

    for offset in 0:(len_i - 1)
        p_i = start_i + offset
        p_k = start_k + offset
        @inbounds col_val[p_i] == col_val[p_k] || return (false, 0.0)

        @inbounds a_i = nz_val[p_i]
        @inbounds a_k = nz_val[p_k]
        if !ratio_set
            abs(a_k) > coeff_tol || return (false, 0.0)
            ratio = a_i / a_k
            ratio_set = true
        end

        abs(a_i - ratio * a_k) <= coeff_tol || return (false, 0.0)
    end

    return (ratio_set, ratio)
end

@inline function _scaled_interval_for_parallel_row(
    lower::Float64,
    upper::Float64,
    ratio::Float64,
)
    scaled_l = lower * ratio
    scaled_u = upper * ratio
    if ratio >= 0.0
        return (scaled_l, scaled_u)
    end
    return (scaled_u, scaled_l)
end

@inline function _intervals_disjoint(
    l1::Float64,
    u1::Float64,
    l2::Float64,
    u2::Float64,
    tol::Float64,
)
    return max(l1, l2) > min(u1, u2) + tol
end

@inline function _interval_contains(
    outer_l::Float64,
    outer_u::Float64,
    inner_l::Float64,
    inner_u::Float64,
    tol::Float64,
)
    return outer_l <= inner_l + tol && inner_u <= outer_u + tol
end

@inline function _row_interval_is_equality(
    lower::Float64,
    upper::Float64,
    tol::Float64,
)
    return isfinite(lower) && isfinite(upper) && abs(lower - upper) <= tol
end

function _kernel_parallel_row_groups!(
    status_flag,
    row_delete,
    sorted_hash,
    sorted_rows,
    keep_row,
    AL,
    AU,
    row_ptr,
    col_val,
    nz_val,
    coeff_tol,
    interval_tol,
    m,
)
    s = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if s <= m
        @inbounds hs = sorted_hash[s]
        if s > 1
            @inbounds sorted_hash[s - 1] == hs && return
        end

        e = s
        while e < m
            @inbounds sorted_hash[e + 1] == hs || break
            e += 1
        end
        e > s || return

        rep = Int32(0)
        rep_l = 0.0
        rep_u = 0.0
        rep_is_eq = false

        for a in s:e
            @inbounds row = sorted_rows[a]
            keep_row[row] == UInt8(0) && continue

            row_l = @inbounds AL[row]
            row_u = @inbounds AU[row]
            row_is_eq = _row_interval_is_equality(row_l, row_u, interval_tol)

            if rep == Int32(0) || (row_is_eq && !rep_is_eq)
                rep = row
                rep_l = row_l
                rep_u = row_u
                rep_is_eq = row_is_eq
            end
        end

        rep == Int32(0) && return

        found_parallel_peer = false
        for a in s:e
            @inbounds row = sorted_rows[a]
            if row == rep || keep_row[row] == UInt8(0)
                continue
            end

            is_parallel, ratio = _parallel_row_ratio(rep, row, row_ptr, col_val, nz_val, coeff_tol)
            is_parallel || continue
            found_parallel_peer = true

            @inbounds row_l, row_u = _scaled_interval_for_parallel_row(AL[row], AU[row], ratio)
            if _intervals_disjoint(rep_l, rep_u, row_l, row_u, interval_tol)
                CUDA.@atomic status_flag[1] = max(status_flag[1], Int32(1))
                return
            end

            rep_l = max(rep_l, row_l)
            rep_u = min(rep_u, row_u)
            @inbounds row_delete[row] = UInt8(1)
        end

        if found_parallel_peer
            @inbounds AL[rep] = rep_l
            @inbounds AU[rep] = rep_u
        end
    end
    return
end

function apply_rule_parallel_rows!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    _stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    m = length(plan.keep_row_mask)
    if m <= 1
        return nothing
    end

    row_hash = CUDA.zeros(UInt64, m)
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_parallel_row_hashes!(
        row_hash,
        plan.keep_row_mask,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.zero_tol,
        Int32(m),
    )

    order64 = sortperm(row_hash)
    sorted_rows = Int32.(order64)
    sorted_hash = row_hash[order64]

    status_flag = CUDA.zeros(Int32, 1)
    row_delete = CUDA.zeros(UInt8, m)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_parallel_row_groups!(
        status_flag,
        row_delete,
        sorted_hash,
        sorted_rows,
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.zero_tol,
        pparams.feasibility_tol,
        Int32(m),
    )

    status = CUDA.@allowscalar Int(status_flag[1])
    if status != 0
        _mark_infeasible_parallel_rows!(
            plan,
            "Parallel-row infeasibility: some live parallel-row pair has disjoint scaled intervals.",
        )
        return nothing
    end

    removed_any = Int(sum(Int32.(row_delete .!= UInt8(0)))) > 0
    tightened_any = false
    if removed_any
        copyto!(plan.keep_row_mask, UInt8.((plan.keep_row_mask .!= UInt8(0)) .& .!(row_delete .!= UInt8(0))))
        tightened_any = true
    end

    if tightened_any
        plan.has_row_action = true
        plan.has_change = true
    end

    return nothing
end
