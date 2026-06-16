"""
Layer-2 rule: doubleton equality rows.

Current GPU scope:
- detect one live equality row with exactly two live columns
- choose a deterministic eliminated/stay pair
- apply the exact substitution formula from the document
- update the objective, row sides, and stay-column bounds
- rebuild the local matrix on GPU without host materialization

"""

using CUDA.CUSPARSE: CuSparseMatrixCSR, sort_csr

const _DOUBLETONEQ_MAX_RATIO_PIVOT = 1e3
const _DOUBLETONEQ_MAX_FILL_IN_PROXY = 10
# Always canonicalize batch delta matrices through COO->CSR before CUSPARSE geam.
# The hand-built CSR path scatters entries by row but does not sort or coalesce
# row columns, which can make sparse addition fragile on some instances.
const _DOUBLETONEQ_DIRECT_CSR_MAX_ENTRIES = 0
const _DOUBLETONEQ_BATCH_MATCHING_ROUNDS = 8
const _DOUBLETONEQ_BATCH_INNER_ROUNDS = 1024

@inline function _doubleton_eq_delta_csr_mode()
    env_name = "GPUPRESOLVER_DOUBLETON_DELTA_CSR_MODE"
    mode = lowercase(strip(_gpu_presolver_env_value(env_name, "csr")))
    if mode in ("coo",)
        return :coo
    elseif mode in ("direct", "direct_sorted", "csr", "default")
        return :direct_sorted
    end
    error("Unsupported $(env_name)=$(repr(mode)); expected \"coo\" or \"csr\".")
end

@inline function _is_integral_ratio_doubleton_eq(num::Float64, den::Float64, tol::Float64)
    abs(den) > tol || return false
    ratio = abs(num / den)
    isfinite(ratio) || return false
    return abs(ratio - round(ratio)) <= tol
end

@inline function _is_acceptable_doubleton_eq_pivot(
    keep_val::Float64,
    elim_val::Float64,
)
    abs(elim_val) > 0.0 || return false
    pivot_ratio = abs(keep_val / elim_val)
    return pivot_ratio <= _DOUBLETONEQ_MAX_RATIO_PIVOT &&
           pivot_ratio >= 1.0 / _DOUBLETONEQ_MAX_RATIO_PIVOT
end

@inline function _choose_doubleton_eq_columns(
    col1,
    val1,
    col2,
    val2,
    col_nnz,
    tol,
)
    integral12 = _is_integral_ratio_doubleton_eq(val1, val2, tol)
    integral21 = _is_integral_ratio_doubleton_eq(val2, val1, tol)

    if col_nnz[col1] == Int32(1) && col_nnz[col2] != Int32(1)
        return (Int32(col1), Float64(val1), Int32(col2), Float64(val2))
    elseif col_nnz[col1] != Int32(1) && col_nnz[col2] == Int32(1)
        return (Int32(col2), Float64(val2), Int32(col1), Float64(val1))
    elseif integral12 && !integral21
        return (Int32(col2), Float64(val2), Int32(col1), Float64(val1))
    elseif integral21 && !integral12
        return (Int32(col1), Float64(val1), Int32(col2), Float64(val2))
    elseif col_nnz[col1] < col_nnz[col2]
        return (Int32(col1), Float64(val1), Int32(col2), Float64(val2))
    else
        return (Int32(col2), Float64(val2), Int32(col1), Float64(val1))
    end
end

function _doubleton_eq_mapped_interval(
    rhs::Float64,
    keep_val::Float64,
    elim_val::Float64,
    elim_l::Float64,
    elim_u::Float64,
)
    alpha = -keep_val / elim_val
    beta = rhs / elim_val
    mapped_l = -Inf
    mapped_u = Inf

    if isfinite(elim_l)
        bound_val = (elim_l - beta) / alpha
        if alpha > 0.0
            mapped_l = bound_val
        else
            mapped_u = bound_val
        end
    end

    if isfinite(elim_u)
        bound_val = (elim_u - beta) / alpha
        if alpha > 0.0
            mapped_u = bound_val
        else
            mapped_l = bound_val
        end
    end

    if mapped_l > mapped_u
        mapped_l, mapped_u = mapped_u, mapped_l
    end

    return (mapped_l, mapped_u)
end

@inline function _doubleton_eq_lower_source(
    old_keep_l::Float64,
    mapped_l::Float64,
    new_keep_l::Float64,
    tol::Float64,
)
    induced = isfinite(mapped_l) && abs(new_keep_l - mapped_l) <= tol
    inherited = isfinite(old_keep_l) && abs(new_keep_l - old_keep_l) <= tol
    if inherited && induced
        return DOUBLETON_EQ_BOUND_BOTH
    elseif induced
        return DOUBLETON_EQ_BOUND_INDUCED
    else
        return DOUBLETON_EQ_BOUND_INHERITED
    end
end

@inline function _doubleton_eq_upper_source(
    old_keep_u::Float64,
    mapped_u::Float64,
    new_keep_u::Float64,
    tol::Float64,
)
    induced = isfinite(mapped_u) && abs(new_keep_u - mapped_u) <= tol
    inherited = isfinite(old_keep_u) && abs(new_keep_u - old_keep_u) <= tol
    if inherited && induced
        return DOUBLETON_EQ_BOUND_BOTH
    elseif induced
        return DOUBLETON_EQ_BOUND_INDUCED
    else
        return DOUBLETON_EQ_BOUND_INHERITED
    end
end

function _doubleton_eq_fill_in_proxy_host(
    row_ptr::AbstractVector{<:Integer},
    col_val::AbstractVector{<:Integer},
    keep_col::Int,
    elim_col::Int,
)
    fill_in = -1
    keep_start = Int(row_ptr[keep_col])
    keep_stop = Int(row_ptr[keep_col + 1]) - 1
    if keep_stop < keep_start
        return fill_in
    end
    jj = keep_start

    elim_start = Int(row_ptr[elim_col])
    elim_stop = Int(row_ptr[elim_col + 1]) - 1
    if elim_stop < elim_start
        return fill_in
    end
    kk = elim_start
    while jj <= keep_stop && kk <= elim_stop
        keep_row = Int(col_val[jj])
        elim_row = Int(col_val[kk])
        if keep_row == elim_row
            jj += 1
            kk += 1
        elseif elim_row < keep_row
            kk += 1
            fill_in += 1
        else
            jj += 1
        end
    end
    fill_in += elim_stop - kk + 1
    return fill_in
end

@inline function _doubleton_eq_fill_in_proxy_device(
    row_ptr,
    col_val,
    keep_col,
    elim_col,
)
    fill_in = Int32(-1)
    keep_start = @inbounds row_ptr[keep_col]
    keep_stop = @inbounds row_ptr[keep_col + 1] - Int32(1)
    if keep_stop < keep_start
        return fill_in
    end
    jj = keep_start

    elim_start = @inbounds row_ptr[elim_col]
    elim_stop = @inbounds row_ptr[elim_col + 1] - Int32(1)
    if elim_stop < elim_start
        return fill_in
    end
    kk = elim_start
    while jj <= keep_stop && kk <= elim_stop
        keep_row = @inbounds col_val[jj]
        elim_row = @inbounds col_val[kk]
        if keep_row == elim_row
            jj += Int32(1)
            kk += Int32(1)
        elseif elim_row < keep_row
            kk += Int32(1)
            fill_in += Int32(1)
        else
            jj += Int32(1)
        end
    end
    fill_in += elim_stop - kk + Int32(1)
    return fill_in
end

@inline function _doubleton_eq_next_live_col_row(
    row_ptr,
    col_val,
    nz_val,
    keep_row,
    zero_tol,
    col,
    pos,
)
    stop = @inbounds row_ptr[col + Int32(1)] - Int32(1)
    while pos <= stop
        row = @inbounds col_val[pos]
        val = @inbounds nz_val[pos]
        if keep_row[row] != UInt8(0) && abs(val) > zero_tol
            return row, pos
        end
        pos += Int32(1)
    end
    return Int32(typemax(Int32)), pos
end

@inline function _doubleton_eq_fill_in_proxy_device_live(
    row_ptr,
    col_val,
    nz_val,
    keep_row,
    zero_tol,
    keep_col,
    elim_col,
)
    fill_in = Int32(-1)
    keep_start = @inbounds row_ptr[keep_col]
    keep_stop = @inbounds row_ptr[keep_col + Int32(1)] - Int32(1)
    elim_start = @inbounds row_ptr[elim_col]
    elim_stop = @inbounds row_ptr[elim_col + Int32(1)] - Int32(1)
    if keep_stop < keep_start || elim_stop < elim_start
        return fill_in
    end

    jj = keep_start
    kk = elim_start
    while jj <= keep_stop && kk <= elim_stop
        keep_row_j, jj_live = _doubleton_eq_next_live_col_row(
            row_ptr,
            col_val,
            nz_val,
            keep_row,
            zero_tol,
            keep_col,
            jj,
        )
        elim_row_k, kk_live = _doubleton_eq_next_live_col_row(
            row_ptr,
            col_val,
            nz_val,
            keep_row,
            zero_tol,
            elim_col,
            kk,
        )
        if keep_row_j == Int32(typemax(Int32)) || elim_row_k == Int32(typemax(Int32))
            break
        end
        if keep_row_j == elim_row_k
            jj = jj_live + Int32(1)
            kk = kk_live + Int32(1)
        elseif elim_row_k < keep_row_j
            kk = kk_live + Int32(1)
            fill_in += Int32(1)
        else
            jj = jj_live + Int32(1)
        end
    end

    while kk <= elim_stop
        elim_row_k, kk_live = _doubleton_eq_next_live_col_row(
            row_ptr,
            col_val,
            nz_val,
            keep_row,
            zero_tol,
            elim_col,
            kk,
        )
        elim_row_k == Int32(typemax(Int32)) && break
        fill_in += Int32(1)
        kk = kk_live + Int32(1)
    end
    return fill_in
end

function _kernel_doubleton_eq_live_col_nnz!(
    col_nnz,
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    keep_col,
    zero_tol,
    n,
)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if col <= n
        if @inbounds(keep_col[col] == UInt8(0))
            @inbounds col_nnz[col] = Int32(0)
            return
        end

        count = Int32(0)
        @inbounds start_idx = AT_row_ptr[col]
        @inbounds stop_idx = AT_row_ptr[col + Int32(1)] - Int32(1)
        if start_idx <= stop_idx
            for p in start_idx:stop_idx
                @inbounds row = AT_col_val[p]
                @inbounds a = AT_nz_val[p]
                if keep_row[row] != UInt8(0) && abs(a) > zero_tol
                    count += Int32(1)
                end
            end
        end
        @inbounds col_nnz[col] = count
    end
    return
end

function _kernel_doubleton_eq_candidates!(
    candidate_mask,
    candidate_elim_col,
    candidate_elim_val,
    candidate_keep_col,
    candidate_keep_val,
    row_ptr,
    col_val,
    nz_val,
    keep_row,
    keep_col,
    col_nnz,
    AL,
    AU,
    zero_tol,
    tol,
    ratio_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && keep_row[i] != UInt8(0)
        @inbounds lhs = AL[i]
        @inbounds rhs = AU[i]
        if !isfinite(lhs) || !isfinite(rhs) || abs(lhs - rhs) > tol
            return
        end

        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - Int32(1)
        live_count = Int32(0)
        col1 = Int32(0)
        col2 = Int32(0)
        val1 = 0.0
        val2 = 0.0

        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                @inbounds a = nz_val[p]
                if keep_col[col] != UInt8(0) && abs(a) > zero_tol
                    live_count += Int32(1)
                    if live_count == Int32(1)
                        col1 = col
                        val1 = a
                    elseif live_count == Int32(2)
                        col2 = col
                        val2 = a
                    else
                        return
                    end
                end
            end
        end

        live_count == Int32(2) || return
        elim_col, elim_val, keep_col_j, keep_val = _choose_doubleton_eq_columns(
            col1,
            val1,
            col2,
            val2,
            col_nnz,
            ratio_tol,
        )
        _is_acceptable_doubleton_eq_pivot(keep_val, elim_val) || return

        @inbounds candidate_mask[i] = UInt8(1)
        @inbounds candidate_elim_col[i] = elim_col
        @inbounds candidate_elim_val[i] = elim_val
        @inbounds candidate_keep_col[i] = keep_col_j
        @inbounds candidate_keep_val[i] = keep_val
    end
    return
end

function _kernel_select_doubleton_eq_target_row!(
    target_row_ref,
    candidate_mask,
    candidate_keep_col,
    candidate_elim_col,
    col_nnz,
    AT_row_ptr,
    AT_col_val,
    max_fill_in_proxy,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(candidate_mask[i] != UInt8(0))
        keep_col = @inbounds candidate_keep_col[i]
        elim_col = @inbounds candidate_elim_col[i]
        keep_nnz = @inbounds col_nnz[keep_col]
        elim_nnz = @inbounds col_nnz[elim_col]
        if elim_nnz > keep_nnz + max_fill_in_proxy + Int32(1)
            return
        end
        fill_in = _doubleton_eq_fill_in_proxy_device(
            AT_row_ptr,
            AT_col_val,
            keep_col,
            elim_col,
        )
        if fill_in <= max_fill_in_proxy
            CUDA.@atomic target_row_ref[1] = min(target_row_ref[1], Int32(i))
        end
    end
    return
end

function _kernel_select_doubleton_eq_target_row_direct!(
    target_row_ref,
    row_ptr,
    col_val,
    nz_val,
    keep_row,
    keep_col,
    col_nnz,
    AL,
    AU,
    AT_row_ptr,
    AT_col_val,
    zero_tol,
    tol,
    ratio_tol,
    max_fill_in_proxy,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && keep_row[i] != UInt8(0)
        @inbounds lhs = AL[i]
        @inbounds rhs = AU[i]
        if !isfinite(lhs) || !isfinite(rhs) || abs(lhs - rhs) > tol
            return
        end

        @inbounds row_start = row_ptr[i]
        @inbounds row_stop = row_ptr[i + 1] - Int32(1)
        live_count = Int32(0)
        col1 = Int32(0)
        col2 = Int32(0)
        val1 = 0.0
        val2 = 0.0

        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = col_val[p]
                @inbounds a = nz_val[p]
                if keep_col[col] != UInt8(0) && abs(a) > zero_tol
                    live_count += Int32(1)
                    if live_count == Int32(1)
                        col1 = col
                        val1 = a
                    elseif live_count == Int32(2)
                        col2 = col
                        val2 = a
                    else
                        return
                    end
                end
            end
        end

        live_count == Int32(2) || return
        elim_col, elim_val, keep_col_j, keep_val = _choose_doubleton_eq_columns(
            col1,
            val1,
            col2,
            val2,
            col_nnz,
            ratio_tol,
        )
        _is_acceptable_doubleton_eq_pivot(keep_val, elim_val) || return

        keep_nnz = @inbounds col_nnz[keep_col_j]
        elim_nnz = @inbounds col_nnz[elim_col]
        if elim_nnz > keep_nnz + max_fill_in_proxy + Int32(1)
            return
        end

        fill_in = _doubleton_eq_fill_in_proxy_device(
            AT_row_ptr,
            AT_col_val,
            keep_col_j,
            elim_col,
        )
        if fill_in <= max_fill_in_proxy
            CUDA.@atomic target_row_ref[1] = min(target_row_ref[1], Int32(i))
        end
    end
    return
end

function _kernel_doubleton_eq_row_counts_and_shifts!(
    row_nnz_new,
    row_shift,
    row_rewrite_required_ref,
    row_ptr,
    col_val,
    nz_val,
    target_row,
    keep_col,
    elim_col,
    alpha,
    beta,
    zero_tol,
    m,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m
        @inbounds src_first = row_ptr[r]
        @inbounds src_last = row_ptr[r + 1] - Int32(1)
        len = src_last >= src_first ? src_last - src_first + Int32(1) : Int32(0)

        if r == target_row
            @inbounds row_nnz_new[r] = len
            @inbounds row_shift[r] = 0.0
            return
        end

        elim_present = false
        keep_present = false
        are = 0.0
        old_keep = 0.0

        if src_first <= src_last
            for p in src_first:src_last
                @inbounds col = col_val[p]
                @inbounds a = nz_val[p]
                if col == elim_col
                    elim_present = true
                    are = a
                elseif col == keep_col
                    keep_present = true
                    old_keep = a
                end
            end
        end

        new_keep = old_keep + alpha * are
        new_len = len
        if elim_present && !keep_present && abs(new_keep) > zero_tol
            new_len += Int32(1)
        elseif keep_present && abs(new_keep) <= zero_tol
            new_len -= Int32(1)
        end
        if new_len != len
            CUDA.@atomic row_rewrite_required_ref[1] = max(row_rewrite_required_ref[1], Int32(1))
        end

        @inbounds row_nnz_new[r] = max(new_len, Int32(0))
        @inbounds row_shift[r] = elim_present ? (are * beta) : 0.0
    end
    return
end

function _kernel_update_doubleton_eq_existing_keep_values!(
    nzVal_work,
    row_ptr,
    col_val,
    nz_val,
    target_row,
    keep_col,
    elim_col,
    alpha,
    m,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m && r != target_row
        @inbounds src_first = row_ptr[r]
        @inbounds src_last = row_ptr[r + 1] - Int32(1)
        src_first <= src_last || return

        elim_present = false
        are = 0.0
        keep_pos = Int32(0)
        old_keep = 0.0

        for p in src_first:src_last
            @inbounds col = col_val[p]
            @inbounds a = nz_val[p]
            if col == elim_col
                elim_present = true
                are = a
            elseif col == keep_col
                keep_pos = p
                old_keep = a
            end
        end

        if elim_present && keep_pos != Int32(0)
            @inbounds nzVal_work[keep_pos] = old_keep + alpha * are
        end
    end
    return
end

function _doubleton_eq_current_elim_support(
    AT_source::CuSparseMatrixCSR{Float64,Int32},
    keep_row::CuVector{UInt8},
    elim_col::Integer,
    target_row::Integer,
)
    col_start = Int(_copy_scalar_to_host(AT_source.rowPtr, Int(elim_col)))
    col_stop = Int(_copy_scalar_to_host(AT_source.rowPtr, Int(elim_col) + 1)) - 1
    if col_start > col_stop
        return (Int32[], Float64[])
    end

    rows = _copy_vector_to_host(AT_source.colVal[col_start:col_stop])
    vals = _copy_vector_to_host(AT_source.nzVal[col_start:col_stop])
    keep_row_h = _copy_vector_to_host(keep_row)
    keep_rows = Int32[]
    keep_vals = Float64[]
    sizehint!(keep_rows, length(rows))
    sizehint!(keep_vals, length(vals))
    target_i32 = Int32(target_row)
    for t in eachindex(rows)
        rows[t] == target_i32 && continue
        keep_row_h[Int(rows[t])] == UInt8(0) && continue
        push!(keep_rows, rows[t])
        push!(keep_vals, vals[t])
    end
    return (keep_rows, keep_vals)
end

function _kernel_pack_doubleton_eq_record_gpu!(
    indices,
    vals,
    AT_rowPtr,
    AT_colVal,
    AT_nzVal,
    elim_col,
    keep_col,
    target_row,
    elim_val,
    keep_val,
    rhs,
    old_elim_l,
    old_elim_u,
    old_keep_l,
    old_keep_u,
    keep_l_new,
    keep_u_new,
    elim_obj,
    lower_source,
    upper_source,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds indices[1] = elim_col
        @inbounds indices[2] = keep_col
        @inbounds indices[3] = target_row

        @inbounds col_start = AT_rowPtr[elim_col]
        @inbounds col_stop = AT_rowPtr[elim_col + Int32(1)] - Int32(1)
        support_count = Int32(0)
        if col_start <= col_stop
            for p in col_start:col_stop
                @inbounds row = AT_colVal[p]
                if row != target_row
                    support_count += Int32(1)
                    @inbounds indices[4 + support_count] = row
                    @inbounds vals[12 + support_count] = AT_nzVal[p]
                end
            end
        end
        @inbounds indices[4] = support_count

        @inbounds vals[1] = elim_val
        @inbounds vals[2] = keep_val
        @inbounds vals[3] = rhs
        @inbounds vals[4] = old_elim_l
        @inbounds vals[5] = old_elim_u
        @inbounds vals[6] = old_keep_l
        @inbounds vals[7] = old_keep_u
        @inbounds vals[8] = keep_l_new
        @inbounds vals[9] = keep_u_new
        @inbounds vals[10] = elim_obj
        @inbounds vals[11] = lower_source
        @inbounds vals[12] = upper_source
    end
    return
end

function _append_doubleton_eq_record_gpu!(
    tape::PostsolveTape_gpu,
    AT_source::CuSparseMatrixCSR{Float64,Int32},
    elim_col::Integer,
    keep_col::Integer,
    target_row::Integer,
    elim_val::Float64,
    keep_val::Float64,
    rhs::Float64,
    old_elim_l::Float64,
    old_elim_u::Float64,
    old_keep_l::Float64,
    old_keep_u::Float64,
    keep_l_new::Float64,
    keep_u_new::Float64,
    elim_obj::Float64,
    lower_source::Float64,
    upper_source::Float64,
)
    col_start = Int(_copy_scalar_to_host(AT_source.rowPtr, Int(elim_col)))
    col_stop = Int(_copy_scalar_to_host(AT_source.rowPtr, Int(elim_col) + 1)) - 1
    support_count = max(col_stop - col_start, 0)
    indices = CuVector{Int32}(undef, 4 + support_count)
    vals = CuVector{Float64}(undef, 12 + support_count)
    @cuda threads=1 blocks=1 _kernel_pack_doubleton_eq_record_gpu!(
        indices,
        vals,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        Int32(elim_col),
        Int32(keep_col),
        Int32(target_row),
        elim_val,
        keep_val,
        rhs,
        old_elim_l,
        old_elim_u,
        old_keep_l,
        old_keep_u,
        keep_l_new,
        keep_u_new,
        elim_obj,
        lower_source,
        upper_source,
    )
    record = PostsolveTape_gpu(
        CuVector(Int32[Int32(DOUBLETON_EQ)]),
        CuVector(Int32[1, Int32(length(indices) + 1)]),
        CuVector(Int32[1, Int32(length(vals) + 1)]),
        CuVector(UInt8[UInt8(POSTSOLVE_DUAL_MINIMAL)]),
        indices,
        vals,
    )
    append_postsolve_tape!(tape, record)
    return tape
end

function _kernel_copy_doubleton_eq_rows!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    target_row,
    keep_col,
    elim_col,
    alpha,
    zero_tol,
    m,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m
        @inbounds src_first = rowPtr_org[r]
        @inbounds src_last = rowPtr_org[r + 1] - Int32(1)
        @inbounds write_ptr = rowPtr_new[r]

        if src_first > src_last
            return
        end

        if r == target_row
            for p in src_first:src_last
                @inbounds colVal_new[write_ptr] = colVal_org[p]
                @inbounds nzVal_new[write_ptr] = nzVal_org[p]
                write_ptr += Int32(1)
            end
            return
        end

        elim_present = false
        keep_present = false
        are = 0.0
        old_keep = 0.0

        for p in src_first:src_last
            @inbounds col = colVal_org[p]
            @inbounds a = nzVal_org[p]
            if col == elim_col
                elim_present = true
                are = a
            elseif col == keep_col
                keep_present = true
                old_keep = a
            end
        end

        new_keep = old_keep + alpha * are
        insert_keep = elim_present && !keep_present && abs(new_keep) > zero_tol
        inserted = !insert_keep

        for p in src_first:src_last
            @inbounds col = colVal_org[p]
            @inbounds a = nzVal_org[p]

            if insert_keep && !inserted && col > keep_col
                @inbounds colVal_new[write_ptr] = keep_col
                @inbounds nzVal_new[write_ptr] = new_keep
                write_ptr += Int32(1)
                inserted = true
            end

            if col == keep_col
                if abs(new_keep) > zero_tol
                    @inbounds colVal_new[write_ptr] = keep_col
                    @inbounds nzVal_new[write_ptr] = new_keep
                    write_ptr += Int32(1)
                end
            else
                @inbounds colVal_new[write_ptr] = col
                @inbounds nzVal_new[write_ptr] = a
                write_ptr += Int32(1)
            end
        end

        if insert_keep && !inserted
            @inbounds colVal_new[write_ptr] = keep_col
            @inbounds nzVal_new[write_ptr] = new_keep
        end
    end
    return
end

function _kernel_pack_doubleton_eq_metadata!(
    meta_i32,
    meta_f64,
    target_row_ref,
    candidate_elim_col,
    candidate_elim_val,
    candidate_keep_col,
    candidate_keep_val,
    AU,
    l,
    u,
    c,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds target_row = target_row_ref[1]
        @inbounds elim_col = candidate_elim_col[target_row]
        @inbounds keep_col = candidate_keep_col[target_row]

        @inbounds meta_i32[1] = target_row
        @inbounds meta_i32[2] = elim_col
        @inbounds meta_i32[3] = keep_col

        @inbounds meta_f64[1] = candidate_elim_val[target_row]
        @inbounds meta_f64[2] = candidate_keep_val[target_row]
        @inbounds meta_f64[3] = AU[target_row]
        @inbounds meta_f64[4] = l[elim_col]
        @inbounds meta_f64[5] = u[elim_col]
        @inbounds meta_f64[6] = c[elim_col]
        @inbounds meta_f64[7] = l[keep_col]
        @inbounds meta_f64[8] = u[keep_col]
    end
    return
end

function _kernel_pack_doubleton_eq_metadata_direct!(
    meta_i32,
    meta_f64,
    target_row_ref,
    row_ptr,
    col_val,
    nz_val,
    keep_col_mask,
    col_nnz,
    AU,
    l,
    u,
    c,
    zero_tol,
    ratio_tol,
    m,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds target_row = target_row_ref[1]
        @inbounds meta_i32[1] = target_row
        if target_row < Int32(1) || target_row > m
            @inbounds meta_i32[2] = Int32(0)
            @inbounds meta_i32[3] = Int32(0)
            return
        end

        @inbounds row_start = row_ptr[target_row]
        @inbounds row_stop = row_ptr[target_row + Int32(1)] - Int32(1)
        live_count = Int32(0)
        col1 = Int32(0)
        col2 = Int32(0)
        val1 = 0.0
        val2 = 0.0

        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col_j = col_val[p]
                @inbounds a = nz_val[p]
                if keep_col_mask[col_j] != UInt8(0) && abs(a) > zero_tol
                    live_count += Int32(1)
                    if live_count == Int32(1)
                        col1 = col_j
                        val1 = a
                    elseif live_count == Int32(2)
                        col2 = col_j
                        val2 = a
                    end
                end
            end
        end

        elim_col, elim_val, keep_col_j, keep_val = _choose_doubleton_eq_columns(
            col1,
            val1,
            col2,
            val2,
            col_nnz,
            ratio_tol,
        )

        @inbounds meta_i32[2] = elim_col
        @inbounds meta_i32[3] = keep_col_j

        @inbounds meta_f64[1] = elim_val
        @inbounds meta_f64[2] = keep_val
        @inbounds meta_f64[3] = AU[target_row]
        @inbounds meta_f64[4] = l[elim_col]
        @inbounds meta_f64[5] = u[elim_col]
        @inbounds meta_f64[6] = c[elim_col]
        @inbounds meta_f64[7] = l[keep_col_j]
        @inbounds meta_f64[8] = u[keep_col_j]
    end
    return
end

function _kernel_apply_doubleton_eq_scalar_updates!(
    l,
    u,
    c,
    AL,
    AU,
    keep_row_mask,
    keep_col_mask,
    row_shift,
    target_row,
    keep_col,
    elim_col,
    keep_l_new,
    keep_u_new,
    obj_delta,
    delete_row,
    delete_col,
    m,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= m
        @inbounds AL[t] -= row_shift[t]
        @inbounds AU[t] -= row_shift[t]
    end

    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds l[keep_col] = keep_l_new
        @inbounds u[keep_col] = keep_u_new
        @inbounds c[keep_col] += obj_delta
        if delete_row != UInt8(0)
            @inbounds keep_row_mask[target_row] = UInt8(0)
        end
        if delete_col != UInt8(0)
            @inbounds keep_col_mask[elim_col] = UInt8(0)
        end
    end
    return
end

function _kernel_mark_doubleton_eq_batch_acceptable!(
    acceptable_mask,
    candidate_mask,
    candidate_keep_col,
    candidate_elim_col,
    col_nnz,
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    zero_tol,
    max_fill_in_proxy,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(candidate_mask[i] != UInt8(0))
        keep_col = @inbounds candidate_keep_col[i]
        elim_col = @inbounds candidate_elim_col[i]
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
        if fill_in <= max_fill_in_proxy
            @inbounds acceptable_mask[i] = UInt8(1)
        end
    end
    return
end

function _kernel_claim_doubleton_eq_batch_columns!(
    col_owner,
    active_mask,
    blocked_col,
    candidate_keep_col,
    candidate_elim_col,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(active_mask[i] != UInt8(0))
        keep_col = @inbounds candidate_keep_col[i]
        elim_col = @inbounds candidate_elim_col[i]
        if @inbounds(blocked_col[keep_col] == UInt8(0) && blocked_col[elim_col] == UInt8(0))
            CUDA.@atomic col_owner[keep_col] = min(col_owner[keep_col], Int32(i))
            CUDA.@atomic col_owner[elim_col] = min(col_owner[elim_col], Int32(i))
        end
    end
    return
end

function _kernel_select_doubleton_eq_batch_rows!(
    selected_mask,
    active_mask,
    blocked_col,
    candidate_keep_col,
    candidate_elim_col,
    col_owner,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m && @inbounds(active_mask[i] != UInt8(0))
        keep_col = @inbounds candidate_keep_col[i]
        elim_col = @inbounds candidate_elim_col[i]
        if @inbounds(
            blocked_col[keep_col] == UInt8(0) &&
            blocked_col[elim_col] == UInt8(0) &&
            col_owner[keep_col] == Int32(i) &&
            col_owner[elim_col] == Int32(i)
        )
            @inbounds selected_mask[i] = UInt8(1)
            @inbounds active_mask[i] = UInt8(0)
            @inbounds blocked_col[keep_col] = UInt8(1)
            @inbounds blocked_col[elim_col] = UInt8(1)
        end
    end
    return
end

function _kernel_doubleton_eq_batch_bound_transfer!(
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
    keep_val,
    elim_val,
    rhs,
    old_elim_l,
    old_elim_u,
    old_keep_l,
    old_keep_u,
    tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds a_keep = keep_val[t]
        @inbounds a_elim = elim_val[t]
        @inbounds rhs_t = rhs[t]
        alpha_t = -a_keep / a_elim
        beta_t = rhs_t / a_elim

        mapped_l = -Inf
        mapped_u = Inf

        @inbounds elim_l = old_elim_l[t]
        if isfinite(elim_l)
            bound_val = (elim_l - beta_t) / alpha_t
            if alpha_t > 0.0
                mapped_l = bound_val
            else
                mapped_u = bound_val
            end
        end

        @inbounds elim_u = old_elim_u[t]
        if isfinite(elim_u)
            bound_val = (elim_u - beta_t) / alpha_t
            if alpha_t > 0.0
                mapped_u = bound_val
            else
                mapped_l = bound_val
            end
        end

        if mapped_l > mapped_u
            mapped_l, mapped_u = mapped_u, mapped_l
        end

        @inbounds keep_l_old = old_keep_l[t]
        @inbounds keep_u_old = old_keep_u[t]
        keep_l_t = isfinite(mapped_l) ? max(keep_l_old, mapped_l) : keep_l_old
        keep_u_t = isfinite(mapped_u) ? min(keep_u_old, mapped_u) : keep_u_old

        @inbounds keep_l_new[t] = keep_l_t
        @inbounds keep_u_new[t] = keep_u_t
        @inbounds alpha[t] = alpha_t
        @inbounds beta[t] = beta_t
        @inbounds lower_source[t] = _doubleton_eq_lower_source(
            keep_l_old,
            mapped_l,
            keep_l_t,
            tol,
        )
        @inbounds upper_source[t] = _doubleton_eq_upper_source(
            keep_u_old,
            mapped_u,
            keep_u_t,
            tol,
        )

        if keep_l_t > keep_u_t + tol
            @inbounds infeas_flag[1] = Int32(1)
            CUDA.@atomic infeas_row_ref[1] = min(infeas_row_ref[1], @inbounds(selected_rows[t]))
            return
        end

        fixed = isfinite(keep_l_t) && isfinite(keep_u_t) && keep_u_t <= keep_l_t + tol
        @inbounds keep_fixed_mask[t] = fixed ? UInt8(1) : UInt8(0)
        @inbounds fixed_at[t] = fixed ? 0.5 * (keep_l_t + keep_u_t) : 0.0
    end
    return
end

function _kernel_apply_doubleton_eq_batch_decisions!(
    l,
    u,
    c,
    fixed_col_mask,
    fixed_val,
    row_delete,
    col_delete,
    selected_rows,
    keep_col,
    elim_col,
    keep_l_new,
    keep_u_new,
    alpha,
    elim_obj,
    keep_fixed_mask,
    fixed_at,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds row = selected_rows[t]
        @inbounds keep = keep_col[t]
        @inbounds elim = elim_col[t]

        if @inbounds(keep_fixed_mask[t] != UInt8(0))
            @inbounds val = fixed_at[t]
            @inbounds l[keep] = val
            @inbounds u[keep] = val
            @inbounds fixed_col_mask[keep] = UInt8(1)
            @inbounds fixed_val[keep] = val
        else
            @inbounds l[keep] = keep_l_new[t]
            @inbounds u[keep] = keep_u_new[t]
            @inbounds c[keep] += elim_obj[t] * alpha[t]
            @inbounds row_delete[row] = UInt8(1)
            @inbounds col_delete[elim] = UInt8(1)
        end
    end
    return
end

function _kernel_doubleton_eq_subst_entry_counts!(
    entry_counts,
    subst_rows,
    subst_elim_col,
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds target_row = subst_rows[t]
        @inbounds elim = subst_elim_col[t]
        @inbounds start_idx = AT_row_ptr[elim]
        @inbounds stop_idx = AT_row_ptr[elim + 1] - Int32(1)
        count = Int32(0)
        if start_idx <= stop_idx
            for p in start_idx:stop_idx
                @inbounds row = AT_col_val[p]
                @inbounds a = AT_nz_val[p]
                if row != target_row && keep_row[row] != UInt8(0) && abs(a) > zero_tol
                    count += Int32(1)
                end
            end
        end
        @inbounds entry_counts[t] = count
    end
    return
end

function _kernel_build_doubleton_eq_subst_entries!(
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
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds target_row = subst_rows[t]
        @inbounds keep = subst_keep_col[t]
        @inbounds elim = subst_elim_col[t]
        @inbounds alpha_t = subst_alpha[t]
        @inbounds beta_t = subst_beta[t]
        @inbounds write_pos = entry_starts[t]
        @inbounds start_idx = AT_row_ptr[elim]
        @inbounds stop_idx = AT_row_ptr[elim + 1] - Int32(1)

        if start_idx <= stop_idx
            for p in start_idx:stop_idx
                @inbounds row = AT_col_val[p]
                @inbounds a = AT_nz_val[p]
                if row != target_row && keep_row[row] != UInt8(0) && abs(a) > zero_tol
                    @inbounds delta_rows[write_pos] = row
                    @inbounds delta_cols[write_pos] = keep
                    @inbounds delta_vals[write_pos] = alpha_t * a
                    CUDA.@atomic row_shift[row] += a * beta_t
                    write_pos += Int32(1)
                end
            end
        end
    end
    return
end

function _kernel_accumulate_doubleton_eq_fixed_row_shift!(
    row_shift,
    fixed_keep_col,
    fixed_val,
    keep_row,
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds keep = fixed_keep_col[t]
        @inbounds val = fixed_val[t]
        @inbounds start_idx = AT_row_ptr[keep]
        @inbounds stop_idx = AT_row_ptr[keep + 1] - Int32(1)

        if start_idx <= stop_idx
            for p in start_idx:stop_idx
                @inbounds row = AT_col_val[p]
                @inbounds a = AT_nz_val[p]
                if keep_row[row] != UInt8(0) && abs(a) > zero_tol
                    CUDA.@atomic row_shift[row] += a * val
                end
            end
        end
    end
    return
end

function _kernel_pack_doubleton_eq_selected_metadata!(
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
    AU,
    l,
    u,
    c,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds row = selected_rows[t]
        @inbounds keep = candidate_keep_col[row]
        @inbounds elim = candidate_elim_col[row]
        @inbounds selected_keep_col[t] = keep
        @inbounds selected_elim_col[t] = elim
        @inbounds selected_keep_val[t] = candidate_keep_val[row]
        @inbounds selected_elim_val[t] = candidate_elim_val[row]
        @inbounds selected_rhs[t] = AU[row]
        @inbounds selected_old_elim_l[t] = l[elim]
        @inbounds selected_old_elim_u[t] = u[elim]
        @inbounds selected_elim_obj[t] = c[elim]
        @inbounds selected_old_keep_l[t] = l[keep]
        @inbounds selected_old_keep_u[t] = u[keep]
    end
    return
end

function _kernel_pack_doubleton_eq_subst_metadata!(
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
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds idx = subst_pair_idx[t]
        @inbounds subst_rows[t] = selected_rows[idx]
        @inbounds subst_keep_col[t] = selected_keep_col[idx]
        @inbounds subst_elim_col[t] = selected_elim_col[idx]
        @inbounds subst_keep_val[t] = selected_keep_val[idx]
        @inbounds subst_elim_val[t] = selected_elim_val[idx]
        @inbounds subst_rhs[t] = selected_rhs[idx]
        @inbounds subst_old_elim_l[t] = selected_old_elim_l[idx]
        @inbounds subst_old_elim_u[t] = selected_old_elim_u[idx]
        @inbounds subst_old_keep_l[t] = selected_old_keep_l[idx]
        @inbounds subst_old_keep_u[t] = selected_old_keep_u[idx]
        @inbounds subst_new_keep_l[t] = keep_l_new[idx]
        @inbounds subst_new_keep_u[t] = keep_u_new[idx]
        @inbounds subst_elim_obj[t] = selected_elim_obj[idx]
        @inbounds subst_alpha[t] = alpha[idx]
        @inbounds subst_beta[t] = beta[idx]
        @inbounds subst_lower_source[t] = lower_source[idx]
        @inbounds subst_upper_source[t] = upper_source[idx]
    end
    return
end

function _kernel_pack_doubleton_eq_fixed_metadata!(
    fixed_keep_col,
    fixed_keep_val,
    fixed_pair_idx,
    selected_keep_col,
    fixed_at,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds idx = fixed_pair_idx[t]
        @inbounds fixed_keep_col[t] = selected_keep_col[idx]
        @inbounds fixed_keep_val[t] = fixed_at[idx]
    end
    return
end

function _kernel_count_doubleton_eq_delta_rows!(
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

function _kernel_scatter_doubleton_eq_delta_csr!(
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

function _build_doubleton_eq_delta_csr(
    delta_rows::CuVector{Int32},
    delta_cols::CuVector{Int32},
    delta_vals::CuVector{Float64},
    dims::Tuple{Int,Int},
)
    nnz = length(delta_rows)
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
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_count_doubleton_eq_delta_rows!(
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
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_scatter_doubleton_eq_delta_csr!(
        csr_cols,
        csr_vals,
        row_write_ptr,
        delta_rows,
        delta_cols,
        delta_vals,
        Int32(nnz),
    )

    return CuSparseMatrixCSR(row_ptr, csr_cols, csr_vals, (m, n))
end

function _kernel_count_doubleton_eq_nonzero_entries!(
    row_nnz_new,
    rowPtr_org,
    nzVal_org,
    zero_tol,
    m,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m
        @inbounds src_first = rowPtr_org[r]
        @inbounds src_last = rowPtr_org[r + 1] - 1
        cnt = Int32(0)
        if src_first <= src_last
            for p in src_first:src_last
                @inbounds if abs(nzVal_org[p]) > zero_tol
                    cnt += Int32(1)
                end
            end
        end
        @inbounds row_nnz_new[r] = cnt
    end
    return
end

function _kernel_copy_doubleton_eq_nonzero_entries!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    zero_tol,
    m,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m
        @inbounds src_first = rowPtr_org[r]
        @inbounds src_last = rowPtr_org[r + 1] - 1
        @inbounds write_ptr = rowPtr_new[r]
        if src_first <= src_last
            for p in src_first:src_last
                @inbounds val = nzVal_org[p]
                if abs(val) > zero_tol
                    @inbounds colVal_new[write_ptr] = colVal_org[p]
                    @inbounds nzVal_new[write_ptr] = val
                    write_ptr += Int32(1)
                end
            end
        end
    end
    return
end

function _prune_doubleton_eq_structural_zeros(
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    zero_tol::Float64,
)
    m, n = size(A_csr)
    if m == 0
        rowPtr_new = CUDA.fill(Int32(1), 1)
        colVal_new = CuVector{Int32}(undef, 0)
        nzVal_new = CuVector{Float64}(undef, 0)
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (0, n))
    end

    row_nnz_new = CUDA.zeros(Int32, m)
    blocks_rows = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_count_doubleton_eq_nonzero_entries!(
        row_nnz_new,
        A_csr.rowPtr,
        A_csr.nzVal,
        zero_tol,
        Int32(m),
    )

    prefix = cumsum(row_nnz_new)
    rowPtr_new = CUDA.fill(Int32(1), m + 1)
    rowPtr_new[2:end] .= prefix .+ Int32(1)
    nnz_new = Int(_copy_scalar_to_host(prefix, m))
    colVal_new = CuVector{Int32}(undef, nnz_new)
    nzVal_new = CuVector{Float64}(undef, nnz_new)

    if nnz_new > 0
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_copy_doubleton_eq_nonzero_entries!(
            colVal_new,
            nzVal_new,
            rowPtr_new,
            A_csr.rowPtr,
            A_csr.colVal,
            A_csr.nzVal,
            zero_tol,
            Int32(m),
        )
    end

    return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m, n))
end

function _kernel_pack_doubleton_eq_batch_records_gpu!(
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
    AT_row_ptr,
    AT_col_val,
    AT_nz_val,
    keep_row,
    zero_tol,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        @inbounds idx0 = idx_starts[t]
        @inbounds val0 = val_starts[t]
        @inbounds row = subst_rows[t]
        @inbounds keep = subst_keep_col[t]
        @inbounds elim = subst_elim_col[t]
        @inbounds support_count = support_counts[t]

        @inbounds indices[idx0] = elim
        @inbounds indices[idx0 + Int32(1)] = keep
        @inbounds indices[idx0 + Int32(2)] = row
        @inbounds indices[idx0 + Int32(3)] = support_count

        @inbounds vals[val0] = subst_elim_val[t]
        @inbounds vals[val0 + Int32(1)] = subst_keep_val[t]
        @inbounds vals[val0 + Int32(2)] = subst_rhs[t]
        @inbounds vals[val0 + Int32(3)] = subst_old_elim_l[t]
        @inbounds vals[val0 + Int32(4)] = subst_old_elim_u[t]
        @inbounds vals[val0 + Int32(5)] = subst_old_keep_l[t]
        @inbounds vals[val0 + Int32(6)] = subst_old_keep_u[t]
        @inbounds vals[val0 + Int32(7)] = subst_new_keep_l[t]
        @inbounds vals[val0 + Int32(8)] = subst_new_keep_u[t]
        @inbounds vals[val0 + Int32(9)] = subst_elim_obj[t]
        @inbounds vals[val0 + Int32(10)] = subst_lower_source[t]
        @inbounds vals[val0 + Int32(11)] = subst_upper_source[t]

        write = Int32(0)
        @inbounds col_start = AT_row_ptr[elim]
        @inbounds col_stop = AT_row_ptr[elim + Int32(1)] - Int32(1)
        if col_start <= col_stop
            for p in col_start:col_stop
                @inbounds support_row = AT_col_val[p]
                @inbounds support_val = AT_nz_val[p]
                if support_row != row && keep_row[support_row] != UInt8(0) && abs(support_val) > zero_tol
                    write += Int32(1)
                    if write <= support_count
                        @inbounds indices[idx0 + Int32(3) + write] = support_row
                        @inbounds vals[val0 + Int32(11) + write] = support_val
                    end
                end
            end
        end
    end
    return
end

function _append_doubleton_eq_batch_postsolve_records!(
    plan::PresolvePlan_gpu,
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
        rows_h = Array(subst_rows)
        elim_h = Array(subst_elim_col)
        keep_h = Array(subst_keep_col)
        keep_val_h = Array(subst_keep_val)
        elim_val_h = Array(subst_elim_val)
        rhs_h = Array(subst_rhs)
        old_elim_l_h = Array(subst_old_elim_l)
        old_elim_u_h = Array(subst_old_elim_u)
        old_keep_l_h = Array(subst_old_keep_l)
        old_keep_u_h = Array(subst_old_keep_u)
        new_keep_l_h = Array(subst_new_keep_l)
        new_keep_u_h = Array(subst_new_keep_u)
        elim_obj_h = Array(subst_elim_obj)
        lower_source_h = Array(subst_lower_source)
        upper_source_h = Array(subst_upper_source)
        for t in 1:subst_count
            support_rows, support_coeffs = _doubleton_eq_current_elim_support(
                AT_source,
                keep_row,
                elim_h[t],
                rows_h[t],
            )
            append_doubleton_eq_record!(
                plan.tape,
                elim_h[t],
                keep_h[t],
                rows_h[t],
                elim_val_h[t],
                keep_val_h[t],
                rhs_h[t],
                old_elim_l_h[t],
                old_elim_u_h[t],
                old_keep_l_h[t],
                old_keep_u_h[t],
                new_keep_l_h[t],
                new_keep_u_h[t],
                elim_obj_h[t];
                support_rows=support_rows,
                support_coeffs=support_coeffs,
                lower_source=lower_source_h[t],
                upper_source=upper_source_h[t],
                dual_mode=POSTSOLVE_DUAL_MINIMAL,
            )
        end
    end
    return nothing
end

function _apply_rule_doubleton_eq_batch!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    m = length(plan.keep_row_mask)
    n = length(plan.keep_col_mask)
    if m == 0 || n == 0
        return nothing
    end
    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A
    AT_source = lp.AT
    col_nnz = stats.col_nnz
    profile_batch = _gpu_presolver_env_enabled("GPUPRESOLVER_DOUBLETON_BATCH_PROFILE")
    batch_round = 0
    blocks_rows = cld(m, GPU_PRESOLVE_THREADS)
    blocks_cols = cld(n, GPU_PRESOLVE_THREADS)
    scan_enabled = pparams.doubleton_eq_scan
    batch_round_limit = if scan_enabled && pparams.doubleton_eq_max_batch_rounds > 0
        min(_DOUBLETONEQ_BATCH_INNER_ROUNDS, pparams.doubleton_eq_max_batch_rounds)
    else
        _DOUBLETONEQ_BATCH_INNER_ROUNDS
    end
    scan_start = time()

    while batch_round < batch_round_limit
        if scan_enabled &&
           pparams.doubleton_eq_max_time > 0.0 &&
           isfinite(pparams.doubleton_eq_max_time) &&
           time() - scan_start >= pparams.doubleton_eq_max_time
            profile_batch && println(">>> [doubleton_eq batch] scan stop: max_time")
            break
        end
        batch_round += 1
    candidate_mask = CUDA.zeros(UInt8, m)
    candidate_elim_col = CUDA.fill(Int32(0), m)
    candidate_elim_val = CUDA.zeros(Float64, m)
    candidate_keep_col = CUDA.fill(Int32(0), m)
    candidate_keep_val = CUDA.zeros(Float64, m)
    ratio_tol = max(1e-9, pparams.bound_tol)

    t_stage = time()
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
        col_nnz,
        plan.new_AL,
        plan.new_AU,
        pparams.zero_tol,
        pparams.bound_tol,
        ratio_tol,
        Int32(m),
    )
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] candidates = ", round(time() - t_stage; digits=4), "s")
    end

    candidate_count = Int(sum(Int32.(candidate_mask .!= UInt8(0))))
    if candidate_count == 0
        batch_round == 1 && return nothing
        break
    end

    acceptable_mask = CUDA.zeros(UInt8, m)
    col_owner = CUDA.fill(typemax(Int32), n)
    t_stage = time()
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_mark_doubleton_eq_batch_acceptable!(
        acceptable_mask,
        candidate_mask,
        candidate_keep_col,
        candidate_elim_col,
        col_nnz,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        plan.keep_row_mask,
        pparams.zero_tol,
        Int32(pparams.doubleton_eq_max_fill_in_proxy),
        Int32(m),
    )
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] acceptable = ", round(time() - t_stage; digits=4), "s")
    end
    acceptable_count = scan_enabled ? Int(sum(Int32.(acceptable_mask .!= UInt8(0)))) : 0

    selected_mask = CUDA.zeros(UInt8, m)
    active_mask = copy(acceptable_mask)
    blocked_col = CUDA.zeros(UInt8, n)
    t_stage = time()
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
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] match/select = ", round(time() - t_stage; digits=4), "s")
    end

    t_stage = time()
    _, selected_rows, selected_count = build_maps_from_mask(selected_mask)
    Int(selected_count) == 0 && return nothing
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] compact selected = ", round(time() - t_stage; digits=4), "s (k=", Int(selected_count), ")")
    end
    if scan_enabled
        selected_int = Int(selected_count)
        selected_ratio = acceptable_count == 0 ? 0.0 : selected_int / acceptable_count
        if selected_int < pparams.doubleton_eq_min_selected_per_batch ||
           selected_ratio < pparams.doubleton_eq_min_selected_ratio
            if profile_batch
                println(
                    ">>> [doubleton_eq batch] scan stop: selected=",
                    selected_int,
                    " acceptable=",
                    acceptable_count,
                    " ratio=",
                    round(selected_ratio; digits=6),
                )
            end
            break
        end
    end

    k = Int(selected_count)
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
    t_stage = time()
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
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] pack selected = ", round(time() - t_stage; digits=4), "s")
    end

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

    t_stage = time()
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
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] bound transfer = ", round(time() - t_stage; digits=4), "s")
    end

    if Int(_copy_scalar_to_host(infeas_flag, 1)) != 0
        bad_row = Int(_copy_scalar_to_host(infeas_row_ref, 1))
        plan.has_infeasible = true
        plan.status_message = "Doubleton-equality infeasibility at row $(bad_row): transferred bounds made l > u."
        return nothing
    end

    fixed_col_mask = CUDA.zeros(UInt8, n)
    fixed_val = CUDA.zeros(Float64, n)
    row_delete = CUDA.zeros(UInt8, m)
    col_delete = CUDA.zeros(UInt8, n)

    t_stage = time()
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_sel _kernel_apply_doubleton_eq_batch_decisions!(
        plan.new_l,
        plan.new_u,
        plan.new_c,
        fixed_col_mask,
        fixed_val,
        row_delete,
        col_delete,
        selected_rows,
        selected_keep_col,
        selected_elim_col,
        keep_l_new,
        keep_u_new,
        alpha,
        selected_elim_obj,
        keep_fixed_mask,
        fixed_at,
        Int32(k),
    )
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] apply decisions = ", round(time() - t_stage; digits=4), "s")
    end

    t_stage = time()
    _, subst_pair_idx, subst_count = build_maps_from_mask(UInt8.(keep_fixed_mask .== UInt8(0)))
    _, fixed_pair_idx, fixed_pair_count = build_maps_from_mask(keep_fixed_mask)
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] split fixed/subst = ", round(time() - t_stage; digits=4), "s (subst=", Int(subst_count), ", fixed=", Int(fixed_pair_count), ")")
    end

    subst_rows = CuVector{Int32}(undef, Int(subst_count))
    subst_keep_col = CuVector{Int32}(undef, Int(subst_count))
    subst_elim_col = CuVector{Int32}(undef, Int(subst_count))
    subst_keep_val = CuVector{Float64}(undef, Int(subst_count))
    subst_elim_val = CuVector{Float64}(undef, Int(subst_count))
    subst_rhs = CuVector{Float64}(undef, Int(subst_count))
    subst_old_elim_l = CuVector{Float64}(undef, Int(subst_count))
    subst_old_elim_u = CuVector{Float64}(undef, Int(subst_count))
    subst_old_keep_l = CuVector{Float64}(undef, Int(subst_count))
    subst_old_keep_u = CuVector{Float64}(undef, Int(subst_count))
    subst_new_keep_l = CuVector{Float64}(undef, Int(subst_count))
    subst_new_keep_u = CuVector{Float64}(undef, Int(subst_count))
    subst_elim_obj = CuVector{Float64}(undef, Int(subst_count))
    subst_alpha = CuVector{Float64}(undef, Int(subst_count))
    subst_beta = CuVector{Float64}(undef, Int(subst_count))
    subst_lower_source = CuVector{Float64}(undef, Int(subst_count))
    subst_upper_source = CuVector{Float64}(undef, Int(subst_count))
    if Int(subst_count) > 0
        blocks_subst_meta = cld(Int(subst_count), GPU_PRESOLVE_THREADS)
        t_stage = time()
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst_meta _kernel_pack_doubleton_eq_subst_metadata!(
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
        if profile_batch
            CUDA.synchronize()
            println(">>> [doubleton_eq batch] pack subst = ", round(time() - t_stage; digits=4), "s")
        end
    end
    fixed_keep_col = CuVector{Int32}(undef, Int(fixed_pair_count))
    fixed_keep_val = CuVector{Float64}(undef, Int(fixed_pair_count))
    if Int(fixed_pair_count) > 0
        blocks_fixed_meta = cld(Int(fixed_pair_count), GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_fixed_meta _kernel_pack_doubleton_eq_fixed_metadata!(
            fixed_keep_col,
            fixed_keep_val,
            fixed_pair_idx,
            selected_keep_col,
            fixed_at,
            Int32(fixed_pair_count),
        )
    end

    keep_row_new = UInt8.((plan.keep_row_mask .!= UInt8(0)) .& (row_delete .== UInt8(0)))
    keep_col_new = UInt8.((plan.keep_col_mask .!= UInt8(0)) .& (fixed_col_mask .== UInt8(0)) .& (col_delete .== UInt8(0)))

    row_shift = CUDA.zeros(Float64, m)
    A_new = A_source
    matrix_changed = false
    subst_support_counts = CuVector{Int32}(undef, Int(subst_count))
    if Int(subst_count) > 0
        subst_k = Int(subst_count)
        t_stage = time()
        blocks_subst = cld(subst_k, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_subst _kernel_doubleton_eq_subst_entry_counts!(
            subst_support_counts,
            subst_rows,
            subst_elim_col,
            AT_source.rowPtr,
            AT_source.colVal,
            AT_source.nzVal,
            keep_row_new,
            pparams.zero_tol,
            Int32(subst_k),
        )
        entry_prefix = cumsum(subst_support_counts)
        total_entries = subst_k == 0 ? 0 : Int(_copy_scalar_to_host(entry_prefix, subst_k))
        if profile_batch
            CUDA.synchronize()
            println(">>> [doubleton_eq batch] entry counts = ", round(time() - t_stage; digits=4), "s (nnz=", total_entries, ")")
        end
        if total_entries > 0
            entry_starts = CUDA.fill(Int32(1), subst_k)
            if subst_k > 1
                entry_starts[2:end] .= entry_prefix[1:(end - 1)] .+ Int32(1)
            end

            delta_rows = CuVector{Int32}(undef, total_entries)
            delta_cols = CuVector{Int32}(undef, total_entries)
            delta_vals = CuVector{Float64}(undef, total_entries)
            t_stage = time()
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
                Int32(subst_k),
            )
            if profile_batch
                CUDA.synchronize()
                println(">>> [doubleton_eq batch] build delta = ", round(time() - t_stage; digits=4), "s")
            end

            t_stage = time()
            delta_csr_mode = _doubleton_eq_delta_csr_mode()
            delta_csr = if delta_csr_mode == :direct_sorted || total_entries <= _DOUBLETONEQ_DIRECT_CSR_MAX_ENTRIES
                direct_csr = _build_doubleton_eq_delta_csr(
                    delta_rows,
                    delta_cols,
                    delta_vals,
                    size(A_source),
                )
                # csrsort canonicalizes row-column order for cuSPARSE geam.
                sort_csr(direct_csr)
            else
                delta_coo = CUDA.CUSPARSE.CuSparseMatrixCOO(delta_rows, delta_cols, delta_vals, size(A_source))
                CuSparseMatrixCSR(delta_coo)
            end
            if profile_batch
                CUDA.synchronize()
                println(">>> [doubleton_eq batch] delta csr = ", round(time() - t_stage; digits=4), "s")
            end
            t_stage = time()
            A_tmp = A_source + delta_csr
            A_new = _prune_doubleton_eq_structural_zeros(A_tmp, pparams.zero_tol)
            matrix_changed = true
            if profile_batch
                CUDA.synchronize()
                println(">>> [doubleton_eq batch] sparse add + prune = ", round(time() - t_stage; digits=4), "s")
            end
        end
    end

    if Int(fixed_pair_count) > 0
        fixed_k = Int(fixed_pair_count)
        blocks_fixed = cld(fixed_k, GPU_PRESOLVE_THREADS)
        t_stage = time()
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_fixed _kernel_accumulate_doubleton_eq_fixed_row_shift!(
            row_shift,
            fixed_keep_col,
            fixed_keep_val,
            keep_row_new,
            AT_source.rowPtr,
            AT_source.colVal,
            AT_source.nzVal,
            pparams.zero_tol,
            Int32(fixed_k),
        )
        if profile_batch
            CUDA.synchronize()
            println(">>> [doubleton_eq batch] fixed row shift = ", round(time() - t_stage; digits=4), "s")
        end
    end

    t_stage = time()
    plan.new_AL .-= row_shift
    plan.new_AU .-= row_shift
    if profile_batch
        CUDA.synchronize()
        println(">>> [doubleton_eq batch] row bounds shift = ", round(time() - t_stage; digits=4), "s")
    end

    if Int(fixed_pair_count) > 0
        append_fixed_col_postsolve_records!(
            plan,
            lp,
            pparams,
            fixed_col_mask,
            fixed_val,
            plan.new_c,
            keep_row_new,
        )
        append_plan_fixed_from_mask!(plan, fixed_col_mask, fixed_val)
        plan.obj_constant_delta += _stable_presolve_masked_product_sum(plan.new_c, fixed_val, fixed_col_mask)
    end

    if Int(subst_count) > 0
        _append_doubleton_eq_batch_postsolve_records!(
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
        plan.obj_constant_delta += _stable_presolve_objective_sum(subst_elim_obj .* subst_beta)
    end

    if Int(subst_count) > 0
        plan.new_A = A_new
        plan.new_AT_leading_slack = nothing
        plan.new_AT_slack_after = nothing
        A_source = A_new
    end
    copyto!(plan.keep_row_mask, keep_row_new)
    copyto!(plan.keep_col_mask, keep_col_new)
    if Int(subst_count) > 0
        plan.has_row_action = true
    end
    plan.has_col_action = true
    plan.has_change = true

    if matrix_changed
        AT_source = transpose_csr(A_source)
    end
    col_nnz = CuVector{Int32}(undef, n)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_cols _kernel_doubleton_eq_live_col_nnz!(
        col_nnz,
        AT_source.rowPtr,
        AT_source.colVal,
        AT_source.nzVal,
        plan.keep_row_mask,
        plan.keep_col_mask,
        pparams.zero_tol,
        Int32(n),
    )
    end
    return nothing
end

"""
Rule: apply one previously selected doubleton equality row.
"""
function _apply_rule_doubleton_eq_scalar!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
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

    A_source = isnothing(plan.new_A) ? lp.A : plan.new_A

    ratio_tol = max(1e-9, pparams.bound_tol)

    blocks_rows = cld(m, GPU_PRESOLVE_THREADS)
    target_row_ref = CUDA.fill(Int32(m + 1), 1)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_select_doubleton_eq_target_row_direct!(
        target_row_ref,
        A_source.rowPtr,
        A_source.colVal,
        A_source.nzVal,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.col_nnz,
        plan.new_AL,
        plan.new_AU,
        lp.AT.rowPtr,
        lp.AT.colVal,
        pparams.zero_tol,
        pparams.bound_tol,
        ratio_tol,
        Int32(pparams.doubleton_eq_max_fill_in_proxy),
        Int32(m),
    )
    target_row = Int(_copy_scalar_to_host(target_row_ref, 1))
    target_row > m && return nothing

    target_i32 = CuVector{Int32}(undef, 3)
    target_f64 = CuVector{Float64}(undef, 8)
    @cuda threads=1 blocks=1 _kernel_pack_doubleton_eq_metadata_direct!(
        target_i32,
        target_f64,
        target_row_ref,
        A_source.rowPtr,
        A_source.colVal,
        A_source.nzVal,
        plan.keep_col_mask,
        stats.col_nnz,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        plan.new_c,
        pparams.zero_tol,
        ratio_tol,
        Int32(m),
    )

    target_i32_h = Array(target_i32)
    target_f64_h = Array(target_f64)
    target_row = Int(target_i32_h[1])
    elim_col = Int(target_i32_h[2])
    keep_col = Int(target_i32_h[3])
    elim_val = target_f64_h[1]
    keep_val = target_f64_h[2]
    rhs = target_f64_h[3]
    old_elim_l = target_f64_h[4]
    old_elim_u = target_f64_h[5]
    elim_obj = target_f64_h[6]
    old_keep_l = target_f64_h[7]
    old_keep_u = target_f64_h[8]

    mapped_l, mapped_u = _doubleton_eq_mapped_interval(
        rhs,
        keep_val,
        elim_val,
        old_elim_l,
        old_elim_u,
    )

    keep_l_new = isfinite(mapped_l) ? max(old_keep_l, mapped_l) : old_keep_l
    keep_u_new = isfinite(mapped_u) ? min(old_keep_u, mapped_u) : old_keep_u
    lower_source = _doubleton_eq_lower_source(
        old_keep_l,
        mapped_l,
        keep_l_new,
        pparams.bound_tol,
    )
    upper_source = _doubleton_eq_upper_source(
        old_keep_u,
        mapped_u,
        keep_u_new,
        pparams.bound_tol,
    )
    if keep_l_new > keep_u_new + pparams.bound_tol
        plan.has_infeasible = true
        plan.status_message = "Doubleton-equality infeasibility at row $(target_row): transferred bounds made l > u."
    end
    plan.has_infeasible && return nothing

    keep_fixed = isfinite(keep_l_new) && isfinite(keep_u_new) && keep_u_new <= keep_l_new + pparams.bound_tol

    beta = rhs / elim_val
    alpha = -keep_val / elim_val

    if keep_fixed
        fixed_at = 0.5 * (keep_l_new + keep_u_new)

        fixed_mask = CUDA.zeros(UInt8, n)
        fixed_val = CUDA.zeros(Float64, n)
        fixed_mask[keep_col:keep_col] .= UInt8(1)
        fixed_val[keep_col:keep_col] .= fixed_at

        row_shift = CUDA.zeros(Float64, m)
        blocks_cols = cld(n, GPU_PRESOLVE_THREADS)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_cols _kernel_dual_fix_row_shift!(
            row_shift,
            fixed_mask,
            CUDA.zeros(UInt8, n),
            fixed_val,
            plan.keep_row_mask,
            lp.AT.rowPtr,
            lp.AT.colVal,
            lp.AT.nzVal,
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

        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_apply_doubleton_eq_scalar_updates!(
            plan.new_l,
            plan.new_u,
            plan.new_c,
            plan.new_AL,
            plan.new_AU,
            plan.keep_row_mask,
            plan.keep_col_mask,
            row_shift,
            Int32(target_row),
            Int32(keep_col),
            Int32(keep_col),
            fixed_at,
            fixed_at,
            0.0,
            UInt8(0),
            UInt8(1),
            Int32(m),
        )
        plan.has_col_action = true
        plan.has_change = true
        return nothing
    end

    row_nnz_new = CUDA.zeros(Int32, m)
    row_shift = CUDA.zeros(Float64, m)
    row_rewrite_required_ref = CUDA.zeros(Int32, 1)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_doubleton_eq_row_counts_and_shifts!(
        row_nnz_new,
        row_shift,
        row_rewrite_required_ref,
        A_source.rowPtr,
        A_source.colVal,
        A_source.nzVal,
        Int32(target_row),
        Int32(keep_col),
        Int32(elim_col),
        alpha,
        beta,
        pparams.zero_tol,
        Int32(m),
    )

    row_rewrite_required = Int(_copy_scalar_to_host(row_rewrite_required_ref, 1)) != 0
    if row_rewrite_required
        prefix = cumsum(row_nnz_new)
        rowPtr_new = CUDA.fill(Int32(1), m + 1)
        rowPtr_new[2:end] .= prefix .+ Int32(1)
        nnz_new = m == 0 ? 0 : Int(_copy_scalar_to_host(prefix, m))
        colVal_new = CuVector{Int32}(undef, nnz_new)
        nzVal_new = CuVector{Float64}(undef, nnz_new)
        if nnz_new > 0
            @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_copy_doubleton_eq_rows!(
                colVal_new,
                nzVal_new,
                rowPtr_new,
                A_source.rowPtr,
                A_source.colVal,
                A_source.nzVal,
                Int32(target_row),
                Int32(keep_col),
                Int32(elim_col),
                alpha,
                pparams.zero_tol,
                Int32(m),
            )
        end

        plan.new_A = CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, size(A_source))
    else
        nzVal_work = copy(A_source.nzVal)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_update_doubleton_eq_existing_keep_values!(
            nzVal_work,
            A_source.rowPtr,
            A_source.colVal,
            A_source.nzVal,
            Int32(target_row),
            Int32(keep_col),
            Int32(elim_col),
            alpha,
            Int32(m),
        )
        plan.new_A = CuSparseMatrixCSR(A_source.rowPtr, A_source.colVal, nzVal_work, size(A_source))
    end
    plan.new_AT_leading_slack = nothing
    plan.new_AT_slack_after = nothing
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_apply_doubleton_eq_scalar_updates!(
        plan.new_l,
        plan.new_u,
        plan.new_c,
        plan.new_AL,
        plan.new_AU,
        plan.keep_row_mask,
        plan.keep_col_mask,
        row_shift,
        Int32(target_row),
        Int32(keep_col),
        Int32(elim_col),
        keep_l_new,
        keep_u_new,
        elim_obj * alpha,
        UInt8(1),
        UInt8(1),
        Int32(m),
    )
    if pparams.record_postsolve_tape
        _append_doubleton_eq_record_gpu!(
            plan.tape_gpu,
            lp.AT,
            elim_col,
            keep_col,
            target_row,
            elim_val,
            keep_val,
            rhs,
            old_elim_l,
            old_elim_u,
            old_keep_l,
            old_keep_u,
            keep_l_new,
            keep_u_new,
            elim_obj,
            lower_source,
            upper_source,
        )
        if pparams.record_postsolve_tape_cpu
            support_rows, support_coeffs = _doubleton_eq_current_elim_support(
                lp.AT,
                plan.keep_row_mask,
                elim_col,
                target_row,
            )
            append_doubleton_eq_record!(
                plan.tape,
                elim_col,
                keep_col,
                target_row,
                elim_val,
                keep_val,
                rhs,
                old_elim_l,
                old_elim_u,
                old_keep_l,
                old_keep_u,
                keep_l_new,
                keep_u_new,
                elim_obj,
                support_rows=support_rows,
                support_coeffs=support_coeffs,
                lower_source=lower_source,
                upper_source=upper_source,
                dual_mode=POSTSOLVE_DUAL_MINIMAL,
            )
        end
    end
    plan.obj_constant_delta += elim_obj * beta
    plan.has_row_action = true
    plan.has_col_action = true
    plan.has_change = true
    return nothing
end

"""
Rule: apply doubleton equality elimination.

This uses the current GPU batch implementation.
"""
function apply_rule_doubleton_eq!(
    plan::PresolvePlan_gpu,
    lp::LP_info_gpu,
    stats::PresolveStats_gpu,
    pparams::PresolveParams,
)
    if plan.has_infeasible || plan.has_unbounded
        return nothing
    end

    return _apply_rule_doubleton_eq_batch!(plan, lp, stats, pparams)
end
