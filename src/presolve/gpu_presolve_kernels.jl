"""
Low-level GPU primitives for presolve/postsolve.

All kernels assume 1-based indexing consistent with CUDA.jl sparse arrays.
"""

using CUDA
using CUDA: CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR

const GPU_PRESOLVE_THREADS = 256

function _kernel_compute_row_nnz!(
    row_nnz,
    rowPtr,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds row_nnz[i] = rowPtr[i + 1] - rowPtr[i]
    end
    return
end

"""
Compute row nonzero counts from CSR row pointers.
"""
function compute_row_nnz!(
    row_nnz::CuVector{Int32},
    A_csr::CuSparseMatrixCSR,
)
    m, _ = size(A_csr)
    @assert length(row_nnz) == m
    if m == 0
        return nothing
    end
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_row_nnz!(
        row_nnz,
        A_csr.rowPtr,
        Int32(m),
    )
    return nothing
end

"""
Compute original-column nonzero counts using `AT` (CSR of transpose).
"""
function compute_col_nnz!(
    col_nnz::CuVector{Int32},
    AT_csr::CuSparseMatrixCSR,
)
    # `AT` has one row per original column.
    return compute_row_nnz!(col_nnz, AT_csr)
end

function _kernel_singleton_row_support!(
    singleton_col,
    singleton_val,
    row_nnz,
    rowPtr,
    colVal,
    nzVal,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds nnz_i = row_nnz[i]
        if nnz_i == Int32(1)
            @inbounds p = rowPtr[i]
            @inbounds singleton_col[i] = colVal[p]
            @inbounds singleton_val[i] = nzVal[p]
        else
            @inbounds singleton_col[i] = Int32(-1)
            @inbounds singleton_val[i] = 0.0
        end
    end
    return
end

"""
Extract singleton-row support `(col, value)` from CSR matrix.
"""
function compute_singleton_row_support!(
    singleton_col::CuVector{Int32},
    singleton_val::CuVector{Float64},
    row_nnz::CuVector{Int32},
    A_csr::CuSparseMatrixCSR{Float64,Int32},
)
    m, _ = size(A_csr)
    @assert length(singleton_col) == m
    @assert length(singleton_val) == m
    @assert length(row_nnz) == m
    if m == 0
        return nothing
    end
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_singleton_row_support!(
        singleton_col,
        singleton_val,
        row_nnz,
        A_csr.rowPtr,
        A_csr.colVal,
        A_csr.nzVal,
        Int32(m),
    )
    return nothing
end

"""
Extract singleton-column support `(row, value)` from `AT` CSR.
"""
function compute_singleton_col_support!(
    singleton_row::CuVector{Int32},
    singleton_val::CuVector{Float64},
    col_nnz::CuVector{Int32},
    AT_csr::CuSparseMatrixCSR{Float64,Int32},
)
    # In `AT`, rows correspond to original columns.
    return compute_singleton_row_support!(singleton_row, singleton_val, col_nnz, AT_csr)
end

function _kernel_build_maps_from_mask!(
    org2red,
    red2org,
    keep_mask,
    prefix,
    m0,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m0
        @inbounds keep = keep_mask[i] != UInt8(0)
        if keep
            @inbounds r = prefix[i]
            @inbounds org2red[i] = r
            @inbounds red2org[r] = Int32(i)
        else
            @inbounds org2red[i] = Int32(-1)
        end
    end
    return
end

function _copy_vector_to_host(v::AbstractVector{T}) where {T}
    n = length(v)
    out = Vector{T}(undef, n)
    n == 0 && return out
    copyto!(out, 1, v, 1, n)
    return out
end

@inline function _copy_scalar_to_host(v::AbstractVector{T}, idx::Integer) where {T}
    out = Vector{T}(undef, 1)
    copyto!(out, 1, v, idx, 1)
    return out[1]
end

"""
Build compact index maps from a 0/1 keep mask.

Returns `(org2red, red2org, m1)`.
"""
function build_maps_from_mask(keep_mask::CuVector{UInt8})
    m0 = length(keep_mask)
    if m0 == 0
        return (CuVector{Int32}(undef, 0), CuVector{Int32}(undef, 0), Int32(0))
    end

    keep_i32 = Int32.(keep_mask .!= UInt8(0))
    prefix = cumsum(keep_i32)
    m1 = Int32(_copy_scalar_to_host(prefix, m0))

    org2red = CuVector{Int32}(undef, m0)
    red2org = CuVector{Int32}(undef, Int(m1))

    blocks = cld(m0, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_build_maps_from_mask!(
        org2red,
        red2org,
        keep_mask,
        prefix,
        Int32(m0),
    )
    return (org2red, red2org, m1)
end

function build_maps_from_mask(keep_mask::CuVector{Bool})
    return build_maps_from_mask(UInt8.(keep_mask))
end

function _kernel_build_org2red_from_red2org!(
    org2red,
    red2org,
    m1,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m1
        @inbounds org = red2org[r]
        @inbounds org2red[org] = Int32(r)
    end
    return
end

"""
Build `org2red` from `red2org`, filling removed entries with `-1`.
"""
function build_org2red_from_red2org(
    red2org::CuVector{Int32},
    org_size::Integer,
)
    org2red = CUDA.fill(Int32(-1), Int(org_size))
    m1 = length(red2org)
    if m1 == 0
        return org2red
    end
    blocks = cld(m1, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_build_org2red_from_red2org!(
        org2red,
        red2org,
        Int32(m1),
    )
    return org2red
end

"""
Compose red2org maps:
- `global_red2org_old`: old reduced -> original
- `local_red2org`: new reduced -> old reduced
Returns `new reduced -> original`.
"""
function compose_red2org(
    global_red2org_old::CuVector{Int32},
    local_red2org::CuVector{Int32},
)
    return gather_by_red2org(global_red2org_old, local_red2org)
end

function _kernel_row_nnz_from_map!(
    row_nnz_red,
    rowPtr_org,
    row_red2org,
    m1,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m1
        @inbounds org = row_red2org[r]
        @inbounds row_nnz_red[r] = rowPtr_org[org + 1] - rowPtr_org[org]
    end
    return
end

function _kernel_copy_rows_csr!(
    colVal_red,
    nzVal_red,
    rowPtr_red,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    row_red2org,
    m1,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m1
        @inbounds org = row_red2org[r]
        @inbounds src_first = rowPtr_org[org]
        @inbounds src_last = rowPtr_org[org + 1] - 1
        @inbounds dst_first = rowPtr_red[r]

        len = src_last - src_first + 1
        if len > 0
            for k in Int32(0):(len - 1)
                @inbounds dst_idx = dst_first + k
                @inbounds src_idx = src_first + k
                @inbounds colVal_red[dst_idx] = colVal_org[src_idx]
                @inbounds nzVal_red[dst_idx] = nzVal_org[src_idx]
            end
        end
    end
    return
end

"""
Compact CSR by selected rows (`row_red2org`).
"""
function compact_csr_by_rows(
    A_csr::CuSparseMatrixCSR{T,Int32},
    row_red2org::CuVector{Int32},
) where {T}
    m1 = length(row_red2org)
    _, n0 = size(A_csr)

    if m1 == 0
        rowPtr_red = CUDA.fill(Int32(1), 1)
        colVal_red = CuVector{Int32}(undef, 0)
        nzVal_red = CuVector{T}(undef, 0)
        return CuSparseMatrixCSR(rowPtr_red, colVal_red, nzVal_red, (0, n0))
    end

    row_nnz_red = CUDA.zeros(Int32, m1)
    blocks_rows = cld(m1, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_row_nnz_from_map!(
        row_nnz_red,
        A_csr.rowPtr,
        row_red2org,
        Int32(m1),
    )

    prefix = cumsum(row_nnz_red)
    rowPtr_red = CUDA.fill(Int32(1), m1 + 1)
    rowPtr_red[2:end] .= prefix .+ Int32(1)

    nnz_red = Int(_copy_scalar_to_host(prefix, m1))
    colVal_red = CuVector{Int32}(undef, nnz_red)
    nzVal_red = CuVector{T}(undef, nnz_red)

    if nnz_red > 0
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_copy_rows_csr!(
            colVal_red,
            nzVal_red,
            rowPtr_red,
            A_csr.rowPtr,
            A_csr.colVal,
            A_csr.nzVal,
            row_red2org,
            Int32(m1),
        )
    end

    return CuSparseMatrixCSR(rowPtr_red, colVal_red, nzVal_red, (m1, n0))
end

function _kernel_count_kept_cols_per_row!(
    row_nnz_new,
    rowPtr_org,
    colVal_org,
    col_org2red,
    m0,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m0
        @inbounds src_first = rowPtr_org[r]
        @inbounds src_last = rowPtr_org[r + 1] - 1
        cnt = Int32(0)
        if src_first <= src_last
            for p in src_first:src_last
                @inbounds col_org = colVal_org[p]
                @inbounds col_red = col_org2red[col_org]
                if col_red != Int32(-1)
                    cnt += Int32(1)
                end
            end
        end
        @inbounds row_nnz_new[r] = cnt
    end
    return
end

function _kernel_copy_kept_cols_csr!(
    colVal_new,
    nzVal_new,
    rowPtr_new,
    rowPtr_org,
    colVal_org,
    nzVal_org,
    col_org2red,
    m0,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m0
        @inbounds src_first = rowPtr_org[r]
        @inbounds src_last = rowPtr_org[r + 1] - 1
        @inbounds write_ptr = rowPtr_new[r]

        if src_first <= src_last
            for p in src_first:src_last
                @inbounds col_org = colVal_org[p]
                @inbounds col_red = col_org2red[col_org]
                if col_red != Int32(-1)
                    @inbounds colVal_new[write_ptr] = col_red
                    @inbounds nzVal_new[write_ptr] = nzVal_org[p]
                    write_ptr += Int32(1)
                end
            end
        end
    end
    return
end

"""
Compact CSR by selected columns (`col_red2org`).
"""
function compact_csr_by_cols(
    A_csr::CuSparseMatrixCSR{T,Int32},
    col_red2org::CuVector{Int32},
) where {T}
    m0, n0 = size(A_csr)
    n1 = length(col_red2org)

    if m0 == 0
        rowPtr_new = CUDA.fill(Int32(1), 1)
        colVal_new = CuVector{Int32}(undef, 0)
        nzVal_new = CuVector{T}(undef, 0)
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (0, n1))
    end

    if n1 == 0
        rowPtr_new = CUDA.fill(Int32(1), m0 + 1)
        colVal_new = CuVector{Int32}(undef, 0)
        nzVal_new = CuVector{T}(undef, 0)
        return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m0, 0))
    end

    col_org2red = build_org2red_from_red2org(col_red2org, n0)

    row_nnz_new = CUDA.zeros(Int32, m0)
    blocks_rows = cld(m0, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_count_kept_cols_per_row!(
        row_nnz_new,
        A_csr.rowPtr,
        A_csr.colVal,
        col_org2red,
        Int32(m0),
    )

    prefix = cumsum(row_nnz_new)
    rowPtr_new = CUDA.fill(Int32(1), m0 + 1)
    rowPtr_new[2:end] .= prefix .+ Int32(1)

    nnz_new = Int(_copy_scalar_to_host(prefix, m0))
    colVal_new = CuVector{Int32}(undef, nnz_new)
    nzVal_new = CuVector{T}(undef, nnz_new)

    if nnz_new > 0
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks_rows _kernel_copy_kept_cols_csr!(
            colVal_new,
            nzVal_new,
            rowPtr_new,
            A_csr.rowPtr,
            A_csr.colVal,
            A_csr.nzVal,
            col_org2red,
            Int32(m0),
        )
    end

    return CuSparseMatrixCSR(rowPtr_new, colVal_new, nzVal_new, (m0, n1))
end

function _kernel_gather_by_red2org!(
    dst_red,
    src_org,
    red2org,
    m1,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m1
        @inbounds org = red2org[r]
        @inbounds dst_red[r] = src_org[org]
    end
    return
end

"""
Gather `dst_red[r] = src_org[red2org[r]]`.
"""
function gather_by_red2org(
    src_org::CuVector{T},
    red2org::CuVector{Int32},
) where {T}
    m1 = length(red2org)
    dst_red = CuVector{T}(undef, m1)
    if m1 == 0
        return dst_red
    end
    blocks = cld(m1, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_gather_by_red2org!(
        dst_red,
        src_org,
        red2org,
        Int32(m1),
    )
    return dst_red
end

function concat_cuvector(
    a::CuVector{T},
    b::CuVector{T},
) where {T}
    la = length(a)
    lb = length(b)
    if la == 0
        return copy(b)
    elseif lb == 0
        return copy(a)
    end

    out = CuVector{T}(undef, la + lb)
    copyto!(out, 1, a, 1, la)
    copyto!(out, la + 1, b, 1, lb)
    return out
end

function append_plan_fixed_from_mask!(
    plan::PresolvePlan_gpu,
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

function _kernel_scatter_by_red2org!(
    dst_org,
    src_red,
    red2org,
    m1,
)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if r <= m1
        @inbounds org = red2org[r]
        @inbounds dst_org[org] = src_red[r]
    end
    return
end

"""
Scatter `src_red` into original indices using `red2org`.
"""
function scatter_by_red2org!(
    dst_org::CuVector{T},
    src_red::CuVector{T},
    red2org::CuVector{Int32},
) where {T}
    @assert length(src_red) == length(red2org)
    m1 = length(src_red)
    if m1 == 0
        return dst_org
    end
    blocks = cld(m1, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_scatter_by_red2org!(
        dst_org,
        src_red,
        red2org,
        Int32(m1),
    )
    return dst_org
end

function _kernel_compute_row_activity_bounds!(
    row_min,
    row_max,
    rowPtr,
    colVal,
    nzVal,
    l,
    u,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        acc_min = 0.0
        acc_max = 0.0
        @inbounds row_start = rowPtr[i]
        @inbounds row_stop = rowPtr[i + 1] - 1
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds col = colVal[p]
                @inbounds a = nzVal[p]
                @inbounds lj = l[col]
                @inbounds uj = u[col]
                if a >= 0.0
                    acc_min += a * lj
                    acc_max += a * uj
                else
                    acc_min += a * uj
                    acc_max += a * lj
                end
            end
        end
        @inbounds row_min[i] = acc_min
        @inbounds row_max[i] = acc_max
    end
    return
end

function compute_row_activity_bounds!(
    row_min::CuVector{Float64},
    row_max::CuVector{Float64},
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    l::CuVector{Float64},
    u::CuVector{Float64},
)
    m, _ = size(A_csr)
    @assert length(row_min) == m
    @assert length(row_max) == m
    if m == 0
        return nothing
    end
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_row_activity_bounds!(
        row_min,
        row_max,
        A_csr.rowPtr,
        A_csr.colVal,
        A_csr.nzVal,
        l,
        u,
        Int32(m),
    )
    return nothing
end

function _kernel_compute_row_activity_summary!(
    row_min_fin,
    row_max_fin,
    row_min_neg_inf_count,
    row_max_pos_inf_count,
    rowPtr,
    colVal,
    nzVal,
    l,
    u,
    zero_tol,
    m,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        acc_min_fin = 0.0
        acc_max_fin = 0.0
        neg_inf_count = Int32(0)
        pos_inf_count = Int32(0)
        @inbounds row_start = rowPtr[i]
        @inbounds row_stop = rowPtr[i + 1] - 1
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds a = nzVal[p]
                if abs(a) <= zero_tol
                    continue
                end
                @inbounds col = colVal[p]
                @inbounds lj = l[col]
                @inbounds uj = u[col]

                if a >= 0.0
                    term_min = a * lj
                    term_max = a * uj
                else
                    term_min = a * uj
                    term_max = a * lj
                end

                if isfinite(term_min)
                    acc_min_fin += term_min
                elseif term_min < 0.0
                    neg_inf_count += Int32(1)
                end

                if isfinite(term_max)
                    acc_max_fin += term_max
                elseif term_max > 0.0
                    pos_inf_count += Int32(1)
                end
            end
        end
        @inbounds row_min_fin[i] = acc_min_fin
        @inbounds row_max_fin[i] = acc_max_fin
        @inbounds row_min_neg_inf_count[i] = neg_inf_count
        @inbounds row_max_pos_inf_count[i] = pos_inf_count
    end
    return
end

function compute_row_activity_summary!(
    row_min_fin::CuVector{Float64},
    row_max_fin::CuVector{Float64},
    row_min_neg_inf_count::CuVector{Int32},
    row_max_pos_inf_count::CuVector{Int32},
    A_csr::CuSparseMatrixCSR{Float64,Int32},
    l::CuVector{Float64},
    u::CuVector{Float64},
    zero_tol::Float64,
)
    m, _ = size(A_csr)
    @assert length(row_min_fin) == m
    @assert length(row_max_fin) == m
    @assert length(row_min_neg_inf_count) == m
    @assert length(row_max_pos_inf_count) == m
    if m == 0
        return nothing
    end
    blocks = cld(m, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_row_activity_summary!(
        row_min_fin,
        row_max_fin,
        row_min_neg_inf_count,
        row_max_pos_inf_count,
        A_csr.rowPtr,
        A_csr.colVal,
        A_csr.nzVal,
        l,
        u,
        zero_tol,
        Int32(m),
    )
    return nothing
end

@inline function _row_residual_interval_from_activity(
    row_min::Float64,
    row_max::Float64,
    aij::Float64,
    lj::Float64,
    uj::Float64,
)
    if aij >= 0.0
        return (row_min - aij * lj, row_max - aij * uj)
    end
    return (row_min - aij * uj, row_max - aij * lj)
end

@inline function _row_residual_min_from_summary(
    row_min_fin::Float64,
    row_min_neg_inf_count::Int32,
    term_min::Float64,
)
    if isfinite(term_min)
        return row_min_neg_inf_count == Int32(0) ? (row_min_fin - term_min) : -Inf
    end
    if term_min < 0.0
        return row_min_neg_inf_count > Int32(1) ? -Inf : row_min_fin
    end
    return row_min_fin
end

@inline function _row_residual_max_from_summary(
    row_max_fin::Float64,
    row_max_pos_inf_count::Int32,
    term_max::Float64,
)
    if isfinite(term_max)
        return row_max_pos_inf_count == Int32(0) ? (row_max_fin - term_max) : Inf
    end
    if term_max > 0.0
        return row_max_pos_inf_count > Int32(1) ? Inf : row_max_fin
    end
    return row_max_fin
end

function _kernel_compute_col_max_abs!(
    col_max_abs,
    rowPtr,
    nzVal,
    n,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        vmax = 0.0
        @inbounds row_start = rowPtr[j]
        @inbounds row_stop = rowPtr[j + 1] - 1
        if row_start <= row_stop
            for p in row_start:row_stop
                @inbounds vmax = max(vmax, abs(nzVal[p]))
            end
        end
        @inbounds col_max_abs[j] = vmax
    end
    return
end

function compute_col_max_abs!(
    col_max_abs::CuVector{Float64},
    AT_csr::CuSparseMatrixCSR{Float64,Int32},
)
    n, _ = size(AT_csr)
    @assert length(col_max_abs) == n
    if n == 0
        return nothing
    end
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_col_max_abs!(
        col_max_abs,
        AT_csr.rowPtr,
        AT_csr.nzVal,
        Int32(n),
    )
    return nothing
end

"""
Transpose CSR using cuSPARSE through CUDA.jl sparse conversions.
"""
function transpose_csr(A_csr::CuSparseMatrixCSR)
    return CuSparseMatrixCSR(transpose(A_csr))
end
