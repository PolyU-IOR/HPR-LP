# Reusable GPU scratch buffers for one folding pass. Color vectors are grouped
# as `(color_id, color_sig)` tuples when a refinement step needs sorting.
mutable struct FoldingWorkspace
    num_row::Int
    num_col::Int
    num_row_color::Int
    num_col_color::Int
    row_color::CuVector{Tuple{Int32,Float64}}
    col_color::CuVector{Tuple{Int32,Float64}}
    row_color_id::CuVector{Int32}
    col_color_id::CuVector{Int32}
    row_color_sig::CuVector{Float64}
    col_color_sig::CuVector{Float64}
    row_color_start::CuVector{Int32}
    col_color_start::CuVector{Int32}
    row_perm::CuVector{Int32}
    col_perm::CuVector{Int32}
end

function FoldingWorkspace(num_row::Int, num_col::Int)
    return FoldingWorkspace(
        num_row,
        num_col,
        1,
        1,
        CuArray{Tuple{Int32,Float64}}(undef, num_row),
        CuArray{Tuple{Int32,Float64}}(undef, num_col),
        CUDA.zeros(Int32, num_row),
        CUDA.zeros(Int32, num_col),
        CUDA.zeros(Float64, num_row),
        CUDA.zeros(Float64, num_col),
        CUDA.zeros(Int32, num_row),
        CUDA.zeros(Int32, num_col),
        CuArray(Int32.(1:num_row)),
        CuArray(Int32.(1:num_col)),
    )
end

function _ensure_workspace!(
    ws::Union{Nothing,FoldingWorkspace},
    num_row::Int,
    num_col::Int,
)
    if ws === nothing ||
       ws.num_row != num_row ||
       ws.num_col != num_col
        return FoldingWorkspace(num_row, num_col)
    end
    return ws
end

# Deterministic per-round random signatures make refinement reproducible while
# still separating colors by their current matrix neighborhoods.
@inline function _mix64(x::UInt64)
    x += 0x9e3779b97f4a7c15 % UInt64
    x = (x ⊻ (x >> 30)) * (0xbf58476d1ce4e5b9 % UInt64)
    x = (x ⊻ (x >> 27)) * (0x94d049bb133111eb % UInt64)
    return x ⊻ (x >> 31)
end

@inline _color_seed(round::Int, side::UInt64) =
    _mix64(UInt64(round) ⊻ (side * (0x9e3779b97f4a7c15 % UInt64)))

function _randn_color(n::Int, seed::UInt64)
    value = CUDA.zeros(Float64, n)
    CUDA.seed!(seed)
    CUDA.randn!(value)
    return value
end

function _pack_color(
    color::CuDeviceVector{Tuple{Int32,Float64}},
    color_id::CuDeviceVector{Int32},
    color_sig::CuDeviceVector{Float64},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        color[i] = (color_id[i], Float64(Float32(color_sig[i])))
    end
    return nothing
end

function _mark_color_start(
    color_start::CuDeviceVector{Int32},
    color::CuDeviceVector{Tuple{Int32,Float64}},
    perm::CuDeviceVector{Int32},
    tol::Float64,
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        if i == 1
            color_start[i] = Int32(1)
        else
            cur = Int(perm[i])
            prev = Int(perm[i - 1])
            color_start[i] = (color[cur][1] != color[prev][1] ||
                              color[cur][2] > color[prev][2] + tol) ? Int32(1) : Int32(0)
        end
    end
    return nothing
end

function _write_color_id!(
    color_id::CuDeviceVector{Int32},
    perm::CuDeviceVector{Int32},
    prefix::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        color_id[Int(perm[i])] = prefix[i]
    end
    return nothing
end

# Assign contiguous color ids by sorting each item by its current color id and
# scalar feature. `color_start` marks the first item of each sorted group.
function _update_color_id!(
    color_id::CuVector{Int32},
    color_start::CuVector{Int32},
    perm::CuVector{Int32},
    color::CuVector{Tuple{Int32,Float64}},
    color_sig::CuVector{Float64},
    tol::Float64,
)
    n = length(color_sig)
    n == 0 && return color_id, 0

    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _pack_color(
        color,
        color_id,
        color_sig,
        n,
    )
    sortperm!(perm, color; initialized=false)
    @cuda threads=threads blocks=blocks _mark_color_start(
        color_start,
        color,
        perm,
        tol,
        n,
    )
    prefix = accumulate(+, color_start)
    new_color_num = Int(Array(view(prefix, n:n))[1])
    @cuda threads=threads blocks=blocks _write_color_id!(color_id, perm, prefix, n)
    return color_id, new_color_num
end

function _set_color_sig!(
    color_sig::CuDeviceVector{Float64},
    color_id::CuDeviceVector{Int32},
    color_sig_value::CuDeviceVector{Float64},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        color_sig[i] = color_sig_value[Int(color_id[i])]
    end
    return nothing
end

function _write_color_sig!(
    color_sig::CuVector{Float64},
    color_id::CuVector{Int32},
    color_num::Int,
    seed::UInt64,
)
    n = length(color_sig)
    n == 0 && return nothing

    threads = 256
    blocks = cld(n, threads)
    color_sig_value = _randn_color(color_num, seed)
    @cuda threads=threads blocks=blocks _set_color_sig!(
        color_sig,
        color_id,
        color_sig_value,
        n,
    )
    return nothing
end

# One refinement step: consume `color_sig` as the current feature, update color
# ids, then overwrite it with a fresh random value per color for the next round.
function _refine_color!(
    color_sig::CuVector{Float64},
    color_id::CuVector{Int32},
    color_start::CuVector{Int32},
    perm::CuVector{Int32},
    color::CuVector{Tuple{Int32,Float64}},
    prev_color_num::Int,
    tol::Float64,
    seed::UInt64,
)
    _, new_color_num = _update_color_id!(
        color_id,
        color_start,
        perm,
        color,
        color_sig,
        tol,
    )
    changed = new_color_num > prev_color_num
    _write_color_sig!(
        color_sig,
        color_id,
        new_color_num,
        seed,
    )
    return new_color_num, changed
end

# Sparse reduction kernels group original nonzeros by `(row color, col color)`.
# After sorting by the packed entry key, duplicate reduced entries are summed.
function _emit_triplet!(
    entry_key::CuDeviceVector{Int64},
    entry_value::CuDeviceVector{Float64},
    row_ptr::CuDeviceVector{Int32},
    col_idx::CuDeviceVector{Int32},
    nz_value::CuDeviceVector{Float64},
    row_color_id::CuDeviceVector{Int32},
    col_color_id::CuDeviceVector{Int32},
    row_color_count::CuDeviceVector{Int32},
    num_row::Int,
    num_row_color::Int,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= num_row
        row_id = Int(row_color_id[row])
        row_weight = 1.0 / Float64(row_color_count[row_id])
        start_idx = Int(row_ptr[row])
        end_idx = Int(row_ptr[row + 1]) - 1
        @inbounds for idx in start_idx:end_idx
            col_id = Int(col_color_id[col_idx[idx]])
            entry_key[idx] = Int64((col_id - 1) * num_row_color + row_id)
            entry_value[idx] = nz_value[idx] * row_weight
        end
    end
    return nothing
end

function _mark_entry_group!(
    entry_start::CuDeviceVector{Int32},
    entry_key::CuDeviceVector{Int64},
    perm::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        if i == 1
            entry_start[i] = Int32(1)
        else
            cur = Int(perm[i])
            prev = Int(perm[i - 1])
            entry_start[i] = entry_key[cur] != entry_key[prev] ? Int32(1) : Int32(0)
        end
    end
    return nothing
end

function _sum_entry!(
    entry_sum::CuDeviceVector{Float64},
    entry_value::CuDeviceVector{Float64},
    perm::CuDeviceVector{Int32},
    prefix::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        gid = Int(prefix[i])
        CUDA.@atomic entry_sum[gid] += entry_value[Int(perm[i])]
    end
    return nothing
end

function _mark_nonzero!(
    keep::CuDeviceVector{Int32},
    entry_sum::CuDeviceVector{Float64},
    tol::Float64,
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        keep[i] = abs(entry_sum[i]) > tol ? Int32(1) : Int32(0)
    end
    return nothing
end

function _write_triplet!(
    I_red::CuDeviceVector{Int32},
    J_red::CuDeviceVector{Int32},
    V_red::CuDeviceVector{Float64},
    entry_key::CuDeviceVector{Int64},
    entry_sum::CuDeviceVector{Float64},
    keep::CuDeviceVector{Int32},
    keep_prefix::CuDeviceVector{Int32},
    perm::CuDeviceVector{Int32},
    prefix::CuDeviceVector{Int32},
    num_row_color::Int,
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && prefix[i] > 0 && (i == 1 || prefix[i] != prefix[i - 1])
        gid = Int(prefix[i])
        if keep[gid] != Int32(0)
            out = Int(keep_prefix[gid])
            key_value = entry_key[Int(perm[i])]
            row_color_id = Int32(mod(key_value - 1, num_row_color) + 1)
            col_color_id = Int32(div(key_value - 1, num_row_color) + 1)
            I_red[out] = row_color_id
            J_red[out] = col_color_id
            V_red[out] = entry_sum[gid]
        end
    end
    return nothing
end

# Color counts are also the unfolding weights: dual/slack values are distributed
# back to equivalent original rows/columns by the inverse color size.
function _count_color!(
    color_count::CuDeviceVector{Int32},
    color_id::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        CUDA.@atomic color_count[Int(color_id[i])] += Int32(1)
    end
    return nothing
end

function _color_weight!(
    weight::CuDeviceVector{Float64},
    color_id::CuDeviceVector{Int32},
    color_count::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        color = Int(color_id[i])
        weight[i] = 1.0 / Float64(color_count[color])
    end
    return nothing
end

function _sum_bound!(
    lower_sum::CuDeviceVector{Float64},
    upper_sum::CuDeviceVector{Float64},
    lower::CuDeviceVector{Float64},
    upper::CuDeviceVector{Float64},
    color_id::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        color = Int(color_id[i])
        CUDA.@atomic lower_sum[color] += lower[i]
        CUDA.@atomic upper_sum[color] += upper[i]
    end
    return nothing
end

function _sum_cost!(
    cost_sum::CuDeviceVector{Float64},
    cost::CuDeviceVector{Float64},
    color_id::CuDeviceVector{Int32},
    n::Int,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= n
        color = Int(color_id[j])
        CUDA.@atomic cost_sum[color] += cost[j]
    end
    return nothing
end

function _scale_by_count!(
    out::CuDeviceVector{Float64},
    sum_value::CuDeviceVector{Float64},
    color_count::CuDeviceVector{Int32},
    n::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        out[i] = sum_value[i] / Float64(color_count[i])
    end
    return nothing
end
