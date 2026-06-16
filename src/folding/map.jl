# Keep color ids and inverse color sizes together. The ids fold/unfold vector
# positions; the scales distribute reduced dual/slack values back to originals.
function _build_map(
    row_color_id::CuVector{Int32},
    col_color_id::CuVector{Int32},
    num_row_color::Int,
    num_col_color::Int,
)
    device_row_color_id = copy(row_color_id)
    device_col_color_id = copy(col_color_id)
    threads = 256
    row_color_count = CUDA.zeros(Int32, num_row_color)
    col_color_count = CUDA.zeros(Int32, num_col_color)
    @cuda threads=threads blocks=cld(length(device_row_color_id), threads) _count_color!(
        row_color_count,
        device_row_color_id,
        length(device_row_color_id),
    )
    @cuda threads=threads blocks=cld(length(device_col_color_id), threads) _count_color!(
        col_color_count,
        device_col_color_id,
        length(device_col_color_id),
    )

    row_scale = CuVector{Float64}(undef, length(device_row_color_id))
    col_scale = CuVector{Float64}(undef, length(device_col_color_id))
    @cuda threads=threads blocks=cld(length(row_scale), threads) _color_weight!(
        row_scale,
        device_row_color_id,
        row_color_count::CuVector{Int32},
        length(row_scale),
    )
    @cuda threads=threads blocks=cld(length(col_scale), threads) _color_weight!(
        col_scale,
        device_col_color_id,
        col_color_count::CuVector{Int32},
        length(col_scale),
    )

    return DeviceFoldingMap(
        length(device_row_color_id),
        length(device_col_color_id),
        device_row_color_id,
        device_col_color_id,
        row_scale,
        col_scale,
    )
end

# Solver-facing code uses a CPU map after folding is finished.
function _cpu_map(map::DeviceFoldingMap)
    return FoldingMap(
        map.original_nrow,
        map.original_ncol,
        Int.(Array(map.row_color_id)),
        Int.(Array(map.col_color_id)),
        Array(map.row_scale),
        Array(map.col_scale),
    )
end
