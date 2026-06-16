# Build the quotient LP induced by row/column colors.
function reduce_size(
    row_color_id::CuVector{Int32},
    col_color_id::CuVector{Int32},
    num_row_color::Int,
    num_col_color::Int,
    model::LP_info_gpu,
    tolerance::Float64,
)
    num_row, num_col = size(model.A)

    threads = 256
    block_row = cld(num_row, threads)
    block_col = cld(num_col, threads)

    # Build row/column aggregation operators fully on GPU.
    row_color_count = CUDA.zeros(Int32, num_row_color)
    col_color_count = CUDA.zeros(Int32, num_col_color)
    @cuda threads=threads blocks=block_row _count_color!(row_color_count, row_color_id, num_row)
    @cuda threads=threads blocks=block_col _count_color!(col_color_count, col_color_id, num_col)

    num_nz = length(model.A.nzVal)
    entry_key = CuVector{Int64}(undef, num_nz)
    entry_value = CuVector{Float64}(undef, num_nz)
    # Each original nonzero maps to one reduced matrix position; sorting by
    # entry_key lets equal color pairs be summed without leaving the GPU.
    @cuda threads=threads blocks=block_row _emit_triplet!(
        entry_key,
        entry_value,
        model.A.rowPtr,
        model.A.colVal,
        model.A.nzVal,
        row_color_id,
        col_color_id,
        row_color_count,
        num_row,
        num_row_color,
    )

    perm = CuVector{Int32}(undef, num_nz)
    sortperm!(perm, entry_key; initialized=false)
    entry_start = CUDA.zeros(Int32, num_nz)
    @cuda threads=threads blocks=cld(num_nz, threads) _mark_entry_group!(
        entry_start,
        entry_key,
        perm,
        num_nz,
    )
    prefix = accumulate(+, entry_start)
    entry_num = Int(Array(view(prefix, num_nz:num_nz))[1])
    entry_sum = CUDA.zeros(Float64, entry_num)
    @cuda threads=threads blocks=cld(num_nz, threads) _sum_entry!(
        entry_sum,
        entry_value,
        perm,
        prefix,
        num_nz,
    )
    keep = CuVector{Int32}(undef, entry_num)
    @cuda threads=threads blocks=cld(entry_num, threads) _mark_nonzero!(
        keep,
        entry_sum,
        tolerance,
        entry_num,
    )
    keep_prefix = accumulate(+, keep)
    keep_num = Int(Array(view(keep_prefix, entry_num:entry_num))[1])
    I_red = CuVector{Int32}(undef, keep_num)
    J_red = CuVector{Int32}(undef, keep_num)
    V_red = CuVector{Float64}(undef, keep_num)
    @cuda threads=threads blocks=cld(num_nz, threads) _write_triplet!(
        I_red,
        J_red,
        V_red,
        entry_key,
        entry_sum,
        keep,
        keep_prefix,
        perm,
        prefix,
        num_row_color,
        num_nz,
    )
    A_red = CuSparseMatrixCSR(CUDA.CUSPARSE.CuSparseMatrixCOO(I_red, J_red, V_red, (num_row_color, num_col_color)))
    block_row_color = cld(num_row_color, threads)

    # Objective coefficients are summed by column color.
    c_red = CUDA.zeros(Float64, num_col_color)
    @cuda threads=threads blocks=block_col _sum_cost!(
        c_red,
        model.c,
        col_color_id,
        num_col,
    )

    AL_sum = CUDA.zeros(Float64, num_row_color)
    AU_sum = CUDA.zeros(Float64, num_row_color)
    @cuda threads=threads blocks=block_row _sum_bound!(
        AL_sum,
        AU_sum,
        model.AL,
        model.AU,
        row_color_id,
        num_row,
    )

    # Row and column bounds are averaged within each color class so unfolding can
    # distribute dual/slack values by inverse color size.
    AL_red = CUDA.zeros(Float64, num_row_color)
    AU_red = CUDA.zeros(Float64, num_row_color)
    @cuda threads=threads blocks=block_row_color _scale_by_count!(
        AL_red,
        AL_sum,
        row_color_count,
        num_row_color,
    )
    @cuda threads=threads blocks=block_row_color _scale_by_count!(
        AU_red,
        AU_sum,
        row_color_count,
        num_row_color,
    )

    l_sum = CUDA.zeros(Float64, num_col_color)
    u_sum = CUDA.zeros(Float64, num_col_color)
    @cuda threads=threads blocks=block_col _sum_bound!(
        l_sum,
        u_sum,
        model.l,
        model.u,
        col_color_id,
        num_col,
    )

    block_col_color = cld(num_col_color, threads)
    l_red = CUDA.zeros(Float64, num_col_color)
    u_red = CUDA.zeros(Float64, num_col_color)
    @cuda threads=threads blocks=block_col_color _scale_by_count!(
        l_red,
        l_sum,
        col_color_count,
        num_col_color,
    )
    @cuda threads=threads blocks=block_col_color _scale_by_count!(
        u_red,
        u_sum,
        col_color_count,
        num_col_color,
    )

    AT_red = CuSparseMatrixCSR(transpose(A_red))

    return LP_info_gpu(
        A_red,
        AT_red,
        c_red,
        AL_red,
        AU_red,
        l_red,
        u_red,
        model.obj_constant,
        Int32(0),
        CUDA.zeros(Int32, size(AT_red, 1)),
    )
end
