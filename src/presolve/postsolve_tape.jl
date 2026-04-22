"""
Typed postsolve metadata for replaying presolve transformations.
"""

@enum PostsolveReductionType begin
    FIXED_COL
    FIXED_COL_INF
    SUB_COL
    PARALLEL_COL
    PARALLEL_ROW
    DELETED_ROW
    ADDED_ROW
    ADDED_ROWS
    LHS_CHANGE
    RHS_CHANGE
    EQ_TO_INEQ
    BOUND_CHANGE_NO_ROW
    BOUND_CHANGE_THE_ROW
end

@enum PostsolveDualMode::UInt8 begin
    POSTSOLVE_DUAL_NONE = 0
    POSTSOLVE_DUAL_EXACT = 1
    POSTSOLVE_DUAL_MINIMAL = 2
end

mutable struct PostsolveTape
    types::Vector{PostsolveReductionType}
    index_starts::Vector{Int}
    value_starts::Vector{Int}
    dual_modes::Vector{PostsolveDualMode}
    indices::Vector{Int32}
    vals::Vector{Float64}
end

function PostsolveTape()
    return PostsolveTape(
        PostsolveReductionType[],
        Int[1],
        Int[1],
        PostsolveDualMode[],
        Int32[],
        Float64[],
    )
end

mutable struct PostsolveTape_gpu
    types::CuVector{Int32}
    index_starts::CuVector{Int32}
    value_starts::CuVector{Int32}
    dual_modes::CuVector{UInt8}
    indices::CuVector{Int32}
    vals::CuVector{Float64}
end

function PostsolveTape_gpu()
    return PostsolveTape_gpu(
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{UInt8}(undef, 0),
        CuVector{Int32}(undef, 0),
        CuVector{Float64}(undef, 0),
    )
end

function PostsolveTape_gpu(tape::PostsolveTape)
    return PostsolveTape_gpu(
        CuVector(Int32.(tape.types)),
        CuVector(Int32.(tape.index_starts)),
        CuVector(Int32.(tape.value_starts)),
        CuVector(UInt8.(tape.dual_modes)),
        CuVector(tape.indices),
        CuVector(tape.vals),
    )
end

function PostsolveTape(tape::PostsolveTape_gpu)
    type_codes = Array(tape.types)
    dual_codes = Array(tape.dual_modes)
    return PostsolveTape(
        PostsolveReductionType.(type_codes),
        Int.(Array(tape.index_starts)),
        Int.(Array(tape.value_starts)),
        PostsolveDualMode.(dual_codes),
        Array(tape.indices),
        Array(tape.vals),
    )
end

function Base.copy(tape::PostsolveTape)
    return PostsolveTape(
        copy(tape.types),
        copy(tape.index_starts),
        copy(tape.value_starts),
        copy(tape.dual_modes),
        copy(tape.indices),
        copy(tape.vals),
    )
end

function Base.copy(tape::PostsolveTape_gpu)
    return PostsolveTape_gpu(
        copy(tape.types),
        copy(tape.index_starts),
        copy(tape.value_starts),
        copy(tape.dual_modes),
        copy(tape.indices),
        copy(tape.vals),
    )
end

@inline postsolve_record_count(tape::PostsolveTape) = length(tape.types)
@inline postsolve_record_count(tape::PostsolveTape_gpu) = length(tape.types)

function postsolve_record_indices(tape::PostsolveTape, k::Integer)
    1 <= k <= postsolve_record_count(tape) || throw(BoundsError(tape.types, k))
    lo = tape.index_starts[k]
    hi = tape.index_starts[k + 1] - 1
    return hi < lo ? Int32[] : tape.indices[lo:hi]
end

function postsolve_record_values(tape::PostsolveTape, k::Integer)
    1 <= k <= postsolve_record_count(tape) || throw(BoundsError(tape.types, k))
    lo = tape.value_starts[k]
    hi = tape.value_starts[k + 1] - 1
    return hi < lo ? Float64[] : tape.vals[lo:hi]
end

function postsolve_record_dual_mode(tape::PostsolveTape, k::Integer)
    1 <= k <= postsolve_record_count(tape) || throw(BoundsError(tape.dual_modes, k))
    return tape.dual_modes[k]
end

function append_postsolve_record!(
    tape::PostsolveTape,
    reduction_type::PostsolveReductionType;
    indices::AbstractVector{<:Integer}=Int32[],
    vals::AbstractVector{<:Real}=Float64[],
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_NONE,
)
    push!(tape.types, reduction_type)
    append!(tape.indices, Int32.(indices))
    append!(tape.vals, Float64.(vals))
    push!(tape.index_starts, length(tape.indices) + 1)
    push!(tape.value_starts, length(tape.vals) + 1)
    push!(tape.dual_modes, dual_mode)
    return tape
end

function append_postsolve_tape!(dest::PostsolveTape, src::PostsolveTape)
    for k in 1:postsolve_record_count(src)
        append_postsolve_record!(
            dest,
            src.types[k];
            indices=postsolve_record_indices(src, k),
            vals=postsolve_record_values(src, k),
            dual_mode=postsolve_record_dual_mode(src, k),
        )
    end
    return dest
end

function append_postsolve_tape!(dest::PostsolveTape_gpu, src::PostsolveTape_gpu)
    src_count = postsolve_record_count(src)
    src_count == 0 && return dest

    if postsolve_record_count(dest) == 0
        dest.types = copy(src.types)
        dest.index_starts = copy(src.index_starts)
        dest.value_starts = copy(src.value_starts)
        dest.dual_modes = copy(src.dual_modes)
        dest.indices = copy(src.indices)
        dest.vals = copy(src.vals)
        return dest
    end

    idx_offset = Int32(length(dest.indices))
    val_offset = Int32(length(dest.vals))
    dest.types = concat_cuvector(dest.types, src.types)
    dest.dual_modes = concat_cuvector(dest.dual_modes, src.dual_modes)
    dest.indices = concat_cuvector(dest.indices, src.indices)
    dest.vals = concat_cuvector(dest.vals, src.vals)
    dest.index_starts = concat_cuvector(dest.index_starts[1:(end - 1)], src.index_starts .+ idx_offset)
    dest.value_starts = concat_cuvector(dest.value_starts[1:(end - 1)], src.value_starts .+ val_offset)
    return dest
end

function append_bound_change_no_row_record!(
    tape::PostsolveTape,
    col::Integer,
    old_l::Real,
    old_u::Real,
    new_l::Real,
    new_u::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        BOUND_CHANGE_NO_ROW;
        indices=Int32[col],
        vals=Float64[old_l, old_u, new_l, new_u],
        dual_mode=dual_mode,
    )
end

function append_bound_change_the_row_record!(
    tape::PostsolveTape,
    col::Integer,
    row::Integer,
    old_l::Real,
    old_u::Real,
    new_l::Real,
    new_u::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        BOUND_CHANGE_THE_ROW;
        indices=Int32[col, row],
        vals=Float64[old_l, old_u, new_l, new_u],
        dual_mode=dual_mode,
    )
end

function append_fixed_col_record!(
    tape::PostsolveTape,
    col::Integer,
    fixed_val::Real,
    col_obj::Real,
    support_rows::AbstractVector{<:Integer},
    support_coeffs::AbstractVector{<:Real};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    length(support_rows) == length(support_coeffs) || error("FIXED_COL payload mismatch.")
    idx = Int32[col]
    append!(idx, Int32.(support_rows))
    payload = Float64[fixed_val, col_obj]
    append!(payload, Float64.(support_coeffs))
    return append_postsolve_record!(
        tape,
        FIXED_COL;
        indices=idx,
        vals=payload,
        dual_mode=dual_mode,
    )
end

function _kernel_fixed_col_support_counts!(
    support_counts,
    cols,
    row_ptr,
    row_idx,
    keep_row,
    use_keep_row_mask,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        col = cols[t]
        start_ptr = row_ptr[col]
        stop_ptr = row_ptr[col + 1] - Int32(1)
        count = Int32(0)
        for p in start_ptr:stop_ptr
            row = row_idx[p]
            if use_keep_row_mask == UInt8(0) || keep_row[row] != UInt8(0)
                count += Int32(1)
            end
        end
        @inbounds support_counts[t] = count
    end
    return
end

function _kernel_pack_fixed_col_records!(
    indices,
    vals,
    index_starts,
    value_starts,
    cols,
    fixed_vals,
    col_obj,
    row_ptr,
    row_idx,
    nz_val,
    keep_row,
    use_keep_row_mask,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = Int(index_starts[t])
        val0 = Int(value_starts[t])
        col = cols[t]

        @inbounds indices[idx0] = col
        @inbounds vals[val0] = fixed_vals[t]
        @inbounds vals[val0 + 1] = col_obj[t]

        write_pos = 0
        start_ptr = row_ptr[col]
        stop_ptr = row_ptr[col + 1] - Int32(1)
        for p in start_ptr:stop_ptr
            row = row_idx[p]
            if use_keep_row_mask == UInt8(0) || keep_row[row] != UInt8(0)
                write_pos += 1
                @inbounds indices[idx0 + write_pos] = row
                @inbounds vals[val0 + 1 + write_pos] = nz_val[p]
            end
        end
    end
    return
end

function build_fixed_col_records_gpu(
    cols::CuVector{Int32},
    fixed_vals::CuVector{Float64},
    col_obj::CuVector{Float64},
    row_ptr::CuVector{Int32},
    row_idx::CuVector{Int32},
    nz_val::CuVector{Float64},
    keep_row::CuVector{UInt8};
    use_keep_row_mask::Bool=true,
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    fixed_count = length(cols)
    fixed_count == 0 && return PostsolveTape_gpu()

    support_counts = CUDA.zeros(Int32, fixed_count)
    blocks = cld(fixed_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_fixed_col_support_counts!(
        support_counts,
        cols,
        row_ptr,
        row_idx,
        keep_row,
        use_keep_row_mask ? UInt8(1) : UInt8(0),
        Int32(fixed_count),
    )

    idx_widths = support_counts .+ Int32(1)
    val_widths = support_counts .+ Int32(2)
    index_starts = _build_record_starts_gpu(idx_widths)
    value_starts = _build_record_starts_gpu(val_widths)
    total_idx = Int(_copy_scalar_to_host(index_starts, fixed_count + 1)) - 1
    total_val = Int(_copy_scalar_to_host(value_starts, fixed_count + 1)) - 1

    types = CUDA.fill(Int32(FIXED_COL), fixed_count)
    dual_modes = CUDA.fill(UInt8(dual_mode), fixed_count)
    indices = CuVector{Int32}(undef, total_idx)
    vals = CuVector{Float64}(undef, total_val)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_fixed_col_records!(
        indices,
        vals,
        index_starts,
        value_starts,
        cols,
        fixed_vals,
        col_obj,
        row_ptr,
        row_idx,
        nz_val,
        keep_row,
        use_keep_row_mask ? UInt8(1) : UInt8(0),
        Int32(fixed_count),
    )

    return PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals)
end

function _kernel_fixed_col_inf_record_widths!(
    record_widths,
    active_row_counts,
    cols,
    col_to_row_ptr,
    col_to_row_idx,
    row_ptr,
    keep_row,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        col = cols[t]
        width = Int32(2)
        active_rows = Int32(0)

        start_ptr = col_to_row_ptr[col]
        stop_ptr = col_to_row_ptr[col + 1] - Int32(1)
        if start_ptr <= stop_ptr
            for p in start_ptr:stop_ptr
                row = col_to_row_idx[p]
                if keep_row[row] != UInt8(0)
                    active_rows += Int32(1)
                    width += Int32(1) + (row_ptr[row + 1] - row_ptr[row])
                end
            end
        end

        @inbounds record_widths[t] = width
        @inbounds active_row_counts[t] = active_rows
    end
    return
end

function _kernel_pack_fixed_col_inf_records!(
    indices,
    vals,
    index_starts,
    value_starts,
    active_row_counts,
    cols,
    fixed_vals,
    row_ptr,
    row_idx,
    nz_val,
    col_to_row_ptr,
    col_to_row_idx,
    keep_row,
    l,
    u,
    AL,
    AU,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = Int(index_starts[t])
        val0 = Int(value_starts[t])
        col = cols[t]
        fixed_at = fixed_vals[t]
        fix_to_pos_inf = isinf(fixed_at) && fixed_at > 0.0

        @inbounds indices[idx0] = fix_to_pos_inf ? Int32(1) : Int32(-1)
        @inbounds indices[idx0 + 1] = col
        @inbounds vals[val0] = Float64(active_row_counts[t])
        @inbounds vals[val0 + 1] = fix_to_pos_inf ? l[col] : u[col]

        write_idx = idx0 + 2
        write_val = val0 + 2
        start_ptr = col_to_row_ptr[col]
        stop_ptr = col_to_row_ptr[col + 1] - Int32(1)
        if start_ptr <= stop_ptr
            for p in start_ptr:stop_ptr
                row = col_to_row_idx[p]
                if keep_row[row] == UInt8(0)
                    continue
                end

                row_start = row_ptr[row]
                row_stop = row_ptr[row + 1] - Int32(1)
                row_len = row_stop - row_start + Int32(1)

                @inbounds indices[write_idx] = row_len
                @inbounds vals[write_val] = isfinite(AL[row]) ? AL[row] : AU[row]

                for q in row_start:row_stop
                    offset = Int(q - row_start)
                    @inbounds indices[write_idx + 1 + offset] = row_idx[q]
                    @inbounds vals[write_val + 1 + offset] = nz_val[q]
                end

                write_idx += Int(row_len) + 1
                write_val += Int(row_len) + 1
            end
        end
    end
    return
end

function build_fixed_col_inf_records_gpu(
    cols::CuVector{Int32},
    fixed_vals::CuVector{Float64},
    row_ptr::CuVector{Int32},
    row_idx::CuVector{Int32},
    nz_val::CuVector{Float64},
    col_to_row_ptr::CuVector{Int32},
    col_to_row_idx::CuVector{Int32},
    keep_row::CuVector{UInt8},
    l::CuVector{Float64},
    u::CuVector{Float64},
    AL::CuVector{Float64},
    AU::CuVector{Float64};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_EXACT,
)
    fixed_count = length(cols)
    fixed_count == 0 && return PostsolveTape_gpu()

    record_widths = CUDA.zeros(Int32, fixed_count)
    active_row_counts = CUDA.zeros(Int32, fixed_count)
    blocks = cld(fixed_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_fixed_col_inf_record_widths!(
        record_widths,
        active_row_counts,
        cols,
        col_to_row_ptr,
        col_to_row_idx,
        row_ptr,
        keep_row,
        Int32(fixed_count),
    )

    index_starts = _build_record_starts_gpu(record_widths)
    value_starts = _build_record_starts_gpu(record_widths)
    total_idx = Int(_copy_scalar_to_host(index_starts, fixed_count + 1)) - 1
    total_val = Int(_copy_scalar_to_host(value_starts, fixed_count + 1)) - 1

    types = CUDA.fill(Int32(FIXED_COL_INF), fixed_count)
    dual_modes = CUDA.fill(UInt8(dual_mode), fixed_count)
    indices = CuVector{Int32}(undef, total_idx)
    vals = CuVector{Float64}(undef, total_val)

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_fixed_col_inf_records!(
        indices,
        vals,
        index_starts,
        value_starts,
        active_row_counts,
        cols,
        fixed_vals,
        row_ptr,
        row_idx,
        nz_val,
        col_to_row_ptr,
        col_to_row_idx,
        keep_row,
        l,
        u,
        AL,
        AU,
        Int32(fixed_count),
    )

    return PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals)
end

function append_fixed_col_records_gpu!(
    tape::PostsolveTape_gpu,
    cols::CuVector{Int32},
    fixed_vals::CuVector{Float64},
    col_obj::CuVector{Float64},
    row_ptr::CuVector{Int32},
    row_idx::CuVector{Int32},
    nz_val::CuVector{Float64},
    keep_row::CuVector{UInt8};
    use_keep_row_mask::Bool=true,
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    append_postsolve_tape!(
        tape,
        build_fixed_col_records_gpu(
            cols,
            fixed_vals,
            col_obj,
            row_ptr,
            row_idx,
            nz_val,
            keep_row;
            use_keep_row_mask=use_keep_row_mask,
            dual_mode=dual_mode,
        ),
    )
    return tape
end

function _kernel_build_bound_change_slots!(
    slot_active,
    slot_has_row,
    slot_col,
    slot_row,
    slot_old_l,
    slot_old_u,
    slot_new_l,
    slot_new_u,
    cols,
    old_l,
    old_u,
    new_l,
    new_u,
    lower_changed,
    upper_changed,
    support_l_row,
    support_u_row,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        base = 2 * (t - 1)
        col = @inbounds cols[t]
        cur_l = @inbounds old_l[t]
        cur_u = @inbounds old_u[t]

        lower_slot = base + 1
        if @inbounds lower_changed[t] != UInt8(0)
            row = @inbounds support_l_row[t]
            @inbounds slot_active[lower_slot] = UInt8(1)
            @inbounds slot_has_row[lower_slot] = row != typemax(Int32) ? UInt8(1) : UInt8(0)
            @inbounds slot_col[lower_slot] = col
            @inbounds slot_row[lower_slot] = row
            @inbounds slot_old_l[lower_slot] = cur_l
            @inbounds slot_old_u[lower_slot] = cur_u
            @inbounds slot_new_l[lower_slot] = new_l[t]
            @inbounds slot_new_u[lower_slot] = cur_u
            cur_l = @inbounds new_l[t]
        end

        upper_slot = base + 2
        if @inbounds upper_changed[t] != UInt8(0)
            row = @inbounds support_u_row[t]
            @inbounds slot_active[upper_slot] = UInt8(1)
            @inbounds slot_has_row[upper_slot] = row != typemax(Int32) ? UInt8(1) : UInt8(0)
            @inbounds slot_col[upper_slot] = col
            @inbounds slot_row[upper_slot] = row
            @inbounds slot_old_l[upper_slot] = cur_l
            @inbounds slot_old_u[upper_slot] = cur_u
            @inbounds slot_new_l[upper_slot] = cur_l
            @inbounds slot_new_u[upper_slot] = new_u[t]
        end
    end
    return
end

function _kernel_pack_bound_change_records!(
    indices,
    vals,
    index_starts,
    cols,
    rows,
    has_row,
    old_l,
    old_u,
    new_l,
    new_u,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = Int(@inbounds index_starts[t])
        @inbounds indices[idx0] = cols[t]
        if @inbounds has_row[t] != UInt8(0)
            @inbounds indices[idx0 + 1] = rows[t]
        end

        val0 = 4 * (t - 1) + 1
        @inbounds vals[val0] = old_l[t]
        @inbounds vals[val0 + 1] = old_u[t]
        @inbounds vals[val0 + 2] = new_l[t]
        @inbounds vals[val0 + 3] = new_u[t]
    end
    return
end

function append_bound_change_records_gpu!(
    tape::PostsolveTape_gpu,
    cols::CuVector{Int32},
    old_l::CuVector{Float64},
    old_u::CuVector{Float64},
    new_l::CuVector{Float64},
    new_u::CuVector{Float64},
    lower_changed::CuVector{UInt8},
    upper_changed::CuVector{UInt8},
    support_l_row::CuVector{Int32},
    support_u_row::CuVector{Int32};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    changed_count = length(cols)
    changed_count == 0 && return tape

    slot_count = 2 * changed_count
    slot_active = CUDA.zeros(UInt8, slot_count)
    slot_has_row = CUDA.zeros(UInt8, slot_count)
    slot_col = CuVector{Int32}(undef, slot_count)
    slot_row = CuVector{Int32}(undef, slot_count)
    slot_old_l = CuVector{Float64}(undef, slot_count)
    slot_old_u = CuVector{Float64}(undef, slot_count)
    slot_new_l = CuVector{Float64}(undef, slot_count)
    slot_new_u = CuVector{Float64}(undef, slot_count)

    blocks = cld(changed_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_build_bound_change_slots!(
        slot_active,
        slot_has_row,
        slot_col,
        slot_row,
        slot_old_l,
        slot_old_u,
        slot_new_l,
        slot_new_u,
        cols,
        old_l,
        old_u,
        new_l,
        new_u,
        lower_changed,
        upper_changed,
        support_l_row,
        support_u_row,
        Int32(changed_count),
    )

    _, active_slots, active_count = build_maps_from_mask(slot_active)
    Int(active_count) == 0 && return tape

    has_row_sel = gather_by_red2org(slot_has_row, active_slots)
    cols_sel = gather_by_red2org(slot_col, active_slots)
    rows_sel = gather_by_red2org(slot_row, active_slots)
    old_l_sel = gather_by_red2org(slot_old_l, active_slots)
    old_u_sel = gather_by_red2org(slot_old_u, active_slots)
    new_l_sel = gather_by_red2org(slot_new_l, active_slots)
    new_u_sel = gather_by_red2org(slot_new_u, active_slots)

    idx_widths = Int32.(1 .+ (has_row_sel .!= UInt8(0)))
    index_starts = _build_record_starts_gpu(idx_widths)
    value_starts = _build_constant_record_starts_gpu(Int(active_count), Int32(4))
    total_idx = Int(_copy_scalar_to_host(index_starts, Int(active_count) + 1)) - 1
    types = ifelse.(has_row_sel .!= UInt8(0), Int32(BOUND_CHANGE_THE_ROW), Int32(BOUND_CHANGE_NO_ROW))
    dual_modes = CUDA.fill(UInt8(dual_mode), Int(active_count))
    indices = CuVector{Int32}(undef, total_idx)
    vals = CuVector{Float64}(undef, 4 * Int(active_count))

    pack_blocks = cld(Int(active_count), GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=pack_blocks _kernel_pack_bound_change_records!(
        indices,
        vals,
        index_starts,
        cols_sel,
        rows_sel,
        has_row_sel,
        old_l_sel,
        old_u_sel,
        new_l_sel,
        new_u_sel,
        Int32(active_count),
    )

    append_postsolve_tape!(
        tape,
        PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals),
    )
    return tape
end

function append_lhs_change_record!(
    tape::PostsolveTape,
    row::Integer,
    old_AL::Real,
    old_AU::Real,
    new_AL::Real,
    new_AU::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        LHS_CHANGE;
        indices=Int32[row],
        vals=Float64[old_AL, old_AU, new_AL, new_AU],
        dual_mode=dual_mode,
    )
end

function append_rhs_change_record!(
    tape::PostsolveTape,
    row::Integer,
    old_AL::Real,
    old_AU::Real,
    new_AL::Real,
    new_AU::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        RHS_CHANGE;
        indices=Int32[row],
        vals=Float64[old_AL, old_AU, new_AL, new_AU],
        dual_mode=dual_mode,
    )
end

function append_deleted_row_record!(
    tape::PostsolveTape,
    row::Integer,
    old_AL::Real,
    old_AU::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        DELETED_ROW;
        indices=Int32[row],
        vals=Float64[old_AL, old_AU],
        dual_mode=dual_mode,
    )
end

function append_deleted_singleton_row_record!(
    tape::PostsolveTape,
    row::Integer,
    col::Integer,
    coeff::Real,
    old_AL::Real,
    old_AU::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        DELETED_ROW;
        indices=Int32[row, col],
        vals=Float64[old_AL, old_AU, coeff],
        dual_mode=dual_mode,
    )
end

function _kernel_pack_deleted_singleton_row_records!(
    indices,
    vals,
    rows,
    cols,
    coeffs,
    old_AL,
    old_AU,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = 2 * (t - 1) + 1
        val0 = 3 * (t - 1) + 1
        @inbounds indices[idx0] = rows[t]
        @inbounds indices[idx0 + 1] = cols[t]
        @inbounds vals[val0] = old_AL[t]
        @inbounds vals[val0 + 1] = old_AU[t]
        @inbounds vals[val0 + 2] = coeffs[t]
    end
    return
end

function append_deleted_singleton_row_records_gpu!(
    tape::PostsolveTape_gpu,
    rows::CuVector{Int32},
    cols::CuVector{Int32},
    coeffs::CuVector{Float64},
    old_AL::CuVector{Float64},
    old_AU::CuVector{Float64};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    k = length(rows)
    k == 0 && return tape

    types = CUDA.fill(Int32(DELETED_ROW), k)
    index_starts = _build_constant_record_starts_gpu(k, Int32(2))
    value_starts = _build_constant_record_starts_gpu(k, Int32(3))
    dual_modes = CUDA.fill(UInt8(dual_mode), k)
    indices = CuVector{Int32}(undef, 2 * k)
    vals = CuVector{Float64}(undef, 3 * k)

    blocks = cld(k, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_deleted_singleton_row_records!(
        indices,
        vals,
        rows,
        cols,
        coeffs,
        old_AL,
        old_AU,
        Int32(k),
    )

    append_postsolve_tape!(
        tape,
        PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals),
    )
    return tape
end

function append_sub_col_record!(
    tape::PostsolveTape,
    elim_col::Integer,
    row::Integer,
    support_cols::AbstractVector{<:Integer},
    pivot_coeff::Real,
    rhs::Real,
    old_l::Real,
    old_u::Real,
    elim_obj::Real,
    row_deleted::Bool,
    support_coeffs::AbstractVector{<:Real};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    length(support_cols) == length(support_coeffs) || error("SUB_COL payload mismatch.")
    idx = Int32[elim_col, row, length(support_cols)]
    append!(idx, Int32.(support_cols))
    payload = Float64[pivot_coeff, rhs, old_l, old_u, elim_obj, row_deleted ? 1.0 : 0.0]
    append!(payload, Float64.(support_coeffs))
    return append_postsolve_record!(
        tape,
        SUB_COL;
        indices=idx,
        vals=payload,
        dual_mode=dual_mode,
    )
end

function append_parallel_col_record!(
    tape::PostsolveTape,
    from_col::Integer,
    to_col::Integer,
    ratio::Real,
    from_l::Real,
    from_u::Real,
    to_l::Real,
    to_u::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        PARALLEL_COL;
        indices=Int32[from_col, to_col],
        vals=Float64[ratio, from_l, from_u, to_l, to_u],
        dual_mode=dual_mode,
    )
end

function append_parallel_col_records!(
    tape::PostsolveTape,
    merged_from::AbstractVector{<:Integer},
    merged_to::AbstractVector{<:Integer},
    merged_ratio::AbstractVector{<:Real},
    merged_from_l::AbstractVector{<:Real},
    merged_from_u::AbstractVector{<:Real},
    merged_to_l::AbstractVector{<:Real},
    merged_to_u::AbstractVector{<:Real};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    merge_count = length(merged_from)
    merge_count == 0 && return tape

    merged_from_h = _copy_vector_to_host(merged_from)
    merged_to_h = _copy_vector_to_host(merged_to)
    merged_ratio_h = _copy_vector_to_host(merged_ratio)
    merged_from_l_h = _copy_vector_to_host(merged_from_l)
    merged_from_u_h = _copy_vector_to_host(merged_from_u)
    merged_to_l_h = _copy_vector_to_host(merged_to_l)
    merged_to_u_h = _copy_vector_to_host(merged_to_u)

    for t in eachindex(merged_from_h)
        append_parallel_col_record!(
            tape,
            merged_from_h[t],
            merged_to_h[t],
            merged_ratio_h[t],
            merged_from_l_h[t],
            merged_from_u_h[t],
            merged_to_l_h[t],
            merged_to_u_h[t];
            dual_mode=dual_mode,
        )
    end

    return tape
end

function _build_record_starts_gpu(widths::CuVector{Int32})
    k = length(widths)
    k == 0 && return CuVector{Int32}(undef, 0)
    starts = CUDA.fill(Int32(1), k + 1)
    starts[2:end] .= cumsum(widths) .+ Int32(1)
    return starts
end

function _build_constant_record_starts_gpu(k::Integer, width::Int32)
    k <= 0 && return CuVector{Int32}(undef, 0)
    return _build_record_starts_gpu(CUDA.fill(width, Int(k)))
end

function _kernel_pack_parallel_col_records!(
    indices,
    vals,
    merged_from,
    merged_to,
    merged_ratio,
    merged_from_l,
    merged_from_u,
    merged_to_l,
    merged_to_u,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = 2 * (t - 1) + 1
        val0 = 5 * (t - 1) + 1

        @inbounds indices[idx0] = merged_from[t]
        @inbounds indices[idx0 + 1] = merged_to[t]

        @inbounds vals[val0] = merged_ratio[t]
        @inbounds vals[val0 + 1] = merged_from_l[t]
        @inbounds vals[val0 + 2] = merged_from_u[t]
        @inbounds vals[val0 + 3] = merged_to_l[t]
        @inbounds vals[val0 + 4] = merged_to_u[t]
    end
    return
end

function append_parallel_col_records_gpu!(
    tape::PostsolveTape_gpu,
    merged_from::CuVector{Int32},
    merged_to::CuVector{Int32},
    merged_ratio::CuVector{Float64},
    merged_from_l::CuVector{Float64},
    merged_from_u::CuVector{Float64},
    merged_to_l::CuVector{Float64},
    merged_to_u::CuVector{Float64};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    merge_count = length(merged_from)
    merge_count == 0 && return tape

    types = CUDA.fill(Int32(PARALLEL_COL), merge_count)
    index_starts = _build_constant_record_starts_gpu(merge_count, Int32(2))
    value_starts = _build_constant_record_starts_gpu(merge_count, Int32(5))
    dual_modes = CUDA.fill(UInt8(dual_mode), merge_count)
    indices = CuVector{Int32}(undef, 2 * merge_count)
    vals = CuVector{Float64}(undef, 5 * merge_count)

    blocks = cld(merge_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_parallel_col_records!(
        indices,
        vals,
        merged_from,
        merged_to,
        merged_ratio,
        merged_from_l,
        merged_from_u,
        merged_to_l,
        merged_to_u,
        Int32(merge_count),
    )

    append_postsolve_tape!(
        tape,
        PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals),
    )
    return tape
end

function append_parallel_row_record!(
    tape::PostsolveTape,
    kept_row::Integer,
    deleted_row::Integer,
    ratio::Real,
    kept_old_AL::Real,
    kept_old_AU::Real,
    deleted_old_AL::Real,
    deleted_old_AU::Real;
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    return append_postsolve_record!(
        tape,
        PARALLEL_ROW;
        indices=Int32[kept_row, deleted_row],
        vals=Float64[ratio, kept_old_AL, kept_old_AU, deleted_old_AL, deleted_old_AU],
        dual_mode=dual_mode,
    )
end

function _kernel_pack_parallel_row_records!(
    indices,
    vals,
    kept_rows,
    deleted_rows,
    ratios,
    kept_old_AL,
    kept_old_AU,
    deleted_old_AL,
    deleted_old_AU,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = 2 * (t - 1) + 1
        val0 = 5 * (t - 1) + 1
        @inbounds indices[idx0] = kept_rows[t]
        @inbounds indices[idx0 + 1] = deleted_rows[t]
        @inbounds vals[val0] = ratios[t]
        @inbounds vals[val0 + 1] = kept_old_AL[t]
        @inbounds vals[val0 + 2] = kept_old_AU[t]
        @inbounds vals[val0 + 3] = deleted_old_AL[t]
        @inbounds vals[val0 + 4] = deleted_old_AU[t]
    end
    return
end

function append_parallel_row_records_gpu!(
    tape::PostsolveTape_gpu,
    kept_rows::CuVector{Int32},
    deleted_rows::CuVector{Int32},
    ratios::CuVector{Float64},
    kept_old_AL::CuVector{Float64},
    kept_old_AU::CuVector{Float64},
    deleted_old_AL::CuVector{Float64},
    deleted_old_AU::CuVector{Float64};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    k = length(kept_rows)
    k == 0 && return tape

    types = CUDA.fill(Int32(PARALLEL_ROW), k)
    index_starts = _build_constant_record_starts_gpu(k, Int32(2))
    value_starts = _build_constant_record_starts_gpu(k, Int32(5))
    dual_modes = CUDA.fill(UInt8(dual_mode), k)
    indices = CuVector{Int32}(undef, 2 * k)
    vals = CuVector{Float64}(undef, 5 * k)

    blocks = cld(k, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_parallel_row_records!(
        indices,
        vals,
        kept_rows,
        deleted_rows,
        ratios,
        kept_old_AL,
        kept_old_AU,
        deleted_old_AL,
        deleted_old_AU,
        Int32(k),
    )

    append_postsolve_tape!(
        tape,
        PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals),
    )
    return tape
end

function append_parallel_row_records!(
    tape::PostsolveTape,
    kept_rows::AbstractVector{<:Integer},
    deleted_rows::AbstractVector{<:Integer},
    ratios::AbstractVector{<:Real},
    kept_old_AL::AbstractVector{<:Real},
    kept_old_AU::AbstractVector{<:Real},
    deleted_old_AL::AbstractVector{<:Real},
    deleted_old_AU::AbstractVector{<:Real};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    count = length(kept_rows)
    count == length(deleted_rows) == length(ratios) == length(kept_old_AL) ==
        length(kept_old_AU) == length(deleted_old_AL) == length(deleted_old_AU) ||
        error("PARALLEL_ROW payload mismatch.")

    kept_rows_h = _copy_vector_to_host(kept_rows)
    deleted_rows_h = _copy_vector_to_host(deleted_rows)
    ratios_h = _copy_vector_to_host(ratios)
    kept_old_AL_h = _copy_vector_to_host(kept_old_AL)
    kept_old_AU_h = _copy_vector_to_host(kept_old_AU)
    deleted_old_AL_h = _copy_vector_to_host(deleted_old_AL)
    deleted_old_AU_h = _copy_vector_to_host(deleted_old_AU)

    for t in eachindex(kept_rows_h)
        append_parallel_row_record!(
            tape,
            kept_rows_h[t],
            deleted_rows_h[t],
            ratios_h[t],
            kept_old_AL_h[t],
            kept_old_AU_h[t],
            deleted_old_AL_h[t],
            deleted_old_AU_h[t];
            dual_mode=dual_mode,
        )
    end

    return tape
end

function append_sub_col_records_from_payload!(
    tape::PostsolveTape,
    eliminated_cols::AbstractVector{<:Integer},
    pair_rows::AbstractVector{<:Integer},
    support_counts::AbstractVector{<:Integer},
    support_starts::AbstractVector{<:Integer},
    flat_support_cols::AbstractVector{<:Integer},
    flat_support_coeffs::AbstractVector{<:Real},
    pivot_coeff::AbstractVector{<:Real},
    rhs_old::AbstractVector{<:Real},
    old_l::AbstractVector{<:Real},
    old_u::AbstractVector{<:Real},
    elim_obj::AbstractVector{<:Real},
    row_deleted::AbstractVector{<:Integer};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    eliminated_count = length(eliminated_cols)
    eliminated_count == 0 && return tape

    eliminated_cols_h = _copy_vector_to_host(eliminated_cols)
    pair_rows_h = _copy_vector_to_host(pair_rows)
    support_counts_h = _copy_vector_to_host(support_counts)
    support_starts_h = _copy_vector_to_host(support_starts)
    flat_support_cols_h = _copy_vector_to_host(flat_support_cols)
    flat_support_coeffs_h = _copy_vector_to_host(flat_support_coeffs)
    pivot_coeff_h = _copy_vector_to_host(pivot_coeff)
    rhs_old_h = _copy_vector_to_host(rhs_old)
    old_l_h = _copy_vector_to_host(old_l)
    old_u_h = _copy_vector_to_host(old_u)
    elim_obj_h = _copy_vector_to_host(elim_obj)
    row_deleted_h = _copy_vector_to_host(row_deleted)

    for t in eachindex(eliminated_cols_h)
        count = Int(support_counts_h[t])
        if count == 0
            support_cols = Int32[]
            support_coeffs = Float64[]
        else
            start_idx = Int(support_starts_h[t])
            stop_idx = start_idx + count - 1
            support_cols = @view flat_support_cols_h[start_idx:stop_idx]
            support_coeffs = @view flat_support_coeffs_h[start_idx:stop_idx]
        end

        append_sub_col_record!(
            tape,
            eliminated_cols_h[t],
            pair_rows_h[t],
            support_cols,
            pivot_coeff_h[t],
            rhs_old_h[t],
            old_l_h[t],
            old_u_h[t],
            elim_obj_h[t],
            row_deleted_h[t] != 0,
            support_coeffs;
            dual_mode=dual_mode,
        )

        if row_deleted_h[t] != 0
            append_deleted_row_record!(
                tape,
                pair_rows_h[t],
                rhs_old_h[t],
                rhs_old_h[t];
                dual_mode=dual_mode,
            )
        end
    end

    return tape
end

function _kernel_pack_sub_col_records!(
    indices,
    vals,
    index_starts,
    value_starts,
    eliminated_cols,
    pair_rows,
    support_counts,
    support_starts,
    flat_support_cols,
    flat_support_coeffs,
    pivot_coeff,
    rhs_old,
    old_l,
    old_u,
    elim_obj,
    row_deleted,
    k,
)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= k
        idx0 = Int(@inbounds index_starts[t])
        val0 = Int(@inbounds value_starts[t])
        count = Int(@inbounds support_counts[t])
        src0 = Int(@inbounds support_starts[t])

        @inbounds indices[idx0] = eliminated_cols[t]
        @inbounds indices[idx0 + 1] = pair_rows[t]
        @inbounds indices[idx0 + 2] = Int32(count)

        @inbounds vals[val0] = pivot_coeff[t]
        @inbounds vals[val0 + 1] = rhs_old[t]
        @inbounds vals[val0 + 2] = old_l[t]
        @inbounds vals[val0 + 3] = old_u[t]
        @inbounds vals[val0 + 4] = elim_obj[t]
        @inbounds vals[val0 + 5] = row_deleted[t] != 0 ? 1.0 : 0.0

        for j in 0:(count - 1)
            @inbounds indices[idx0 + 3 + j] = flat_support_cols[src0 + j]
            @inbounds vals[val0 + 6 + j] = flat_support_coeffs[src0 + j]
        end
    end
    return
end

function append_sub_col_records_from_payload_gpu!(
    tape::PostsolveTape_gpu,
    eliminated_cols::CuVector{Int32},
    pair_rows::CuVector{Int32},
    support_counts::CuVector{Int32},
    support_starts::CuVector{Int32},
    flat_support_cols::CuVector{Int32},
    flat_support_coeffs::CuVector{Float64},
    pivot_coeff::CuVector{Float64},
    rhs_old::CuVector{Float64},
    old_l::CuVector{Float64},
    old_u::CuVector{Float64},
    elim_obj::CuVector{Float64},
    row_deleted::CuVector{UInt8};
    dual_mode::PostsolveDualMode=POSTSOLVE_DUAL_MINIMAL,
)
    eliminated_count = length(eliminated_cols)
    eliminated_count == 0 && return tape

    idx_widths = support_counts .+ Int32(3)
    val_widths = support_counts .+ Int32(6)
    index_starts = _build_record_starts_gpu(idx_widths)
    value_starts = _build_record_starts_gpu(val_widths)
    total_idx = Int(_copy_scalar_to_host(index_starts, eliminated_count + 1)) - 1
    total_val = Int(_copy_scalar_to_host(value_starts, eliminated_count + 1)) - 1

    types = CUDA.fill(Int32(SUB_COL), eliminated_count)
    dual_modes = CUDA.fill(UInt8(dual_mode), eliminated_count)
    indices = CuVector{Int32}(undef, total_idx)
    vals = CuVector{Float64}(undef, total_val)

    blocks = cld(eliminated_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_pack_sub_col_records!(
        indices,
        vals,
        index_starts,
        value_starts,
        eliminated_cols,
        pair_rows,
        support_counts,
        support_starts,
        flat_support_cols,
        flat_support_coeffs,
        pivot_coeff,
        rhs_old,
        old_l,
        old_u,
        elim_obj,
        row_deleted,
        Int32(eliminated_count),
    )

    append_postsolve_tape!(
        tape,
        PostsolveTape_gpu(types, index_starts, value_starts, dual_modes, indices, vals),
    )
    return tape
end

function _as_i32_host_vector(v)
    return Int32.(_copy_vector_to_host(v))
end

function globalize_postsolve_tape(
    tape::PostsolveTape,
    row_red2org,
    col_red2org,
)
    row_map = _as_i32_host_vector(row_red2org)
    col_map = _as_i32_host_vector(col_red2org)
    mapped = PostsolveTape()

    for k in 1:postsolve_record_count(tape)
        reduction_type = tape.types[k]
        idx = copy(postsolve_record_indices(tape, k))
        vals = copy(postsolve_record_values(tape, k))
        dual_mode = postsolve_record_dual_mode(tape, k)

        if reduction_type == DELETED_ROW
            idx[1] = row_map[idx[1]]
            if length(idx) >= 2
                idx[2] = col_map[idx[2]]
            end
        elseif reduction_type == LHS_CHANGE ||
               reduction_type == RHS_CHANGE ||
               reduction_type == EQ_TO_INEQ
            idx[1] = row_map[idx[1]]
        elseif reduction_type == BOUND_CHANGE_THE_ROW
            idx[1] = col_map[idx[1]]
            idx[2] = row_map[idx[2]]
        elseif reduction_type == FIXED_COL
            idx[1] = col_map[idx[1]]
            for t in 2:length(idx)
                idx[t] = row_map[idx[t]]
            end
        elseif reduction_type == PARALLEL_ROW
            idx[1] = row_map[idx[1]]
            if length(idx) >= 2
                idx[2] = row_map[idx[2]]
            end
        elseif reduction_type == BOUND_CHANGE_NO_ROW ||
               reduction_type == FIXED_COL_INF ||
               reduction_type == PARALLEL_COL
            idx[1] = col_map[idx[1]]
            if length(idx) >= 2 && reduction_type == PARALLEL_COL
                idx[2] = col_map[idx[2]]
            end
        elseif reduction_type == SUB_COL
            idx[1] = col_map[idx[1]]
            idx[2] = row_map[idx[2]]
            support_count = Int(idx[3])
            for t in 1:support_count
                idx[3 + t] = col_map[idx[3 + t]]
            end
        end

        append_postsolve_record!(
            mapped,
            reduction_type;
            indices=idx,
            vals=vals,
            dual_mode=dual_mode,
        )
    end

    return mapped
end

function _kernel_globalize_postsolve_tape_indices!(
    mapped_idx,
    types,
    idx_starts,
    row_map,
    col_map,
    record_count,
)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= record_count
        reduction_type = @inbounds types[k]
        idx0 = Int(@inbounds idx_starts[k])
        idx1 = Int(@inbounds idx_starts[k + 1]) - 1

        if reduction_type == Int32(DELETED_ROW)
            @inbounds mapped_idx[idx0] = row_map[mapped_idx[idx0]]
            if idx1 >= idx0 + 1
                @inbounds mapped_idx[idx0 + 1] = col_map[mapped_idx[idx0 + 1]]
            end
        elseif reduction_type == Int32(LHS_CHANGE) ||
               reduction_type == Int32(RHS_CHANGE) ||
               reduction_type == Int32(EQ_TO_INEQ)
            @inbounds mapped_idx[idx0] = row_map[mapped_idx[idx0]]
        elseif reduction_type == Int32(BOUND_CHANGE_THE_ROW)
            @inbounds mapped_idx[idx0] = col_map[mapped_idx[idx0]]
            @inbounds mapped_idx[idx0 + 1] = row_map[mapped_idx[idx0 + 1]]
        elseif reduction_type == Int32(FIXED_COL)
            @inbounds mapped_idx[idx0] = col_map[mapped_idx[idx0]]
            for pos in (idx0 + 1):idx1
                @inbounds mapped_idx[pos] = row_map[mapped_idx[pos]]
            end
        elseif reduction_type == Int32(PARALLEL_ROW)
            @inbounds mapped_idx[idx0] = row_map[mapped_idx[idx0]]
            if idx1 >= idx0 + 1
                @inbounds mapped_idx[idx0 + 1] = row_map[mapped_idx[idx0 + 1]]
            end
        elseif reduction_type == Int32(BOUND_CHANGE_NO_ROW) ||
               reduction_type == Int32(FIXED_COL_INF) ||
               reduction_type == Int32(PARALLEL_COL)
            @inbounds mapped_idx[idx0] = col_map[mapped_idx[idx0]]
            if reduction_type == Int32(PARALLEL_COL) && idx1 >= idx0 + 1
                @inbounds mapped_idx[idx0 + 1] = col_map[mapped_idx[idx0 + 1]]
            end
        elseif reduction_type == Int32(SUB_COL)
            @inbounds mapped_idx[idx0] = col_map[mapped_idx[idx0]]
            @inbounds mapped_idx[idx0 + 1] = row_map[mapped_idx[idx0 + 1]]
            support_count = Int(@inbounds mapped_idx[idx0 + 2])
            for t in 1:support_count
                pos = idx0 + 2 + t
                @inbounds mapped_idx[pos] = col_map[mapped_idx[pos]]
            end
        end
    end
    return
end

function globalize_postsolve_tape_gpu(
    tape::PostsolveTape_gpu,
    row_red2org::CuVector{Int32},
    col_red2org::CuVector{Int32},
)
    record_count = postsolve_record_count(tape)
    if record_count == 0
        return PostsolveTape_gpu()
    end

    mapped = PostsolveTape_gpu(
        copy(tape.types),
        copy(tape.index_starts),
        copy(tape.value_starts),
        copy(tape.dual_modes),
        copy(tape.indices),
        copy(tape.vals),
    )
    blocks = cld(record_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_globalize_postsolve_tape_indices!(
        mapped.indices,
        mapped.types,
        mapped.index_starts,
        row_red2org,
        col_red2org,
        Int32(record_count),
    )
    return mapped
end
