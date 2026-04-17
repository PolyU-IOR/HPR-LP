"""
GPU postsolve for rebuilding original-dimension solution vectors.
"""

using CUDA
using CUDA: CuVector
using SparseArrays

function _kernel_scatter_fixed_values!(
    x_org,
    fixed_idx,
    fixed_val,
    k,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= k
        @inbounds idx = fixed_idx[i]
        @inbounds x_org[idx] = fixed_val[i]
    end
    return
end

"""
Scatter fixed-variable values stored in the cumulative record.
"""
function postsolve_restore_fixed_x_gpu!(
    x_org::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    k = length(rec.fixed_idx)
    if k == 0
        return x_org
    end

    fixed_val_t = rec.fixed_val
    if eltype(fixed_val_t) != T
        fixed_val_t = T.(fixed_val_t)
    end

    blocks = cld(k, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_scatter_fixed_values!(
        x_org,
        rec.fixed_idx,
        fixed_val_t,
        Int32(k),
    )
    return x_org
end

function _kernel_replay_sub_col!(
    x_org,
    tape_idx,
    tape_val,
    idx_start,
    val_start,
)
    tid = Int(threadIdx().x)
    lane_count = Int(blockDim().x)
    idx0 = Int(idx_start)
    val0 = Int(val_start)
    support_count = Int(@inbounds tape_idx[idx0 + 2])

    partial = 0.0
    j = tid
    while j <= support_count
        col = Int(@inbounds tape_idx[idx0 + 2 + j])
        coeff = @inbounds tape_val[val0 + 5 + j]
        partial += coeff * @inbounds(x_org[col])
        j += lane_count
    end

    scratch = CUDA.@cuStaticSharedMem(Float64, GPU_PRESOLVE_THREADS)
    scratch[tid] = partial
    sync_threads()

    offset = lane_count >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        elim_col = Int(@inbounds tape_idx[idx0])
        pivot_coeff = @inbounds tape_val[val0]
        rhs = @inbounds tape_val[val0 + 1]
        @inbounds x_org[elim_col] = (rhs - scratch[1]) / pivot_coeff
    end
    return
end

function _kernel_replay_sub_col_dual!(
    y_org,
    z_org,
    tape_idx,
    tape_val,
    idx_start,
    val_start,
)
    if threadIdx().x == 1
        idx0 = Int(idx_start)
        val0 = Int(val_start)

        elim_col = Int(@inbounds tape_idx[idx0])
        row = Int(@inbounds tape_idx[idx0 + 1])
        pivot_coeff = @inbounds tape_val[val0]
        elim_obj = @inbounds tape_val[val0 + 4]
        row_deleted = @inbounds tape_val[val0 + 5] != 0.0

        if row_deleted
            @inbounds y_org[row] = elim_obj / pivot_coeff
            @inbounds z_org[elim_col] = 0.0
        else
            @inbounds y_org[row] += elim_obj / pivot_coeff
            @inbounds z_org[elim_col] = elim_obj - pivot_coeff * y_org[row]
        end
    end
    return
end

function _kernel_replay_parallel_col!(
    x_org,
    tape_idx,
    tape_val,
    idx_start,
    val_start,
)
    if threadIdx().x == 1
        idx0 = Int(idx_start)
        val0 = Int(val_start)

        from_idx = Int(@inbounds tape_idx[idx0])
        to_idx = Int(@inbounds tape_idx[idx0 + 1])
        ratio = @inbounds tape_val[val0]
        lo = @inbounds tape_val[val0 + 1]
        hi = @inbounds tape_val[val0 + 2]
        to_l = @inbounds tape_val[val0 + 3]
        to_u = @inbounds tape_val[val0 + 4]

        y_val = @inbounds x_org[to_idx]

        if isfinite(to_u)
            bound = (y_val - to_u) / ratio
            if ratio > 0.0
                lo = max(lo, bound)
            else
                hi = min(hi, bound)
            end
        end
        if isfinite(to_l)
            bound = (y_val - to_l) / ratio
            if ratio > 0.0
                hi = min(hi, bound)
            else
                lo = max(lo, bound)
            end
        end

        x_from = _choose_merged_source_value(lo, hi)
        x_to = y_val - ratio * x_from

        @inbounds x_org[from_idx] = x_from
        @inbounds x_org[to_idx] = x_to
    end
    return
end

function _kernel_replay_parallel_col_dual!(
    z_org,
    tape_idx,
    tape_val,
    idx_start,
    val_start,
)
    if threadIdx().x == 1
        idx0 = Int(idx_start)
        val0 = Int(val_start)

        from_idx = Int(@inbounds tape_idx[idx0])
        to_idx = Int(@inbounds tape_idx[idx0 + 1])
        ratio = @inbounds tape_val[val0]

        @inbounds z_org[from_idx] = ratio * z_org[to_idx]
    end
    return
end

function ensure_postsolve_tape_gpu!(rec::PresolveRecord_gpu)
    if isnothing(rec.tape_gpu)
        if !isempty(rec.tape_gpu_parts)
            tape_gpu = PostsolveTape_gpu()
            for part in rec.tape_gpu_parts
                append_postsolve_tape!(tape_gpu, part)
            end
            rec.tape_gpu = tape_gpu
        else
            rec.tape_gpu = PostsolveTape_gpu(rec.tape)
        end
    end
    return rec.tape_gpu
end

function ensure_postsolve_tape_cpu!(rec::PresolveRecord_gpu)
    if postsolve_record_count(rec.tape) == 0
        tape_gpu = ensure_postsolve_tape_gpu!(rec)
        if postsolve_record_count(tape_gpu) > 0
            rec.tape = PostsolveTape(tape_gpu)
        end
    end
    return rec.tape
end

function _kernel_mark_exact_replay_columns_from_tape_gpu!(
    protected_cols,
    reduction_types,
    idx_starts,
    tape_idx,
    record_count,
    n,
)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= record_count
        reduction_type = @inbounds reduction_types[k]
        idx0 = Int(@inbounds idx_starts[k])

        if reduction_type == Int32(SUB_COL)
            col = Int(@inbounds tape_idx[idx0])
            if 1 <= col <= n
                @inbounds protected_cols[col] = UInt8(1)
            end
        elseif reduction_type == Int32(FIXED_COL_INF)
            col = Int(@inbounds tape_idx[idx0 + 1])
            if 1 <= col <= n
                @inbounds protected_cols[col] = UInt8(1)
            end
        elseif reduction_type == Int32(PARALLEL_COL)
            col = Int(@inbounds tape_idx[idx0])
            if 1 <= col <= n
                @inbounds protected_cols[col] = UInt8(1)
            end
        end
    end
    return
end

function _build_exact_replay_column_mask_gpu(
    rec::PresolveRecord_gpu,
    n::Integer,
)
    protected_cols = CUDA.zeros(UInt8, Int(n))
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return protected_cols

    blocks = cld(record_count, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_mark_exact_replay_columns_from_tape_gpu!(
        protected_cols,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.indices,
        Int32(record_count),
        Int32(length(protected_cols)),
    )

    return protected_cols
end

function _kernel_detect_global_original_dual_refinement!(
    needs_refine,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    record_count,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds needs_refine[1] = UInt8(0)
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            if reduction_type == Int32(BOUND_CHANGE_THE_ROW) ||
               reduction_type == Int32(LHS_CHANGE) ||
               reduction_type == Int32(RHS_CHANGE)
                @inbounds needs_refine[1] = UInt8(1)
                break
            elseif reduction_type == Int32(SUB_COL)
                idx0 = Int(@inbounds idx_starts[k])
                val0 = Int(@inbounds val_starts[k])
                idx1 = Int(@inbounds idx_starts[k + 1]) - 1
                val1 = Int(@inbounds val_starts[k + 1]) - 1
                idx1 >= idx0 + 3 || continue
                val1 >= val0 + 5 || continue
                support_count = Int(@inbounds tape_idx[idx0 + 2])
                row_deleted = @inbounds tape_val[val0 + 5] != 0.0
                if !row_deleted && support_count == 1
                    @inbounds needs_refine[1] = UInt8(1)
                    break
                end
            end
        end
    end
    return
end

function _kernel_replay_x_tape_all!(
    x_org,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    record_count,
)
    tid = Int(threadIdx().x)
    lane_count = Int(blockDim().x)
    scratch = CUDA.@cuStaticSharedMem(Float64, GPU_PRESOLVE_THREADS)

    for k in Int(record_count):-1:1
        reduction_type = @inbounds reduction_types[k]
        idx0 = Int(@inbounds idx_starts[k])
        val0 = Int(@inbounds val_starts[k])

        if reduction_type == Int32(SUB_COL)
            support_count = Int(@inbounds tape_idx[idx0 + 2])

            partial = 0.0
            j = tid
            while j <= support_count
                col = Int(@inbounds tape_idx[idx0 + 2 + j])
                coeff = @inbounds tape_val[val0 + 5 + j]
                partial += coeff * @inbounds(x_org[col])
                j += lane_count
            end

            scratch[tid] = partial
            sync_threads()

            offset = lane_count >>> 1
            while offset > 0
                if tid <= offset
                    scratch[tid] += scratch[tid + offset]
                end
                sync_threads()
                offset >>>= 1
            end

            if tid == 1
                elim_col = Int(@inbounds tape_idx[idx0])
                pivot_coeff = @inbounds tape_val[val0]
                rhs = @inbounds tape_val[val0 + 1]
                @inbounds x_org[elim_col] = (rhs - scratch[1]) / pivot_coeff
            end
        elseif reduction_type == Int32(PARALLEL_COL)
            if tid == 1
                from_idx = Int(@inbounds tape_idx[idx0])
                to_idx = Int(@inbounds tape_idx[idx0 + 1])
                ratio = @inbounds tape_val[val0]
                lo = @inbounds tape_val[val0 + 1]
                hi = @inbounds tape_val[val0 + 2]
                to_l = @inbounds tape_val[val0 + 3]
                to_u = @inbounds tape_val[val0 + 4]

                y_val = @inbounds x_org[to_idx]

                if isfinite(to_u)
                    bound = (y_val - to_u) / ratio
                    if ratio > 0.0
                        lo = max(lo, bound)
                    else
                        hi = min(hi, bound)
                    end
                end
                if isfinite(to_l)
                    bound = (y_val - to_l) / ratio
                    if ratio > 0.0
                        hi = min(hi, bound)
                    else
                        lo = max(lo, bound)
                    end
                end

                x_from = _choose_merged_source_value(lo, hi)
                x_to = y_val - ratio * x_from

                @inbounds x_org[from_idx] = x_from
                @inbounds x_org[to_idx] = x_to
            end
        elseif reduction_type == Int32(FIXED_COL)
            if tid == 1
                col = Int(@inbounds tape_idx[idx0])
                fixed_val = @inbounds tape_val[val0]
                @inbounds x_org[col] = fixed_val
            end
        elseif reduction_type == Int32(FIXED_COL_INF)
            if tid == 1
                fix_to_pos_inf = @inbounds tape_idx[idx0] > 0
                col = Int(@inbounds tape_idx[idx0 + 1])
                n_rows = Int(round(@inbounds tape_val[val0]))
                extreme_val = @inbounds tape_val[val0 + 1]

                idx_pos = idx0 + 2
                val_pos = val0 + 2
                for _ in 1:n_rows
                    row_len = Int(@inbounds tape_idx[idx_pos])
                    side = @inbounds tape_val[val_pos]
                    coeff = 0.0

                    for j in 1:row_len
                        row_col = Int(@inbounds tape_idx[idx_pos + j])
                        row_coeff = @inbounds tape_val[val_pos + j]
                        if row_col == col
                            coeff = row_coeff
                        else
                            side -= row_coeff * @inbounds(x_org[row_col])
                        end
                    end

                    candidate = side / coeff
                    if fix_to_pos_inf
                        extreme_val = max(extreme_val, candidate)
                    else
                        extreme_val = min(extreme_val, candidate)
                    end

                    idx_pos += row_len + 1
                    val_pos += row_len + 1
                end

                @inbounds x_org[col] = extreme_val
            end
        end

        sync_threads()
    end
    return
end

function _kernel_replay_dual_tape_all!(
    y_org,
    z_org,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    record_count,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            idx0 = Int(@inbounds idx_starts[k])
            val0 = Int(@inbounds val_starts[k])

            if reduction_type == Int32(SUB_COL)
                elim_col = Int(@inbounds tape_idx[idx0])
                row = Int(@inbounds tape_idx[idx0 + 1])
                pivot_coeff = @inbounds tape_val[val0]
                elim_obj = @inbounds tape_val[val0 + 4]
                row_deleted = @inbounds tape_val[val0 + 5] != 0.0

                if row_deleted
                    @inbounds y_org[row] = elim_obj / pivot_coeff
                    @inbounds z_org[elim_col] = 0.0
                else
                    @inbounds y_org[row] += elim_obj / pivot_coeff
                    @inbounds z_org[elim_col] = elim_obj - pivot_coeff * y_org[row]
                end
            elseif reduction_type == Int32(PARALLEL_COL)
                from_idx = Int(@inbounds tape_idx[idx0])
                to_idx = Int(@inbounds tape_idx[idx0 + 1])
                ratio = @inbounds tape_val[val0]

                @inbounds z_org[from_idx] = ratio * z_org[to_idx]
            elseif reduction_type == Int32(FIXED_COL)
                idx1 = Int(@inbounds idx_starts[k + 1]) - 1
                col = Int(@inbounds tape_idx[idx0])
                zj = @inbounds tape_val[val0 + 1]
                for pos in (idx0 + 1):idx1
                    row = Int(@inbounds tape_idx[pos])
                    coeff = @inbounds tape_val[val0 + 1 + (pos - idx0)]
                    zj -= coeff * @inbounds(y_org[row])
                end
                @inbounds z_org[col] = zj
            elseif reduction_type == Int32(FIXED_COL_INF)
                col = Int(@inbounds tape_idx[idx0 + 1])
                @inbounds z_org[col] = 0.0
            elseif reduction_type == Int32(EQ_TO_INEQ)
                row = Int(@inbounds tape_idx[idx0])
                @inbounds y_org[row] += @inbounds tape_val[val0]
            end
        end
    end
    return
end

function _kernel_replay_dual_tape_all_with_bound_changes!(
    x_org,
    y_org,
    z_org,
    z_retrieved,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    record_count,
    tol,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            idx0 = Int(@inbounds idx_starts[k])
            val0 = Int(@inbounds val_starts[k])

            if reduction_type == Int32(SUB_COL)
                elim_col = Int(@inbounds tape_idx[idx0])
                row = Int(@inbounds tape_idx[idx0 + 1])
                pivot_coeff = @inbounds tape_val[val0]
                elim_obj = @inbounds tape_val[val0 + 4]
                row_deleted = @inbounds tape_val[val0 + 5] != 0.0

                if row_deleted
                    @inbounds y_org[row] = elim_obj / pivot_coeff
                    @inbounds z_org[elim_col] = 0.0
                else
                    @inbounds y_org[row] += elim_obj / pivot_coeff
                    @inbounds z_org[elim_col] = elim_obj - pivot_coeff * y_org[row]
                end
                @inbounds z_retrieved[elim_col] = UInt8(1)
            elseif reduction_type == Int32(PARALLEL_COL)
                from_idx = Int(@inbounds tape_idx[idx0])
                to_idx = Int(@inbounds tape_idx[idx0 + 1])
                ratio = @inbounds tape_val[val0]

                @inbounds z_org[from_idx] = ratio * z_org[to_idx]
                @inbounds z_retrieved[from_idx] = UInt8(1)
            elseif reduction_type == Int32(FIXED_COL)
                idx1 = Int(@inbounds idx_starts[k + 1]) - 1
                col = Int(@inbounds tape_idx[idx0])
                fixed_val = @inbounds tape_val[val0]
                zj = @inbounds tape_val[val0 + 1]
                for pos in (idx0 + 1):idx1
                    row = Int(@inbounds tape_idx[pos])
                    coeff = @inbounds tape_val[val0 + 1 + (pos - idx0)]
                    zj -= coeff * @inbounds(y_org[row])
                end
                @inbounds x_org[col] = fixed_val
                @inbounds z_org[col] = zj
                @inbounds z_retrieved[col] = UInt8(1)
            elseif reduction_type == Int32(FIXED_COL_INF)
                col = Int(@inbounds tape_idx[idx0 + 1])
                @inbounds z_org[col] = 0.0
                @inbounds z_retrieved[col] = UInt8(1)
            elseif reduction_type == Int32(EQ_TO_INEQ)
                row = Int(@inbounds tape_idx[idx0])
                @inbounds y_org[row] += @inbounds tape_val[val0]
            elseif reduction_type == Int32(BOUND_CHANGE_THE_ROW)
                col = Int(@inbounds tape_idx[idx0])
                row = Int(@inbounds tape_idx[idx0 + 1])
                @inbounds z_retrieved[col] != UInt8(0) || continue

                old_l = @inbounds tape_val[val0]
                old_u = @inbounds tape_val[val0 + 1]
                new_l = @inbounds tape_val[val0 + 2]
                new_u = @inbounds tape_val[val0 + 3]

                lower_changed = new_l > old_l + tol
                upper_changed = new_u < old_u - tol
                (lower_changed ⊻ upper_changed) || continue

                implied_bound = lower_changed ? new_l : new_u
                original_other_bound = lower_changed ? old_u : old_l
                original_other_is_lower = !lower_changed

                xj = @inbounds x_org[col]
                zj = @inbounds z_org[col]

                if isfinite(original_other_bound) && abs(xj - original_other_bound) <= tol
                    sign_ok = original_other_is_lower ? (zj >= -tol) : (zj <= tol)
                    sign_ok && continue
                end

                abs(xj - implied_bound) <= tol || continue

                row_start = Int(@inbounds A_rowPtr[row])
                row_stop = Int(@inbounds A_rowPtr[row + 1]) - 1
                row_start <= row_stop || continue

                aij = 0.0
                for p in row_start:row_stop
                    if Int(@inbounds A_colVal[p]) == col
                        aij = @inbounds A_nzVal[p]
                        break
                    end
                end
                abs(aij) > tol || continue

                delta_y = zj / aij
                @inbounds y_org[row] += delta_y

                for p in row_start:row_stop
                    kcol = Int(@inbounds A_colVal[p])
                    if kcol == col || @inbounds(z_retrieved[kcol] == UInt8(0))
                        continue
                    end
                    coeff = @inbounds A_nzVal[p]
                    @inbounds z_org[kcol] -= (coeff / aij) * zj
                end

                @inbounds z_org[col] = 0.0
                @inbounds z_retrieved[col] = UInt8(1)
            end
        end
    end
    return
end

function postsolve_replay_x_tape_gpu!(
    x_org::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return x_org

    @cuda threads=GPU_PRESOLVE_THREADS blocks=1 _kernel_replay_x_tape_all!(
        x_org,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        Int32(record_count),
    )
    return x_org
end

function postsolve_replay_dual_tape_gpu!(
    x_org::CuVector{T},
    y_org::CuVector{T},
    z_org::CuVector{T},
    z_retrieved::CuVector{UInt8},
    rec::PresolveRecord_gpu;
    original_model_gpu::Union{Nothing,LP_info_gpu}=nothing,
    tol::Float64=1.0e-7,
) where {T}
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return (y_org, z_org)

    if original_model_gpu === nothing
        @cuda threads=1 blocks=1 _kernel_replay_dual_tape_all!(
            y_org,
            z_org,
            tape_gpu.types,
            tape_gpu.index_starts,
            tape_gpu.value_starts,
            tape_gpu.indices,
            tape_gpu.vals,
            Int32(record_count),
        )
    else
        @cuda threads=1 blocks=1 _kernel_replay_dual_tape_all_with_bound_changes!(
            x_org,
            y_org,
            z_org,
            z_retrieved,
            tape_gpu.types,
            tape_gpu.index_starts,
            tape_gpu.value_starts,
            tape_gpu.indices,
            tape_gpu.vals,
            original_model_gpu.A.rowPtr,
            original_model_gpu.A.colVal,
            original_model_gpu.A.nzVal,
            Int32(record_count),
            tol,
        )
    end
    return (y_org, z_org)
end

function _apply_original_dual_local_replay_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model_gpu::LP_info_gpu;
    tol::Float64=1.0e-7,
)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    if record_count == 0 && isempty(rec.fixed_idx)
        return (x_org, y_org, z_org)
    end

    ATy = similar(z_org)
    mul!(ATy, original_model_gpu.AT, y_org)

    if record_count > 0
        @cuda threads=1 blocks=1 _kernel_refine_deleted_rows_from_tape_gpu!(
            x_org,
            y_org,
            z_org,
            ATy,
            tape_gpu.types,
            tape_gpu.index_starts,
            tape_gpu.value_starts,
            tape_gpu.indices,
            tape_gpu.vals,
            original_model_gpu.c,
            original_model_gpu.l,
            original_model_gpu.u,
            Int32(record_count),
            tol,
        )

        dual_eps = min(tol, 1.0e-12)
        @cuda threads=1 blocks=1 _kernel_refine_sub_cols_from_tape_gpu!(
            x_org,
            y_org,
            z_org,
            ATy,
            tape_gpu.types,
            tape_gpu.index_starts,
            tape_gpu.value_starts,
            tape_gpu.indices,
            tape_gpu.vals,
            original_model_gpu.A.rowPtr,
            original_model_gpu.A.colVal,
            original_model_gpu.A.nzVal,
            original_model_gpu.AT.rowPtr,
            original_model_gpu.c,
            original_model_gpu.l,
            original_model_gpu.u,
            Int32(record_count),
            tol,
            dual_eps,
        )

        visited_rows = CUDA.zeros(UInt8, length(y_org))
        @cuda threads=1 blocks=1 _kernel_refine_row_duals_from_tape_gpu!(
            visited_rows,
            x_org,
            y_org,
            z_org,
            ATy,
            tape_gpu.types,
            tape_gpu.index_starts,
            tape_gpu.indices,
            original_model_gpu.A.rowPtr,
            original_model_gpu.A.colVal,
            original_model_gpu.A.nzVal,
            original_model_gpu.c,
            original_model_gpu.AL,
            original_model_gpu.AU,
            original_model_gpu.l,
            original_model_gpu.u,
            Int32(record_count),
            tol,
            dual_eps,
        )

        postsolve_project_column_duals_from_ATy_gpu!(
            x_org,
            z_org,
            ATy,
            rec,
            original_model_gpu;
            tol=tol,
        )
    end

    postsolve_refine_fixed_column_duals_from_ATy_gpu!(
        x_org,
        z_org,
        ATy,
        rec,
        original_model_gpu;
        tol=tol,
    )
    postsolve_cleanup_unbounded_dual_slacks_gpu!(
        z_org,
        original_model_gpu;
        tol=tol,
    )

    return (x_org, y_org, z_org)
end

function _apply_original_dual_exact_local_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model_gpu::LP_info_gpu;
    tol::Float64=1.0e-7,
    protected_cols::Union{Nothing,CuVector{UInt8}}=nothing,
)
    postsolve_refine_deleted_rows_from_original_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
    )
    postsolve_refine_sub_cols_from_original_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
    )
    postsolve_refine_row_duals_from_original_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
        protected_cols=protected_cols,
    )
    postsolve_refine_singleton_chain_rows_from_original_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
        protected_cols=protected_cols,
    )
    postsolve_refine_fixed_column_duals_from_original_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
    )
    postsolve_cleanup_unbounded_dual_slacks_gpu!(
        z_org,
        original_model_gpu;
        tol=tol,
    )

    return (x_org, y_org, z_org)
end

"""
Restore `(x, y, z)` from reduced space to original space on GPU.

Returns `(x_org, y_org, z_org)` as `CuVector`s.
"""
function _needs_global_original_dual_refinement(rec::PresolveRecord_gpu)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return false

    needs_refine = CUDA.zeros(UInt8, 1)
    @cuda threads=1 blocks=1 _kernel_detect_global_original_dual_refinement!(
        needs_refine,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        Int32(record_count),
    )
    return _copy_scalar_to_host(needs_refine, 1) != UInt8(0)
end

function _accept_original_dual_refinement_metrics(
    before,
    after;
    metric_atol::Float64=1.0e-10,
)
    before_p = before[3]
    before_d = before[4]
    before_gap = before[5]
    after_p = after[3]
    after_d = after[4]
    after_gap = after[5]

    all(isfinite, (after_p, after_d, after_gap)) || return false

    p_ok = after_p <= before_p + metric_atol
    d_ok = after_d <= before_d + metric_atol
    gap_ok = after_gap <= before_gap + metric_atol
    improved_p = after_p < before_p - metric_atol
    improved_d = after_d < before_d - metric_atol
    improved_gap = after_gap < before_gap - metric_atol

    return p_ok && d_ok && gap_ok && (improved_p || improved_d || improved_gap)
end

function _accept_original_dual_refinement_dual_priority(
    before,
    after;
    metric_atol::Float64=1.0e-10,
)
    before_p = before[3]
    before_d = before[4]
    after_p = after[3]
    after_d = after[4]
    after_gap = after[5]

    all(isfinite, (after_p, after_d, after_gap)) || return false
    p_ok = after_p <= before_p + metric_atol
    improved_d = after_d < before_d - metric_atol
    return p_ok && improved_d
end

function _apply_original_dual_refinement_guarded_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model_gpu::LP_info_gpu;
    tol::Float64=1.0e-7,
    max_global_iters::Int=10,
)
    kktws = allocate_kkt_workspace_gpu(original_model_gpu)
    needs_global_refine = _needs_global_original_dual_refinement(rec)
    protected_cols = _build_exact_replay_column_mask_gpu(rec, length(z_org))

    _apply_original_dual_local_replay_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
    )
    _apply_original_dual_exact_local_gpu!(
        x_org,
        y_org,
        z_org,
        rec,
        original_model_gpu;
        tol=tol,
        protected_cols=protected_cols,
    )
    best_metrics = compute_original_kkt_metrics_gpu!(kktws, original_model_gpu, x_org, y_org, z_org)
    y_best = copy(y_org)
    z_best = copy(z_org)

    if needs_global_refine && (best_metrics[4] > tol || best_metrics[5] > tol)
        y_trial = copy(y_org)
        z_trial = copy(z_org)
        postsolve_refine_duals_globally_from_original_gpu!(
            x_org,
            y_trial,
            z_trial,
            original_model_gpu;
            tol=tol,
            max_iters=max_global_iters,
            rec=rec,
            protected_cols=protected_cols,
        )
        postsolve_cleanup_unbounded_dual_slacks_gpu!(
            z_trial,
            original_model_gpu;
            tol=tol,
        )
        global_metrics = compute_original_kkt_metrics_gpu!(kktws, original_model_gpu, x_org, y_trial, z_trial)
        if _accept_original_dual_refinement_metrics(best_metrics, global_metrics) ||
           _accept_original_dual_refinement_dual_priority(best_metrics, global_metrics)
            y_best = y_trial
            z_best = z_trial
        end
    end

    copyto!(y_org, y_best)
    copyto!(z_org, z_best)

    return (x_org, y_org, z_org)
end

function postsolve_gpu(
    x_red,
    y_red,
    z_red,
    rec::PresolveRecord_gpu;
    presolve_params::Union{Nothing, PresolveParams}=nothing,
    original_model_gpu::Union{Nothing, LP_info_gpu}=nothing,
    apply_original_dual_refinement::Bool=false,
)
    _params = (presolve_params !== nothing) ? presolve_params : PresolveParams()
    x_red_d = x_red isa CuVector ? x_red : CuVector(x_red)
    y_red_d = y_red isa CuVector ? y_red : CuVector(y_red)
    z_red_d = z_red isa CuVector ? z_red : CuVector(z_red)

    if _params.debug_checks
        postsolve_basic_checks(x_red_d, y_red_d, z_red_d, rec)
    end

    @assert length(x_red_d) == Int(rec.n1)
    @assert length(y_red_d) == Int(rec.m1)
    @assert length(z_red_d) == Int(rec.n1)

    x_org = postsolve_restore_x_gpu(x_red_d, rec)
    y_org = postsolve_restore_y_gpu(y_red_d, rec)
    z_org, z_retrieved = postsolve_restore_z_gpu_with_mask(z_red_d, rec)
    postsolve_replay_dual_tape_gpu!(
        x_org,
        y_org,
        z_org,
        z_retrieved,
        rec;
        original_model_gpu=original_model_gpu,
        tol=1.0e-7,
    )
    if !apply_original_dual_refinement && original_model_gpu !== nothing
        _apply_original_dual_local_replay_gpu!(
            x_org,
            y_org,
            z_org,
            rec,
            original_model_gpu;
            tol=1.0e-7,
        )
    elseif apply_original_dual_refinement && original_model_gpu !== nothing
        _apply_original_dual_refinement_guarded_gpu!(
            x_org,
            y_org,
            z_org,
            rec,
            original_model_gpu;
            tol=1.0e-7,
            max_global_iters=10,
        )
    end

    return (x_org, y_org, z_org)
end

"""
Restore primal `x` from reduced to original space.
"""
function postsolve_restore_x_gpu(
    x_red::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    x_org = CUDA.zeros(T, Int(rec.n0))
    scatter_by_red2org!(x_org, x_red, rec.col_red2org)
    postsolve_restore_fixed_x_gpu!(x_org, rec)
    postsolve_replay_x_tape_gpu!(x_org, rec)
    return x_org
end

@inline function _choose_merged_source_value(lo::Float64, hi::Float64)
    if isfinite(lo)
        return lo
    elseif isfinite(hi)
        return hi
    end
    return 0.0
end

"""
Restore row-dual vector `y` from reduced to original row space.
"""
function postsolve_restore_y_gpu(
    y_red::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    y_org = CUDA.zeros(T, Int(rec.m0))
    postsolve_restore_y_gpu!(y_org, y_red, rec)
    return y_org
end

"""
In-place row-dual restore helper.
"""
function postsolve_restore_y_gpu!(
    y_org::CuVector{T},
    y_red::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    @assert length(y_org) == Int(rec.m0)
    @assert length(y_red) == Int(rec.m1)
    fill!(y_org, zero(T))
    scatter_by_red2org!(y_org, y_red, rec.row_red2org)
    return y_org
end

"""
Restore reduced-cost-like vector `z` from reduced to original column space.

Surviving columns are scattered directly from reduced `z`.
Columns removed during presolve remain zero here and are recovered later by
original-model refinement when `original_model_gpu` is provided.
"""
function postsolve_restore_z_gpu(
    z_red::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    z_org, _ = postsolve_restore_z_gpu_with_mask(z_red, rec)
    return z_org
end

function postsolve_restore_z_gpu_with_mask(
    z_red::CuVector{T},
    rec::PresolveRecord_gpu,
) where {T}
    z_org = CUDA.zeros(T, Int(rec.n0))
    scatter_by_red2org!(z_org, z_red, rec.col_red2org)
    z_retrieved = CUDA.zeros(UInt8, Int(rec.n0))
    if Int(rec.n1) > 0
        scatter_by_red2org!(
            z_retrieved,
            CUDA.fill(UInt8(1), Int(rec.n1)),
            rec.col_red2org,
        )
    end
    return z_org, z_retrieved
end

@inline function _postsolve_at_lower(x::Float64, l::Float64, tol::Float64)
    return isfinite(l) && x <= l + tol
end

@inline function _postsolve_at_upper(x::Float64, u::Float64, tol::Float64)
    return isfinite(u) && x >= u - tol
end

function _postsolve_dual_interval(
    s::Float64,
    a::Float64,
    x::Float64,
    l::Float64,
    u::Float64,
    tol::Float64,
)
    at_lower = _postsolve_at_lower(x, l, tol)
    at_upper = _postsolve_at_upper(x, u, tol)

    if at_lower && at_upper
        return (-Inf, Inf)
    end

    thresh = s / a
    if !at_lower && !at_upper
        return (thresh, thresh)
    elseif at_lower
        return a > 0.0 ? (-Inf, thresh) : (thresh, Inf)
    else
        return a > 0.0 ? (thresh, Inf) : (-Inf, thresh)
    end
end

@inline function _clamp_with_inf(x::Float64, lo::Float64, hi::Float64)
    y = x
    if isfinite(lo)
        y = max(y, lo)
    end
    if isfinite(hi)
        y = min(y, hi)
    end
    return y
end

@inline function _column_bound_sign_violation(
    x::Float64,
    z::Float64,
    l::Float64,
    u::Float64,
    tol::Float64,
)
    at_lower = _postsolve_at_lower(x, l, tol)
    at_upper = _postsolve_at_upper(x, u, tol)

    if at_lower && at_upper
        return 0.0
    elseif !at_lower && !at_upper
        return abs(z)
    elseif at_lower
        return max(-z, 0.0)
    else
        return max(z, 0.0)
    end
end

function _cleanup_unbounded_dual_slacks!(
    z_org::Vector{Float64},
    l::AbstractVector{Float64},
    u::AbstractVector{Float64};
    tol::Float64=1e-12,
)
    dual_eps = min(tol, 1e-12)
    for j in eachindex(z_org)
        zj = z_org[j]
        if zj > dual_eps && !isfinite(l[j])
            z_org[j] = 0.0
        elseif zj < -dual_eps && !isfinite(u[j])
            z_org[j] = 0.0
        end
    end
    return z_org
end

function _kernel_cleanup_unbounded_dual_slacks_gpu!(
    z_org,
    l,
    u,
    n,
    dual_eps,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds zj = z_org[i]
        @inbounds lj = l[i]
        @inbounds uj = u[i]
        if zj > dual_eps && !isfinite(lj)
            @inbounds z_org[i] = 0.0
        elseif zj < -dual_eps && !isfinite(uj)
            @inbounds z_org[i] = 0.0
        end
    end
    return
end

function postsolve_cleanup_unbounded_dual_slacks_gpu!(
    z_org::CuVector{Float64},
    original_model::LP_info_gpu;
    tol::Float64=1e-12,
)
    n = length(z_org)
    n == 0 && return z_org

    dual_eps = min(tol, 1.0e-12)
    blocks = cld(n, GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_cleanup_unbounded_dual_slacks_gpu!(
        z_org,
        original_model.l,
        original_model.u,
        Int32(n),
        dual_eps,
    )

    return z_org
end

function _kernel_project_column_duals_from_ATy_gpu!(
    x_org,
    z_org,
    ATy,
    c,
    l,
    u,
    protected_cols,
    n,
    tol,
    dual_eps,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        if length(protected_cols) > 0 && @inbounds(protected_cols[i] != UInt8(0))
            return
        end

        @inbounds xj = x_org[i]
        @inbounds lj = l[i]
        @inbounds uj = u[i]
        @inbounds r = c[i] - ATy[i]

        at_lower = isfinite(lj) && xj <= lj + tol
        at_upper = isfinite(uj) && xj >= uj - tol

        zj = if at_lower && at_upper
            r
        elseif !at_lower && !at_upper
            0.0
        elseif at_lower
            max(r, 0.0)
        else
            min(r, 0.0)
        end

        if zj > dual_eps && !isfinite(lj)
            zj = 0.0
        elseif zj < -dual_eps && !isfinite(uj)
            zj = 0.0
        end

        @inbounds z_org[i] = zj
    end
    return
end

function _kernel_project_selected_column_duals_from_ATy_gpu!(
    selected_cols,
    x_org,
    z_org,
    ATy,
    c,
    l,
    u,
    k,
    tol,
    dual_eps,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= k
        j = Int(@inbounds selected_cols[i])
        @inbounds xj = x_org[j]
        @inbounds lj = l[j]
        @inbounds uj = u[j]
        @inbounds r = c[j] - ATy[j]

        at_lower = isfinite(lj) && xj <= lj + tol
        at_upper = isfinite(uj) && xj >= uj - tol

        zj = if at_lower && at_upper
            r
        elseif !at_lower && !at_upper
            0.0
        elseif at_lower
            max(r, 0.0)
        else
            min(r, 0.0)
        end

        if zj > dual_eps && !isfinite(lj)
            zj = 0.0
        elseif zj < -dual_eps && !isfinite(uj)
            zj = 0.0
        end

        @inbounds z_org[j] = zj
    end
    return
end

function _kernel_set_selected_column_duals_from_ATy_gpu!(
    selected_cols,
    z_org,
    ATy,
    c,
    k,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= k
        j = Int(@inbounds selected_cols[i])
        @inbounds z_org[j] = c[j] - ATy[j]
    end
    return
end

function postsolve_project_column_duals_from_ATy_gpu!(
    x_org::CuVector{Float64},
    z_org::CuVector{Float64},
    ATy::CuVector{Float64},
    rec::Union{PresolveRecord_gpu, Nothing},
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
    protected_cols::Union{Nothing,CuVector{UInt8}}=nothing,
)
    n = length(z_org)
    n == 0 && return z_org

    blocks = cld(n, GPU_PRESOLVE_THREADS)
    dual_eps = min(tol, 1.0e-12)
    protected_cols_d = if protected_cols !== nothing
        protected_cols
    elseif rec !== nothing
        _build_exact_replay_column_mask_gpu(rec, n)
    else
        CuVector{UInt8}(undef, 0)
    end

    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_project_column_duals_from_ATy_gpu!(
        x_org,
        z_org,
        ATy,
        original_model.c,
        original_model.l,
        original_model.u,
        protected_cols_d,
        Int32(n),
        tol,
        dual_eps,
    )

    return z_org
end

function postsolve_project_selected_column_duals_from_ATy_gpu!(
    x_org::CuVector{Float64},
    z_org::CuVector{Float64},
    ATy::CuVector{Float64},
    selected_cols::CuVector{Int32},
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    k = length(selected_cols)
    k == 0 && return z_org

    blocks = cld(k, GPU_PRESOLVE_THREADS)
    dual_eps = min(tol, 1.0e-12)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_project_selected_column_duals_from_ATy_gpu!(
        selected_cols,
        x_org,
        z_org,
        ATy,
        original_model.c,
        original_model.l,
        original_model.u,
        Int32(k),
        tol,
        dual_eps,
    )

    return z_org
end

function postsolve_refine_fixed_column_duals_from_ATy_gpu!(
    x_org::CuVector{Float64},
    z_org::CuVector{Float64},
    ATy::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    isempty(rec.fixed_idx) && return (x_org, z_org)

    @assert length(x_org) == length(original_model.c)
    @assert length(z_org) == length(original_model.c)
    @assert length(ATy) == length(original_model.c)

    blocks = cld(length(rec.fixed_idx), GPU_PRESOLVE_THREADS)
    @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_set_selected_column_duals_from_ATy_gpu!(
        rec.fixed_idx,
        z_org,
        ATy,
        original_model.c,
        Int32(length(rec.fixed_idx)),
    )

    return (x_org, z_org)
end

function postsolve_refine_fixed_column_duals_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    isempty(rec.fixed_idx) && return (x_org, y_org, z_org)

    @assert length(x_org) == length(original_model.c)
    @assert length(z_org) == length(original_model.c)
    @assert length(y_org) == length(original_model.AL)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)

    postsolve_refine_fixed_column_duals_from_ATy_gpu!(
        x_org,
        z_org,
        ATy,
        rec,
        original_model;
        tol=tol,
    )

    return (x_org, y_org, z_org)
end

function postsolve_refine_column_duals_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    n = length(z_org)
    n == 0 && return (x_org, y_org, z_org)

    @assert length(x_org) == length(original_model.c)
    @assert length(z_org) == length(original_model.c)
    @assert length(y_org) == length(original_model.AL)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)

    postsolve_project_column_duals_from_ATy_gpu!(
        x_org,
        z_org,
        ATy,
        nothing,
        original_model;
        tol=tol,
    )

    return (x_org, y_org, z_org)
end

@inline function _clamp_with_inf_gpu(x::Float64, lo::Float64, hi::Float64)
    y = x
    if isfinite(lo)
        y = max(y, lo)
    end
    if isfinite(hi)
        y = min(y, hi)
    end
    return y
end

@inline function _device_refine_live_row_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    r::Int,
    tol,
    dual_eps,
)
    row_start = Int(@inbounds A_rowPtr[r])
    row_stop = Int(@inbounds A_rowPtr[r + 1]) - 1
    row_start <= row_stop || return nothing

    y_old = @inbounds y_org[r]
    lo = -Inf
    hi = Inf
    exact_sum = 0.0
    exact_count = 0
    activity = 0.0

    for p in row_start:row_stop
        coeff = @inbounds A_nzVal[p]
        abs(coeff) > tol || continue

        col = Int(@inbounds A_colVal[p])
        xj = @inbounds x_org[col]
        lj = @inbounds l[col]
        uj = @inbounds u[col]
        activity += coeff * xj

        at_lower = isfinite(lj) && xj <= lj + tol
        at_upper = isfinite(uj) && xj >= uj - tol

        lo_col = -Inf
        hi_col = Inf
        if !(at_lower && at_upper)
            s = @inbounds c[col] - (@inbounds ATy[col] - coeff * y_old)
            thresh = s / coeff
            if !at_lower && !at_upper
                lo_col = thresh
                hi_col = thresh
                exact_sum += thresh
                exact_count += 1
            elseif at_lower
                if coeff > 0.0
                    hi_col = thresh
                else
                    lo_col = thresh
                end
            else
                if coeff > 0.0
                    lo_col = thresh
                else
                    hi_col = thresh
                end
            end
        end

        lo = max(lo, lo_col)
        hi = min(hi, hi_col)
    end

    row_l = @inbounds AL[r]
    row_u = @inbounds AU[r]
    row_at_lower = isfinite(row_l) && activity <= row_l + tol
    row_at_upper = isfinite(row_u) && activity >= row_u - tol
    row_exact = !row_at_lower && !row_at_upper

    lo_row = if row_at_lower && row_at_upper
        -Inf
    elseif row_exact
        0.0
    elseif row_at_lower
        0.0
    else
        -Inf
    end
    hi_row = if row_at_lower && row_at_upper
        Inf
    elseif row_exact
        0.0
    elseif row_at_lower
        Inf
    else
        0.0
    end

    lo = max(lo, lo_row)
    hi = min(hi, hi_row)

    guess = row_exact ? 0.0 : y_old
    y_new = if row_exact
        0.0
    elseif exact_count > 0
        target = exact_sum / exact_count
        if lo <= hi + tol
            _clamp_with_inf_gpu(target, lo, hi)
        else
            _clamp_with_inf_gpu(target, min(lo, hi), max(lo, hi))
        end
    elseif lo <= hi + tol
        _clamp_with_inf_gpu(guess, lo, hi)
    else
        _clamp_with_inf_gpu(guess, min(lo, hi), max(lo, hi))
    end

    @inbounds y_org[r] = y_new
    delta = y_new - y_old

    if abs(delta) > tol
        for p in row_start:row_stop
            coeff = @inbounds A_nzVal[p]
            abs(coeff) > tol || continue
            col = Int(@inbounds A_colVal[p])
            @inbounds ATy[col] += coeff * delta
        end
    end

    for p in row_start:row_stop
        coeff = @inbounds A_nzVal[p]
        abs(coeff) > tol || continue
        col = Int(@inbounds A_colVal[p])
        @inbounds z_org[col] = c[col] - ATy[col]
    end

    return nothing
end

function _kernel_refine_live_row_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    row,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        _device_refine_live_row_dual_from_original_gpu!(
            x_org,
            y_org,
            z_org,
            ATy,
            A_rowPtr,
            A_colVal,
            A_nzVal,
            c,
            AL,
            AU,
            l,
            u,
            Int(row),
            tol,
            dual_eps,
        )
    end
    return
end

function _kernel_refine_row_duals_from_tape_gpu!(
    visited_rows,
    x_org,
    y_org,
    z_org,
    ATy,
    reduction_types,
    idx_starts,
    tape_idx,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    record_count,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            idx0 = Int(@inbounds idx_starts[k])
            idx1 = Int(@inbounds idx_starts[k + 1]) - 1

            row = Int32(0)
            if reduction_type == Int32(BOUND_CHANGE_THE_ROW)
                idx1 >= idx0 + 1 || continue
                row = @inbounds tape_idx[idx0 + 1]
            elseif reduction_type == Int32(LHS_CHANGE) ||
                   reduction_type == Int32(RHS_CHANGE)
                idx1 >= idx0 || continue
                row = @inbounds tape_idx[idx0]
            else
                continue
            end

            ri = Int(row)
            1 <= ri <= length(visited_rows) || continue
            @inbounds visited_rows[ri] != UInt8(0) && continue
            @inbounds visited_rows[ri] = UInt8(1)
            _device_refine_live_row_dual_from_original_gpu!(
                x_org,
                y_org,
                z_org,
                ATy,
                A_rowPtr,
                A_colVal,
                A_nzVal,
                c,
                AL,
                AU,
                l,
                u,
                ri,
                tol,
                dual_eps,
            )
        end
    end
    return
end

function _kernel_collect_live_row_refinement_targets_from_tape_gpu!(
    row_targets,
    target_count,
    seen_rows,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    record_count,
    singleton_chain_only,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds target_count[1] = Int32(0)
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            row = Int32(0)

            if singleton_chain_only == UInt8(0)
                idx0 = Int(@inbounds idx_starts[k])
                idx1 = Int(@inbounds idx_starts[k + 1]) - 1
                if reduction_type == Int32(BOUND_CHANGE_THE_ROW)
                    idx1 >= idx0 + 1 || continue
                    row = @inbounds tape_idx[idx0 + 1]
                elseif reduction_type == Int32(LHS_CHANGE) ||
                       reduction_type == Int32(RHS_CHANGE)
                    idx1 >= idx0 || continue
                    row = @inbounds tape_idx[idx0]
                else
                    continue
                end
            else
                reduction_type == Int32(SUB_COL) || continue
                idx0 = Int(@inbounds idx_starts[k])
                idx1 = Int(@inbounds idx_starts[k + 1]) - 1
                val0 = Int(@inbounds val_starts[k])
                val1 = Int(@inbounds val_starts[k + 1]) - 1
                idx1 >= idx0 + 3 || continue
                val1 >= val0 + 5 || continue

                support_count = Int(@inbounds tape_idx[idx0 + 2])
                support_count == 1 || continue
                row_deleted = @inbounds tape_val[val0 + 5] != 0.0
                row_deleted && continue
                row = @inbounds tape_idx[idx0 + 1]
            end

            ri = Int(row)
            1 <= ri <= length(seen_rows) || continue
            @inbounds seen_rows[ri] != UInt8(0) && continue
            next_pos = Int(@inbounds target_count[1]) + 1
            @inbounds row_targets[next_pos] = row
            @inbounds target_count[1] = Int32(next_pos)
            @inbounds seen_rows[ri] = UInt8(1)
        end
    end
    return
end

function _kernel_refine_live_rows_from_list_gpu!(
    row_targets,
    target_count,
    x_org,
    y_org,
    z_org,
    ATy,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for pos in 1:Int(target_count)
            row = Int(@inbounds row_targets[pos])
            _device_refine_live_row_dual_from_original_gpu!(
                x_org,
                y_org,
                z_org,
                ATy,
                A_rowPtr,
                A_colVal,
                A_nzVal,
                c,
                AL,
                AU,
                l,
                u,
                row,
                tol,
                dual_eps,
            )
        end
    end
    return
end

function _kernel_refine_singleton_chain_rows_from_tape_gpu!(
    visited_rows,
    x_org,
    y_org,
    z_org,
    ATy,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    record_count,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            reduction_type == Int32(SUB_COL) || continue

            idx0 = Int(@inbounds idx_starts[k])
            idx1 = Int(@inbounds idx_starts[k + 1]) - 1
            val0 = Int(@inbounds val_starts[k])
            val1 = Int(@inbounds val_starts[k + 1]) - 1
            idx1 >= idx0 + 3 || continue
            val1 >= val0 + 5 || continue

            support_count = Int(@inbounds tape_idx[idx0 + 2])
            support_count == 1 || continue
            row_deleted = @inbounds tape_val[val0 + 5] != 0.0
            row_deleted && continue

            row = Int(@inbounds tape_idx[idx0 + 1])
            1 <= row <= length(visited_rows) || continue
            @inbounds visited_rows[row] != UInt8(0) && continue
            @inbounds visited_rows[row] = UInt8(1)
            _device_refine_live_row_dual_from_original_gpu!(
                x_org,
                y_org,
                z_org,
                ATy,
                A_rowPtr,
                A_colVal,
                A_nzVal,
                c,
                AL,
                AU,
                l,
                u,
                row,
                tol,
                dual_eps,
            )
        end
    end
    return
end

function postsolve_refine_row_duals_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
    protected_cols::Union{Nothing,CuVector{UInt8}}=nothing,
)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return (x_org, y_org, z_org)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)
    seen_rows = CUDA.zeros(UInt8, length(y_org))
    row_targets = CuVector{Int32}(undef, length(y_org))
    target_count_d = CUDA.zeros(Int32, 1)

    @cuda threads=1 blocks=1 _kernel_collect_live_row_refinement_targets_from_tape_gpu!(
        row_targets,
        target_count_d,
        seen_rows,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        Int32(record_count),
        UInt8(0),
    )
    target_count = Int(Array(target_count_d)[1])
    target_count == 0 && return (x_org, y_org, z_org)

    dual_eps = min(tol, 1.0e-12)
    for _ in 1:10
        postsolve_project_column_duals_from_ATy_gpu!(
            x_org,
            z_org,
            ATy,
            rec,
            original_model;
            tol=tol,
            protected_cols=protected_cols,
        )
        @cuda threads=1 blocks=1 _kernel_refine_live_rows_from_list_gpu!(
            row_targets,
            Int32(target_count),
            x_org,
            y_org,
            z_org,
            ATy,
            original_model.A.rowPtr,
            original_model.A.colVal,
            original_model.A.nzVal,
            original_model.c,
            original_model.AL,
            original_model.AU,
            original_model.l,
            original_model.u,
            tol,
            dual_eps,
        )
    end

    return (x_org, y_org, z_org)
end

function _kernel_refine_deleted_singleton_row_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    c,
    l,
    u,
    row,
    col,
    coeff,
    row_lower,
    row_upper,
    tol,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        _device_refine_deleted_singleton_row_dual_from_original_gpu!(
            x_org,
            y_org,
            z_org,
            ATy,
            c,
            l,
            u,
            Int(row),
            Int(col),
            coeff,
            row_lower,
            row_upper,
            tol,
        )
    end
    return
end

@inline function _device_refine_deleted_singleton_row_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    c,
    l,
    u,
    r::Int,
    j::Int,
    a::Float64,
    row_lower::Float64,
    row_upper::Float64,
    tol,
)
    abs(a) > tol || return nothing

    s = @inbounds z_org[j]
    xj = @inbounds x_org[j]
    deleted_row_activity = a * xj

    lj = @inbounds l[j]
    uj = @inbounds u[j]
    col_lower = isfinite(lj) && xj <= lj + tol
    col_upper = isfinite(uj) && xj >= uj - tol
    col_exact = !col_lower && !col_upper

    row_lower_active = isfinite(row_lower) && deleted_row_activity <= row_lower + tol
    row_upper_active = isfinite(row_upper) && deleted_row_activity >= row_upper - tol
    row_exact = !row_lower_active && !row_upper_active

    lo_col = -Inf
    hi_col = Inf
    if !(col_lower && col_upper)
        thresh = s / a
        if !col_lower && !col_upper
            lo_col = thresh
            hi_col = thresh
        elseif col_lower
            if a > 0.0
                hi_col = thresh
            else
                lo_col = thresh
            end
        else
            if a > 0.0
                lo_col = thresh
            else
                hi_col = thresh
            end
        end
    end

    lo_row = if row_lower_active && row_upper_active
        -Inf
    elseif row_exact
        0.0
    elseif row_lower_active
        0.0
    else
        -Inf
    end
    hi_row = if row_lower_active && row_upper_active
        Inf
    elseif row_exact
        0.0
    elseif row_lower_active
        Inf
    else
        0.0
    end

    lo = max(lo_col, lo_row)
    hi = min(hi_col, hi_row)

    y_old = @inbounds y_org[r]
    y_new = if row_exact
        0.0
    elseif col_exact
        s / a
    elseif lo <= hi + tol
        _clamp_with_inf_gpu(0.0, lo, hi)
    else
        _clamp_with_inf_gpu(0.0, min(lo, hi), max(lo, hi))
    end

    @inbounds y_org[r] = y_new
    delta = y_new - y_old
    if abs(delta) > tol
        @inbounds ATy[j] += a * delta
    end
    @inbounds z_org[j] = c[j] - ATy[j]

    return nothing
end

function _kernel_refine_deleted_rows_from_tape_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    c,
    l,
    u,
    record_count,
    tol,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            reduction_type == Int32(DELETED_ROW) || continue

            idx0 = Int(@inbounds idx_starts[k])
            idx1 = Int(@inbounds idx_starts[k + 1]) - 1
            val0 = Int(@inbounds val_starts[k])
            val1 = Int(@inbounds val_starts[k + 1]) - 1
            idx1 >= idx0 + 1 || continue
            val1 >= val0 + 2 || continue

            _device_refine_deleted_singleton_row_dual_from_original_gpu!(
                x_org,
                y_org,
                z_org,
                ATy,
                c,
                l,
                u,
                Int(@inbounds tape_idx[idx0]),
                Int(@inbounds tape_idx[idx0 + 1]),
                @inbounds(tape_val[val0 + 2]),
                @inbounds(tape_val[val0]),
                @inbounds(tape_val[val0 + 1]),
                tol,
            )
        end
    end
    return
end

function postsolve_refine_deleted_rows_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return (x_org, y_org, z_org)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)
    @cuda threads=1 blocks=1 _kernel_refine_deleted_rows_from_tape_gpu!(
        x_org,
        y_org,
        z_org,
        ATy,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        original_model.c,
        original_model.l,
        original_model.u,
        Int32(record_count),
        tol,
    )

    return (x_org, y_org, z_org)
end

function _kernel_refine_sub_col_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    AT_rowPtr,
    c,
    l,
    u,
    row,
    elim_col,
    keep_col,
    pivot_coeff,
    keep_coeff,
    row_deleted,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        _device_refine_sub_col_dual_from_original_gpu!(
            x_org,
            y_org,
            z_org,
            ATy,
            A_rowPtr,
            A_colVal,
            A_nzVal,
            AT_rowPtr,
            c,
            l,
            u,
            Int(row),
            Int(elim_col),
            Int(keep_col),
            pivot_coeff,
            keep_coeff,
            row_deleted,
            tol,
            dual_eps,
        )
    end
    return
end

@inline function _device_refine_sub_col_dual_from_original_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    AT_rowPtr,
    c,
    l,
    u,
    r::Int,
    j_elim::Int,
    j_keep::Int,
    a_elim::Float64,
    a_keep::Float64,
    row_deleted::Bool,
    tol,
    dual_eps,
)
    row_deleted && return nothing
    abs(a_elim) > tol || return nothing
    abs(a_keep) > tol || return nothing

    elim_nnz = Int(@inbounds AT_rowPtr[j_elim + 1] - AT_rowPtr[j_elim])
    elim_nnz > 1 || return nothing

    y_old = @inbounds y_org[r]
    lo = -Inf
    hi = Inf
    exact_sum = 0.0
    exact_count = 0

    for (j, a) in ((j_elim, a_elim), (j_keep, a_keep))
        xj = @inbounds x_org[j]
        lj = @inbounds l[j]
        uj = @inbounds u[j]
        at_lower = isfinite(lj) && xj <= lj + tol
        at_upper = isfinite(uj) && xj >= uj - tol

        s = @inbounds c[j] - (@inbounds(ATy[j]) - a * y_old)

        lo_col = -Inf
        hi_col = Inf
        if !(at_lower && at_upper)
            thresh = s / a
            if !at_lower && !at_upper
                lo_col = thresh
                hi_col = thresh
                exact_sum += thresh
                exact_count += 1
            elseif at_lower
                if a > 0.0
                    hi_col = thresh
                else
                    lo_col = thresh
                end
            else
                if a > 0.0
                    lo_col = thresh
                else
                    hi_col = thresh
                end
            end
        end

        lo = max(lo, lo_col)
        hi = min(hi, hi_col)
    end

    guess = y_old
    y_new = if exact_count > 0
        target = exact_sum / exact_count
        if lo <= hi + tol
            _clamp_with_inf_gpu(target, lo, hi)
        else
            _clamp_with_inf_gpu(target, min(lo, hi), max(lo, hi))
        end
    elseif lo <= hi + tol
        _clamp_with_inf_gpu(guess, lo, hi)
    else
        _clamp_with_inf_gpu(guess, min(lo, hi), max(lo, hi))
    end

    @inbounds y_org[r] = y_new
    delta = y_new - y_old

    if abs(delta) > tol
        row_start = Int(@inbounds A_rowPtr[r])
        row_stop = Int(@inbounds A_rowPtr[r + 1]) - 1
        for p in row_start:row_stop
            coeff = @inbounds A_nzVal[p]
            abs(coeff) > tol || continue
            col = Int(@inbounds A_colVal[p])
            @inbounds ATy[col] += coeff * delta
        end
    end

    @inbounds z_org[j_elim] = c[j_elim] - ATy[j_elim]
    @inbounds z_org[j_keep] = c[j_keep] - ATy[j_keep]
    return nothing
end

function _kernel_refine_sub_cols_from_tape_gpu!(
    x_org,
    y_org,
    z_org,
    ATy,
    reduction_types,
    idx_starts,
    val_starts,
    tape_idx,
    tape_val,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    AT_rowPtr,
    c,
    l,
    u,
    record_count,
    tol,
    dual_eps,
)
    if blockIdx().x == 1 && threadIdx().x == 1
        for k in Int(record_count):-1:1
            reduction_type = @inbounds reduction_types[k]
            reduction_type == Int32(SUB_COL) || continue

            idx0 = Int(@inbounds idx_starts[k])
            idx1 = Int(@inbounds idx_starts[k + 1]) - 1
            val0 = Int(@inbounds val_starts[k])
            val1 = Int(@inbounds val_starts[k + 1]) - 1
            idx1 >= idx0 + 3 || continue
            val1 >= val0 + 6 || continue

            support_count = Int(@inbounds tape_idx[idx0 + 2])
            support_count == 1 || continue

            _device_refine_sub_col_dual_from_original_gpu!(
                x_org,
                y_org,
                z_org,
                ATy,
                A_rowPtr,
                A_colVal,
                A_nzVal,
                AT_rowPtr,
                c,
                l,
                u,
                Int(@inbounds tape_idx[idx0 + 1]),
                Int(@inbounds tape_idx[idx0]),
                Int(@inbounds tape_idx[idx0 + 3]),
                @inbounds(tape_val[val0]),
                @inbounds(tape_val[val0 + 6]),
                @inbounds(tape_val[val0 + 5]) != 0.0,
                tol,
                dual_eps,
            )
        end
    end
    return
end

function postsolve_refine_sub_cols_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return (x_org, y_org, z_org)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)

    dual_eps = min(tol, 1.0e-12)
    @cuda threads=1 blocks=1 _kernel_refine_sub_cols_from_tape_gpu!(
        x_org,
        y_org,
        z_org,
        ATy,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        original_model.A.rowPtr,
        original_model.A.colVal,
        original_model.A.nzVal,
        original_model.AT.rowPtr,
        original_model.c,
        original_model.l,
        original_model.u,
        Int32(record_count),
        tol,
        dual_eps,
    )

    return (x_org, y_org, z_org)
end

function postsolve_refine_singleton_chain_rows_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
    protected_cols::Union{Nothing,CuVector{UInt8}}=nothing,
)
    tape_gpu = ensure_postsolve_tape_gpu!(rec)
    record_count = postsolve_record_count(tape_gpu)
    record_count == 0 && return (x_org, y_org, z_org)

    ATy = similar(z_org)
    mul!(ATy, original_model.AT, y_org)
    dual_eps = min(tol, 1.0e-12)
    seen_rows = CUDA.zeros(UInt8, length(y_org))
    row_targets = CuVector{Int32}(undef, length(y_org))
    target_count_d = CUDA.zeros(Int32, 1)

    @cuda threads=1 blocks=1 _kernel_collect_live_row_refinement_targets_from_tape_gpu!(
        row_targets,
        target_count_d,
        seen_rows,
        tape_gpu.types,
        tape_gpu.index_starts,
        tape_gpu.value_starts,
        tape_gpu.indices,
        tape_gpu.vals,
        Int32(record_count),
        UInt8(1),
    )
    target_count = Int(Array(target_count_d)[1])
    target_count == 0 && return (x_org, y_org, z_org)

    for _ in 1:10
        postsolve_project_column_duals_from_ATy_gpu!(
            x_org,
            z_org,
            ATy,
            rec,
            original_model;
            tol=tol,
            protected_cols=protected_cols,
        )
        @cuda threads=1 blocks=1 _kernel_refine_live_rows_from_list_gpu!(
            row_targets,
            Int32(target_count),
            x_org,
            y_org,
            z_org,
            ATy,
            original_model.A.rowPtr,
            original_model.A.colVal,
            original_model.A.nzVal,
            original_model.c,
            original_model.AL,
            original_model.AU,
            original_model.l,
            original_model.u,
            tol,
            dual_eps,
        )
    end

    return (x_org, y_org, z_org)
end

function _kernel_compute_row_dual_updates_from_original_gpu!(
    y_new,
    x_org,
    y_old,
    ATy_old,
    A_rowPtr,
    A_colVal,
    A_nzVal,
    c,
    AL,
    AU,
    l,
    u,
    active_rows,
    m,
    tol,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        if @inbounds(active_rows[i] == UInt8(0))
            @inbounds y_new[i] = y_old[i]
            return
        end

        row_start = Int(@inbounds A_rowPtr[i])
        row_stop = Int(@inbounds A_rowPtr[i + 1]) - 1
        if row_start > row_stop
            @inbounds y_new[i] = y_old[i]
            return
        end

        y_old_i = @inbounds y_old[i]
        lo = -Inf
        hi = Inf
        exact_sum = 0.0
        exact_count = 0
        activity = 0.0

        for p in row_start:row_stop
            coeff = @inbounds A_nzVal[p]
            abs(coeff) > tol || continue

            col = Int(@inbounds A_colVal[p])
            xj = @inbounds x_org[col]
            lj = @inbounds l[col]
            uj = @inbounds u[col]
            activity += coeff * xj

            at_lower = isfinite(lj) && xj <= lj + tol
            at_upper = isfinite(uj) && xj >= uj - tol

            lo_col = -Inf
            hi_col = Inf
            if !(at_lower && at_upper)
                s = @inbounds c[col] - (@inbounds(ATy_old[col]) - coeff * y_old_i)
                thresh = s / coeff
                if !at_lower && !at_upper
                    lo_col = thresh
                    hi_col = thresh
                    exact_sum += thresh
                    exact_count += 1
                elseif at_lower
                    if coeff > 0.0
                        hi_col = thresh
                    else
                        lo_col = thresh
                    end
                else
                    if coeff > 0.0
                        lo_col = thresh
                    else
                        hi_col = thresh
                    end
                end
            end

            lo = max(lo, lo_col)
            hi = min(hi, hi_col)
        end

        row_l = @inbounds AL[i]
        row_u = @inbounds AU[i]
        row_at_lower = isfinite(row_l) && activity <= row_l + tol
        row_at_upper = isfinite(row_u) && activity >= row_u - tol
        row_exact = !row_at_lower && !row_at_upper

        row_lo = if row_at_lower && row_at_upper
            -Inf
        elseif row_exact
            0.0
        elseif row_at_lower
            0.0
        else
            -Inf
        end
        row_hi = if row_at_lower && row_at_upper
            Inf
        elseif row_exact
            0.0
        elseif row_at_lower
            Inf
        else
            0.0
        end

        lo = max(lo, row_lo)
        hi = min(hi, row_hi)

        guess = row_exact ? 0.0 : y_old_i
        y_new_i = if row_exact
            0.0
        elseif exact_count > 0
            target = exact_sum / exact_count
            if lo <= hi + tol
                _clamp_with_inf_gpu(target, lo, hi)
            else
                _clamp_with_inf_gpu(target, row_lo, row_hi)
            end
        elseif lo <= hi + tol
            _clamp_with_inf_gpu(guess, lo, hi)
        else
            _clamp_with_inf_gpu(guess, row_lo, row_hi)
        end

        @inbounds y_new[i] = y_new_i
    end
    return
end

function postsolve_refine_duals_globally_from_original_gpu!(
    x_org::CuVector{Float64},
    y_org::CuVector{Float64},
    z_org::CuVector{Float64},
    original_model::LP_info_gpu;
    tol::Float64=1e-7,
    max_iters::Int=10,
    active_rows::Union{Nothing, CuVector{UInt8}}=nothing,
    rec::Union{Nothing,PresolveRecord_gpu}=nothing,
    protected_cols::Union{Nothing,CuVector{UInt8}}=nothing,
)
    m = length(y_org)
    n = length(z_org)
    (m == 0 || n == 0 || max_iters <= 0) && return (x_org, y_org, z_org)

    @assert length(x_org) == length(original_model.c)
    @assert length(z_org) == length(original_model.c)
    @assert length(y_org) == length(original_model.AL)
    active_rows_d = active_rows === nothing ? CUDA.fill(UInt8(1), m) : active_rows
    @assert length(active_rows_d) == m

    ATy = similar(z_org)
    y_old = similar(y_org)
    y_new = similar(y_org)

    mul!(ATy, original_model.AT, y_org)
    postsolve_project_column_duals_from_ATy_gpu!(
        x_org,
        z_org,
        ATy,
        rec,
        original_model;
        tol=tol,
        protected_cols=protected_cols,
    )

    blocks = cld(m, GPU_PRESOLVE_THREADS)
    for _ in 1:max_iters
        copyto!(y_old, y_org)
        @cuda threads=GPU_PRESOLVE_THREADS blocks=blocks _kernel_compute_row_dual_updates_from_original_gpu!(
            y_new,
            x_org,
            y_old,
            ATy,
            original_model.A.rowPtr,
            original_model.A.colVal,
            original_model.A.nzVal,
            original_model.c,
            original_model.AL,
            original_model.AU,
            original_model.l,
            original_model.u,
            active_rows_d,
            Int32(m),
            tol,
        )
        copyto!(y_org, y_new)
        mul!(ATy, original_model.AT, y_org)
        postsolve_project_column_duals_from_ATy_gpu!(
            x_org,
            z_org,
            ATy,
            rec,
            original_model;
            tol=tol,
            protected_cols=protected_cols,
        )
    end

    return (x_org, y_org, z_org)
end

function _postsolve_row_dual_interval(
    lower::Float64,
    upper::Float64,
    activity::Float64,
    tol::Float64,
)
    at_lower = _postsolve_at_lower(activity, lower, tol)
    at_upper = _postsolve_at_upper(activity, upper, tol)

    if at_lower && at_upper
        return (-Inf, Inf)
    elseif !at_lower && !at_upper
        return (0.0, 0.0)
    elseif at_lower
        return (0.0, Inf)
    else
        return (-Inf, 0.0)
    end
end

function _refine_fixed_column_duals_from_original!(
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_cpu,
)
    isempty(rec.fixed_idx) && return z_org

    ATy = original_model.AT * y_org
    for col in _copy_vector_to_host(rec.fixed_idx)
        j = Int(col)
        z_org[j] = original_model.c[j] - ATy[j]
    end
    return z_org
end

function _refine_fixed_column_duals_from_ATy!(
    z_org::Vector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_cpu,
    ATy::Vector{Float64},
)
    isempty(rec.fixed_idx) && return z_org

    for col in _copy_vector_to_host(rec.fixed_idx)
        j = Int(col)
        z_org[j] = original_model.c[j] - ATy[j]
    end
    return z_org
end

function _update_ATy_from_row_delta!(
    ATy::Vector{Float64},
    original_model::LP_info_cpu,
    row::Int,
    delta::Float64;
    tol::Float64=1e-7,
)
    abs(delta) > tol || return ATy
    1 <= row <= size(original_model.A, 1) || return ATy

    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)
    for p in nzrange(AT, row)
        col = rowvals_AT[p]
        coeff = nzvals_AT[p]
        abs(coeff) > tol || continue
        ATy[col] += coeff * delta
    end

    return ATy
end

function _refine_live_row_dual_from_original_cached!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    ATy::Vector{Float64},
    row_activity::AbstractVector{<:Real},
    original_model::LP_info_cpu,
    row::Int;
    tol::Float64=1e-7,
)
    1 <= row <= length(y_org) || return (y_org, z_org)

    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)
    y_old = y_org[row]

    activity = Float64(row_activity[row])
    lo = -Inf
    hi = Inf
    exact_targets = Float64[]
    cols = Int[]
    coeffs = Float64[]

    for p in nzrange(AT, row)
        col = rowvals_AT[p]
        coeff = nzvals_AT[p]
        abs(coeff) > tol || continue

        push!(cols, col)
        push!(coeffs, coeff)

        s = original_model.c[col] - (ATy[col] - coeff * y_old)

        lo_col, hi_col = _postsolve_dual_interval(
            s,
            coeff,
            x_org[col],
            original_model.l[col],
            original_model.u[col],
            tol,
        )
        lo = max(lo, lo_col)
        hi = min(hi, hi_col)

        col_lower = _postsolve_at_lower(x_org[col], original_model.l[col], tol)
        col_upper = _postsolve_at_upper(x_org[col], original_model.u[col], tol)
        if !col_lower && !col_upper
            push!(exact_targets, s / coeff)
        end
    end

    isempty(cols) && return (y_org, z_org)

    lo_row, hi_row = _postsolve_row_dual_interval(
        original_model.AL[row],
        original_model.AU[row],
        activity,
        tol,
    )
    lo = max(lo, lo_row)
    hi = min(hi, hi_row)

    row_lower = _postsolve_at_lower(activity, original_model.AL[row], tol)
    row_upper = _postsolve_at_upper(activity, original_model.AU[row], tol)
    row_exact = !row_lower && !row_upper
    if row_exact
        push!(exact_targets, 0.0)
    end

    guess = row_exact ? 0.0 : y_old
    y_new = if row_exact
        0.0
    elseif !isempty(exact_targets)
        target = sum(exact_targets) / length(exact_targets)
        if lo <= hi + tol
            _clamp_with_inf(target, lo, hi)
        else
            _clamp_with_inf(target, min(lo, hi), max(lo, hi))
        end
    elseif lo <= hi + tol
        _clamp_with_inf(guess, lo, hi)
    else
        _clamp_with_inf(guess, min(lo, hi), max(lo, hi))
    end

    y_org[row] = y_new
    delta = y_new - y_old
    if abs(delta) > tol
        for t in eachindex(cols)
            col = cols[t]
            ATy[col] += coeffs[t] * delta
        end
    end

    for col in cols
        z_org[col] = original_model.c[col] - ATy[col]
    end

    return (y_org, z_org)
end

function _refine_sub_col_dual_from_original!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    ATy::Vector{Float64},
    row_activity::AbstractVector{<:Real},
    original_model::LP_info_cpu,
    row::Int,
    cols::Vector{Int},
    coeffs::Vector{Float64};
    row_deleted::Bool=false,
    tol::Float64=1e-7,
)
    1 <= row <= length(y_org) || return (y_org, z_org)
    length(cols) == length(coeffs) || return (y_org, z_org)

    length(cols) == 2 || return (y_org, z_org)

    elim_col = cols[1]
    keep_col = cols[2]
    pivot_coeff = coeffs[1]
    keep_coeff = coeffs[2]
    abs(pivot_coeff) > tol || return (y_org, z_org)
    abs(keep_coeff) > tol || return (y_org, z_org)

    elim_nnz = Int(original_model.A.colptr[elim_col + 1] - original_model.A.colptr[elim_col])
    is_doubleton_eq = row_deleted && elim_nnz > 1
    is_doubleton_eq || return (y_org, z_org)

    y_old = y_org[row]
    lo = -Inf
    hi = Inf
    exact_targets = Float64[]

    for t in eachindex(cols)
        col = cols[t]
        coeff = coeffs[t]
        s = original_model.c[col] - (ATy[col] - coeff * y_old)
        lo_col, hi_col = _postsolve_dual_interval(
            s,
            coeff,
            x_org[col],
            original_model.l[col],
            original_model.u[col],
            tol,
        )
        lo = max(lo, lo_col)
        hi = min(hi, hi_col)

        col_lower = _postsolve_at_lower(x_org[col], original_model.l[col], tol)
        col_upper = _postsolve_at_upper(x_org[col], original_model.u[col], tol)
        if !col_lower && !col_upper
            push!(exact_targets, s / coeff)
        end
    end

    guess = y_old
    y_new = if !isempty(exact_targets)
        target = sum(exact_targets) / length(exact_targets)
        if lo <= hi + tol
            _clamp_with_inf(target, lo, hi)
        else
            _clamp_with_inf(target, min(lo, hi), max(lo, hi))
        end
    elseif lo <= hi + tol
        _clamp_with_inf(guess, lo, hi)
    else
        _clamp_with_inf(guess, min(lo, hi), max(lo, hi))
    end

    y_org[row] = y_new
    _update_ATy_from_row_delta!(ATy, original_model, row, y_new - y_old; tol=tol)
    _project_single_column_dual_from_ATy!(x_org, z_org, ATy, original_model, elim_col; tol=tol)
    _project_single_column_dual_from_ATy!(x_org, z_org, ATy, original_model, keep_col; tol=tol)

    return (y_org, z_org)
end

function _refine_live_row_dual_from_original!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    original_model::LP_info_cpu,
    row::Int;
    tol::Float64=1e-7,
)
    1 <= row <= length(y_org) || return (y_org, z_org)
    ATy = original_model.AT * y_org
    row_activity = original_model.A * x_org
    return _refine_live_row_dual_from_original_cached!(
        x_org,
        y_org,
        z_org,
        ATy,
        row_activity,
        original_model,
        row;
        tol=tol,
    )
end

function _project_column_duals_from_ATy!(
    x_org::Vector{Float64},
    z_org::Vector{Float64},
    ATy::Vector{Float64},
    original_model::LP_info_cpu;
    tol::Float64=1e-7,
)
    for col in eachindex(z_org)
        _project_single_column_dual_from_ATy!(
            x_org,
            z_org,
            ATy,
            original_model,
            col;
            tol=tol,
        )
    end
    return z_org
end

function _project_single_column_dual_from_ATy!(
    x_org::Vector{Float64},
    z_org::Vector{Float64},
    ATy::Vector{Float64},
    original_model::LP_info_cpu,
    col::Int;
    tol::Float64=1e-7,
)
    1 <= col <= length(z_org) || return z_org

    r = original_model.c[col] - ATy[col]
    at_lower = _postsolve_at_lower(x_org[col], original_model.l[col], tol)
    at_upper = _postsolve_at_upper(x_org[col], original_model.u[col], tol)

    z_org[col] = if at_lower && at_upper
        r
    elseif !at_lower && !at_upper
        0.0
    elseif at_lower
        max(r, 0.0)
    else
        min(r, 0.0)
    end

    return z_org
end

function _project_column_duals_from_original!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    original_model::LP_info_cpu;
    tol::Float64=1e-7,
)
    ATy = original_model.AT * y_org
    return _project_column_duals_from_ATy!(
        x_org,
        z_org,
        ATy,
        original_model;
        tol=tol,
    )
end

function _bound_change_row_components(
    original_model::LP_info_cpu,
    affected_rows::Vector{Int};
    tol::Float64=1e-7,
)
    isempty(affected_rows) && return Tuple{Vector{Int}, Vector{Int}}[]

    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)

    rows = unique(affected_rows)
    row_support_cols = Vector{Vector{Int}}(undef, length(rows))
    col_to_row_ids = Dict{Int, Vector{Int}}()

    for (rid, row) in enumerate(rows)
        cols = Int[]
        for p in nzrange(AT, row)
            col = rowvals_AT[p]
            abs(nzvals_AT[p]) > tol || continue
            push!(cols, col)
            push!(get!(col_to_row_ids, col, Int[]), rid)
        end
        row_support_cols[rid] = cols
    end

    components = Tuple{Vector{Int}, Vector{Int}}[]
    visited_rows = falses(length(rows))
    queued_cols = Set{Int}()
    row_queue = Int[]

    for rid in eachindex(rows)
        visited_rows[rid] && continue

        empty!(row_queue)
        empty!(queued_cols)
        push!(row_queue, rid)
        visited_rows[rid] = true

        comp_rows = Int[]
        comp_cols = Int[]

        while !isempty(row_queue)
            cur = pop!(row_queue)
            push!(comp_rows, rows[cur])

            for col in row_support_cols[cur]
                if col in queued_cols
                    continue
                end
                push!(queued_cols, col)
                push!(comp_cols, col)
                for nbr in col_to_row_ids[col]
                    if !visited_rows[nbr]
                        visited_rows[nbr] = true
                        push!(row_queue, nbr)
                    end
                end
            end
        end

        push!(components, (comp_rows, comp_cols))
    end

    return components
end

function _postsolve_row_support_cols(
    original_model::LP_info_cpu,
    row::Int;
    tol::Float64=1e-7,
)
    cols = Int[]
    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)
    for p in nzrange(AT, row)
        col = rowvals_AT[p]
        abs(nzvals_AT[p]) > tol || continue
        push!(cols, col)
    end
    return cols
end

function _expanded_row_closure_components(
    original_model::LP_info_cpu,
    seed_rows::Vector{Int};
    tol::Float64=1e-7,
)
    isempty(seed_rows) && return Tuple{Vector{Int}, Vector{Int}}[]

    A = original_model.A
    rowvals_A = rowvals(A)
    nzvals_A = nonzeros(A)

    components = Tuple{Vector{Int}, Vector{Int}}[]
    for (rows, support_cols) in _bound_change_row_components(
        original_model,
        unique(seed_rows);
        tol=tol,
    )
        comp_rows = copy(rows)
        row_set = Set(comp_rows)
        frontier_rows = copy(rows)

        for _ in 1:2
            next_frontier = Int[]
            frontier_seen = Set{Int}()
            frontier_cols = Set{Int}()

            for row in frontier_rows
                for col in _postsolve_row_support_cols(original_model, row; tol=tol)
                    push!(frontier_cols, col)
                end
            end

            for col in frontier_cols
                for q in nzrange(A, col)
                    nbr_row = rowvals_A[q]
                    abs(nzvals_A[q]) > tol || continue
                    if !(nbr_row in row_set)
                        push!(row_set, nbr_row)
                        push!(comp_rows, nbr_row)
                        if !(nbr_row in frontier_seen)
                            push!(frontier_seen, nbr_row)
                            push!(next_frontier, nbr_row)
                        end
                    end
                end
            end

            isempty(next_frontier) && break
            frontier_rows = next_frontier
        end

        push!(components, (comp_rows, support_cols))
    end

    return components
end

function _dense_bound_change_component_matrix(
    original_model::LP_info_cpu,
    rows::Vector{Int},
    support_cols::Vector{Int};
    tol::Float64=1e-7,
)
    M = zeros(Float64, length(support_cols), length(rows))
    isempty(rows) && return M
    isempty(support_cols) && return M

    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)
    col_pos = Dict{Int, Int}(col => pos for (pos, col) in enumerate(support_cols))

    for (j, row) in enumerate(rows)
        for p in nzrange(AT, row)
            col = rowvals_AT[p]
            pos = get(col_pos, col, 0)
            pos == 0 && continue
            coeff = nzvals_AT[p]
            abs(coeff) > tol || continue
            M[pos, j] = coeff
        end
    end

    return M
end

function _sparse_bound_change_component_matrix(
    original_model::LP_info_cpu,
    rows::Vector{Int},
    support_cols::Vector{Int};
    tol::Float64=1e-7,
)
    isempty(rows) && return spzeros(Float64, length(support_cols), 0)
    isempty(support_cols) && return spzeros(Float64, 0, length(rows))

    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)
    col_pos = Dict{Int, Int}(col => pos for (pos, col) in enumerate(support_cols))

    I = Int[]
    J = Int[]
    V = Float64[]

    for (j, row) in enumerate(rows)
        for p in nzrange(AT, row)
            col = rowvals_AT[p]
            pos = get(col_pos, col, 0)
            pos == 0 && continue
            coeff = nzvals_AT[p]
            abs(coeff) > tol || continue
            push!(I, pos)
            push!(J, j)
            push!(V, coeff)
        end
    end

    return sparse(I, J, V, length(support_cols), length(rows))
end

function _bound_change_component_row_dual_bounds(
    original_model::LP_info_cpu,
    x_org::Vector{Float64},
    rows::Vector{Int};
    tol::Float64=1e-7,
)
    lo = Vector{Float64}(undef, length(rows))
    hi = Vector{Float64}(undef, length(rows))
    AT = original_model.AT
    rowvals_AT = rowvals(AT)
    nzvals_AT = nonzeros(AT)

    for (t, row) in enumerate(rows)
        activity = 0.0
        for p in nzrange(AT, row)
            col = rowvals_AT[p]
            coeff = nzvals_AT[p]
            abs(coeff) > tol || continue
            activity += coeff * x_org[col]
        end
        lo[t], hi[t] = _postsolve_row_dual_interval(
            original_model.AL[row],
            original_model.AU[row],
            activity,
            tol,
        )
    end

    return lo, hi
end

function _solve_box_constrained_component_delta(
    M::AbstractMatrix{Float64},
    rhs::Vector{Float64},
    delta_lo::Vector{Float64},
    delta_hi::Vector{Float64};
    tol::Float64=1e-9,
    max_iters::Int=50,
)
    isempty(rhs) && return zeros(Float64, size(M, 2))

    delta = if size(M, 1) == 1 && size(M, 2) == 1
        coeff = M[1, 1]
        abs(coeff) > tol ? [rhs[1] / coeff] : [0.0]
    else
        M \ rhs
    end

    inside_box = true
    for i in eachindex(delta)
        if delta[i] < delta_lo[i] - tol || delta[i] > delta_hi[i] + tol
            inside_box = false
            break
        end
    end
    if inside_box
        residual = M * delta - rhs
        maximum(abs, residual) <= tol && return delta
    end

    for i in eachindex(delta)
        delta[i] = _clamp_with_inf(delta[i], delta_lo[i], delta_hi[i])
    end

    residual = M * delta - rhs
    maximum(abs, residual) <= tol && return delta

    for _ in 1:max_iters
        fixed = Int[]
        free = Int[]
        for i in eachindex(delta)
            if delta[i] <= delta_lo[i] + tol
                delta[i] = delta_lo[i]
                push!(fixed, i)
            elseif delta[i] >= delta_hi[i] - tol
                delta[i] = delta_hi[i]
                push!(fixed, i)
            else
                push!(free, i)
            end
        end

        isempty(free) && break

        rhs_free = copy(rhs)
        if !isempty(fixed)
            rhs_free .-= M[:, fixed] * delta[fixed]
        end

        M_free = M[:, free]
        delta_free = if size(M_free, 1) == 1 && size(M_free, 2) == 1
            coeff = M_free[1, 1]
            abs(coeff) > tol ? [rhs_free[1] / coeff] : [0.0]
        else
            M_free \ rhs_free
        end

        candidate = copy(delta)
        for (k, idx) in enumerate(free)
            candidate[idx] = _clamp_with_inf(delta_free[k], delta_lo[idx], delta_hi[idx])
        end

        change = maximum(abs.(candidate .- delta))
        delta = candidate
        change <= tol && break
    end

    return delta
end

function _estimate_component_lipschitz(
    M::SparseMatrixCSC{Float64, Int};
    iters::Int=8,
)
    n = size(M, 2)
    n == 0 && return 1.0

    x = ones(Float64, n)
    nx = norm(x)
    nx > 0.0 && (x ./= nx)

    L = 1.0
    for _ in 1:iters
        y = M * x
        z = transpose(M) * y
        nz = norm(z)
        nz > 0.0 || return 1.0
        x .= z ./ nz
        L = dot(x, transpose(M) * (M * x))
    end

    return max(L, 1.0e-12)
end

function _solve_box_constrained_component_delta_iterative(
    M::SparseMatrixCSC{Float64, Int},
    rhs::Vector{Float64},
    delta_lo::Vector{Float64},
    delta_hi::Vector{Float64};
    tol::Float64=1e-9,
    max_iters::Int=400,
)
    isempty(rhs) && return zeros(Float64, size(M, 2))

    delta = zeros(Float64, size(M, 2))
    best_delta = copy(delta)
    best_residual = Inf
    alpha = 1.0 / _estimate_component_lipschitz(M)

    for _ in 1:max_iters
        residual = M * delta - rhs
        residual_norm = maximum(abs, residual)
        if residual_norm < best_residual
            best_residual = residual_norm
            copyto!(best_delta, delta)
        end
        residual_norm <= tol && return delta

        grad = transpose(M) * residual
        change = 0.0
        for i in eachindex(delta)
            new_val = _clamp_with_inf(delta[i] - alpha * grad[i], delta_lo[i], delta_hi[i])
            change = max(change, abs(new_val - delta[i]))
            delta[i] = new_val
        end

        change <= tol && break
    end

    return best_delta
end

function _component_dual_statuses(
    x_org::Vector{Float64},
    original_model::LP_info_cpu,
    support_cols::Vector{Int};
    tol::Float64=1e-7,
)
    status = Vector{Int8}(undef, length(support_cols))
    for (t, col) in enumerate(support_cols)
        at_lower = _postsolve_at_lower(x_org[col], original_model.l[col], tol)
        at_upper = _postsolve_at_upper(x_org[col], original_model.u[col], tol)
        status[t] = if at_lower && !at_upper
            Int8(1)
        elseif at_upper && !at_lower
            Int8(-1)
        elseif !at_lower && !at_upper
            Int8(0)
        else
            Int8(2)
        end
    end
    return status
end

@inline function _component_dual_violation(raw::Float64, status::Int8)
    if status == Int8(0)
        return raw
    elseif status == Int8(1)
        return min(raw, 0.0)
    elseif status == Int8(-1)
        return max(raw, 0.0)
    else
        return 0.0
    end
end

function _refine_projected_component_delta!(
    delta::Vector{Float64},
    M::Matrix{Float64},
    residual::Vector{Float64},
    status::Vector{Int8},
    delta_lo::Vector{Float64},
    delta_hi::Vector{Float64};
    tol::Float64=1e-9,
    max_iters::Int=80,
    reg::Float64=1e-10,
)
    isempty(delta) && return delta

    L = opnorm(M)^2 + reg
    alpha = 1.0 / max(L, 1.0e-12)
    best_delta = copy(delta)
    best_obj = Inf
    raw = similar(residual)
    active = similar(residual)

    for _ in 1:max_iters
        mul!(raw, M, delta)
        raw .= residual .- raw

        obj = 0.0
        for i in eachindex(raw)
            active[i] = _component_dual_violation(raw[i], status[i])
            obj += active[i]^2
        end
        obj += reg * sum(abs2, delta)

        if obj < best_obj
            best_obj = obj
            copyto!(best_delta, delta)
        end

        sqrt(obj) <= tol && break

        grad = -(transpose(M) * active) .+ reg .* delta
        change = 0.0
        for i in eachindex(delta)
            new_val = _clamp_with_inf(delta[i] - alpha * grad[i], delta_lo[i], delta_hi[i])
            change = max(change, abs(new_val - delta[i]))
            delta[i] = new_val
        end

        change <= tol && break
    end

    copyto!(delta, best_delta)
    return delta
end

function _targeted_bound_change_dual_reconstruction_with_stats!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    original_model::LP_info_cpu,
    affected_rows::Vector{Int};
    tol::Float64=1e-7,
    max_rows::Int=128,
    max_dense_entries::Int=5_000_000,
    max_sparse_iters::Int=400,
)
    before = compute_original_kkt_metrics(original_model, x_org, y_org, z_org)
    isempty(affected_rows) && return (
        changed=false,
        accepted=false,
        total_components=0,
        used_components=0,
        max_component_rows=0,
        max_component_cols=0,
        max_rhs=0.0,
        before_target_p_feas=before[3],
        before_target_d_feas=before[4],
        before_target_gap=before[5],
        before_target_p_obj=before[1],
        before_target_d_obj=before[2],
        after_target_p_feas=before[3],
        after_target_d_feas=before[4],
        after_target_gap=before[5],
        after_target_p_obj=before[1],
        after_target_d_obj=before[2],
    )

    y_trial = copy(y_org)
    ATy_trial = original_model.AT * y_trial
    row_activity = original_model.A * x_org
    changed = false
    total_components = 0
    used_components = 0
    max_component_rows = 0
    max_component_cols = 0
    max_rhs = 0.0

    for (rows, support_cols) in _bound_change_row_components(
        original_model,
        affected_rows;
        tol=tol,
    )
        total_components += 1
        max_component_rows = max(max_component_rows, length(rows))
        max_component_cols = max(max_component_cols, length(support_cols))
        isempty(rows) && continue
        isempty(support_cols) && continue
        residual = original_model.c[support_cols] .- ATy_trial[support_cols]
        target = copy(residual)
        for (t, col) in enumerate(support_cols)
            at_lower = _postsolve_at_lower(x_org[col], original_model.l[col], tol)
            at_upper = _postsolve_at_upper(x_org[col], original_model.u[col], tol)
            if !at_lower && !at_upper
                target[t] = 0.0
            elseif at_lower && !at_upper
                target[t] = max(target[t], 0.0)
            elseif at_upper && !at_lower
                target[t] = min(target[t], 0.0)
            end
        end

        rhs = residual .- target
        rhs_norm = maximum(abs, rhs)
        max_rhs = max(max_rhs, rhs_norm)
        rhs_norm <= tol && continue

        row_lo, row_hi = _bound_change_component_row_dual_bounds(
            original_model,
            x_org,
            rows;
            tol=tol,
        )
        delta_lo = row_lo .- y_trial[rows]
        delta_hi = row_hi .- y_trial[rows]

        use_dense = length(rows) <= max_rows &&
                    length(rows) * length(support_cols) <= max_dense_entries
        if use_dense
            M = _dense_bound_change_component_matrix(
                original_model,
                rows,
                support_cols;
                tol=tol,
            )
            delta = _solve_box_constrained_component_delta(
                M,
                rhs,
                delta_lo,
                delta_hi;
                tol=max(tol, 1.0e-10),
            )
            all(isfinite, delta) || continue
            if length(rows) <= 16 && length(support_cols) <= 64
                status = _component_dual_statuses(
                    x_org,
                    original_model,
                    support_cols;
                    tol=tol,
                )
                _refine_projected_component_delta!(
                    delta,
                    M,
                    residual,
                    status,
                    delta_lo,
                    delta_hi;
                    tol=max(tol, 1.0e-10),
                )
            end
            y_trial[rows] .+= delta
            ATy_trial[support_cols] .+= M * delta
        else
            M_sparse = _sparse_bound_change_component_matrix(
                original_model,
                rows,
                support_cols;
                tol=tol,
            )
            delta = _solve_box_constrained_component_delta(
                M_sparse,
                rhs,
                delta_lo,
                delta_hi;
                tol=max(tol, 1.0e-10),
            )
            all(isfinite, delta) || continue
            y_trial[rows] .+= delta
            ATy_trial[support_cols] .+= M_sparse * delta
        end
        changed = true
        used_components += 1
    end

    z_trial = copy(z_org)
    after = if changed
        affected_rows_unique = unique(affected_rows)
        for _ in 1:20
            _project_column_duals_from_ATy!(
                x_org,
                z_trial,
                ATy_trial,
                original_model;
                tol=tol,
            )
            for row in affected_rows_unique
                _refine_live_row_dual_from_original_cached!(
                    x_org,
                    y_trial,
                    z_trial,
                    ATy_trial,
                    row_activity,
                    original_model,
                    row;
                    tol=tol,
                )
            end
        end
        _project_column_duals_from_ATy!(
            x_org,
            z_trial,
            ATy_trial,
            original_model;
            tol=tol,
        )
        compute_original_kkt_metrics(original_model, x_org, y_trial, z_trial)
    else
        before
    end

    accepted = changed && after[4] <= before[4] + 1e-10 && after[5] < before[5] - 1e-10
    if accepted
        copyto!(y_org, y_trial)
        copyto!(z_org, z_trial)
    end

    return (
        changed=changed,
        accepted=accepted,
        total_components=total_components,
        used_components=used_components,
        max_component_rows=max_component_rows,
        max_component_cols=max_component_cols,
        max_rhs=max_rhs,
        before_target_p_feas=before[3],
        before_target_d_feas=before[4],
        before_target_gap=before[5],
        before_target_p_obj=before[1],
        before_target_d_obj=before[2],
        after_target_p_feas=after[3],
        after_target_d_feas=after[4],
        after_target_gap=after[5],
        after_target_p_obj=after[1],
        after_target_d_obj=after[2],
    )
end

function _targeted_bound_change_dual_reconstruction!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    original_model::LP_info_cpu,
    affected_rows::Vector{Int};
    tol::Float64=1e-7,
    max_rows::Int=128,
    max_dense_entries::Int=5_000_000,
    max_sparse_iters::Int=400,
)
    stats = _targeted_bound_change_dual_reconstruction_with_stats!(
        x_org,
        y_org,
        z_org,
        original_model,
        affected_rows;
        tol=tol,
        max_rows=max_rows,
        max_dense_entries=max_dense_entries,
        max_sparse_iters=max_sparse_iters,
    )
    return stats.accepted
end

function postsolve_refine_duals_from_original!(
    x_org::Vector{Float64},
    y_org::Vector{Float64},
    z_org::Vector{Float64},
    rec::PresolveRecord_gpu,
    original_model::LP_info_cpu;
    tol::Float64=1e-7,
)
    ensure_postsolve_tape_cpu!(rec)
    record_count = postsolve_record_count(rec.tape)
    if record_count == 0 && isempty(rec.fixed_idx)
        return (x_org, y_org, z_org)
    end

    bound_change_rows = Int[]
    singleton_chain_records = Tuple{Int, Int, Int}[]
    ATy = original_model.AT * y_org
    row_activity = original_model.A * x_org

    for k in record_count:-1:1
        reduction_type = rec.tape.types[k]
        idx = postsolve_record_indices(rec.tape, k)
        vals = postsolve_record_values(rec.tape, k)

        if reduction_type == SUB_COL
            length(idx) >= 3 || continue
            length(vals) >= 6 || continue

            support_count = Int(idx[3])
            length(idx) >= 3 + support_count || continue
            length(vals) >= 6 + support_count || continue

            elim_col = Int(idx[1])
            row = Int(idx[2])
            row_deleted = vals[6] != 0.0

            if !row_deleted && support_count == 1
                push!(singleton_chain_records, (row, elim_col, Int(idx[4])))
            end

            cols = Int[elim_col]
            coeffs = Float64[vals[1]]
            for t in 1:support_count
                push!(cols, Int(idx[3 + t]))
                push!(coeffs, vals[6 + t])
            end

            _refine_sub_col_dual_from_original!(
                x_org,
                y_org,
                z_org,
                ATy,
                row_activity,
                original_model,
                row,
                cols,
                coeffs;
                row_deleted=row_deleted,
                tol=tol,
            )
        elseif reduction_type == BOUND_CHANGE_THE_ROW
            length(idx) >= 2 || continue
            row = Int(idx[2])
            push!(bound_change_rows, row)
            _refine_live_row_dual_from_original_cached!(
                x_org,
                y_org,
                z_org,
                ATy,
                row_activity,
                original_model,
                row;
                tol=tol,
            )
        elseif reduction_type == BOUND_CHANGE_NO_ROW
            length(idx) >= 1 || continue
            col = Int(idx[1])
            _project_single_column_dual_from_ATy!(
                x_org,
                z_org,
                ATy,
                original_model,
                col;
                tol=tol,
            )
        elseif reduction_type == LHS_CHANGE ||
               reduction_type == RHS_CHANGE
            length(idx) >= 1 || continue
            row = Int(idx[1])
            push!(bound_change_rows, row)
            _refine_live_row_dual_from_original_cached!(
                x_org,
                y_org,
                z_org,
                ATy,
                row_activity,
                original_model,
                row;
                tol=tol,
            )
        elseif reduction_type == DELETED_ROW
            length(idx) >= 2 || continue
            length(vals) >= 3 || continue

            row = Int(idx[1])
            col = Int(idx[2])
            row_lower = vals[1]
            row_upper = vals[2]
            coeff = vals[3]
            abs(coeff) > tol || continue

            s = z_org[col]
            xj = x_org[col]
            deleted_row_activity = coeff * xj

            col_lower = _postsolve_at_lower(xj, original_model.l[col], tol)
            col_upper = _postsolve_at_upper(xj, original_model.u[col], tol)
            col_exact = !col_lower && !col_upper
            row_lower_active = _postsolve_at_lower(deleted_row_activity, row_lower, tol)
            row_upper_active = _postsolve_at_upper(deleted_row_activity, row_upper, tol)
            row_exact = !row_lower_active && !row_upper_active

            lo_col, hi_col = _postsolve_dual_interval(
                s,
                coeff,
                xj,
                original_model.l[col],
                original_model.u[col],
                tol,
            )
            lo_row, hi_row = _postsolve_row_dual_interval(
                row_lower,
                row_upper,
                deleted_row_activity,
                tol,
            )

            lo = max(lo_col, lo_row)
            hi = min(hi_col, hi_row)

            y_old = y_org[row]
            y_new = if row_exact
                0.0
            elseif col_exact
                s / coeff
            elseif lo <= hi + tol
                _clamp_with_inf(0.0, lo, hi)
            else
                _clamp_with_inf(0.0, min(lo, hi), max(lo, hi))
            end

            y_org[row] = y_new
            _update_ATy_from_row_delta!(ATy, original_model, row, y_new - y_old; tol=tol)
            z_org[col] = original_model.c[col] - ATy[col]
        end
    end

    if !isempty(bound_change_rows)
        affected_rows = unique(bound_change_rows)
        for _ in 1:10
            _project_column_duals_from_ATy!(
                x_org,
                z_org,
                ATy,
                original_model;
                tol=tol,
            )
            for row in affected_rows
                _refine_live_row_dual_from_original_cached!(
                    x_org,
                    y_org,
                    z_org,
                    ATy,
                    row_activity,
                    original_model,
                    row;
                    tol=tol,
                )
            end
        end

        _targeted_bound_change_dual_reconstruction!(
            x_org,
            y_org,
            z_org,
            original_model,
            affected_rows;
            tol=tol,
        )

        ATy = original_model.AT * y_org
    end

    if !isempty(singleton_chain_records)
        problem_rows = Int[]
        for (row, elim_col, keep_col) in singleton_chain_records
            elim_violation = _column_bound_sign_violation(
                x_org[elim_col],
                z_org[elim_col],
                original_model.l[elim_col],
                original_model.u[elim_col],
                tol,
            )
            keep_violation = _column_bound_sign_violation(
                x_org[keep_col],
                z_org[keep_col],
                original_model.l[keep_col],
                original_model.u[keep_col],
                tol,
            )
            if max(elim_violation, keep_violation) > tol
                push!(problem_rows, row)
            end
        end

        if !isempty(problem_rows)
            chain_components = _expanded_row_closure_components(
                original_model,
                unique(problem_rows);
                tol=tol,
            )
            problem_row_set = Set(problem_rows)
            selected_rows = Int[]
            for (rows, _) in chain_components
                any(in(problem_row_set), rows) || continue
                append!(selected_rows, rows)
            end

            _targeted_bound_change_dual_reconstruction!(
                x_org,
                y_org,
                z_org,
                original_model,
                unique(selected_rows);
                tol=tol,
            )
            ATy = original_model.AT * y_org
        end
    end

    _refine_fixed_column_duals_from_ATy!(
        z_org,
        rec,
        original_model,
        ATy,
    )

    _cleanup_unbounded_dual_slacks!(
        z_org,
        original_model.l,
        original_model.u;
        tol=tol,
    )

    return (x_org, y_org, z_org)
end
