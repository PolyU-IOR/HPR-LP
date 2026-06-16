# Public GPU folding entry point. It colors equivalent rows/columns, reduces the
# LP on device, then returns a CPU map for the solver/unfolding path.
function run_folding(
    original_model::LP_info_cpu;
    tolerance::Float64=1e-8,
    verbose::Bool=false,
    params::Union{Nothing,HPRLP_parameters}=nothing,
)
    if params === nothing
        throw(ArgumentError("Folding requires HPRLP_parameters with folding=\"GPU\"."))
    end
    if !_fold_requested(params)
        throw(ArgumentError("Folding requires params.folding == \"GPU\"."))
    end
    ready, reason = _device_ready(params)
    if !ready
        throw(ArgumentError("Folding unavailable: $(reason)."))
    end
    gpu_model = setup_gpu_model(original_model, params)
    return run_folding(
        gpu_model;
        tolerance=tolerance,
        verbose=verbose,
        params=params,
    )
end

function run_folding(
    original_model::LP_info_gpu;
    tolerance::Float64=1e-8,
    verbose::Bool=false,
    params::Union{Nothing,HPRLP_parameters}=nothing,
)
    if params === nothing
        throw(ArgumentError("Folding requires HPRLP_parameters with folding=\"GPU\"."))
    end
    if !_fold_requested(params)
        throw(ArgumentError("Folding requires params.folding == \"GPU\"."))
    end

    ready, reason = _device_ready(params)
    if !ready
        throw(ArgumentError("Folding unavailable: $(reason)."))
    end

    model = original_model
    device_map = nothing
    levels_applied = 0
    ctx = nothing

    ctx = _ensure_context!(ctx, model, params.device_number)
    num_row, num_col = size(model.A)
    row_color_id, col_color_id, num_row_color, num_col_color, refinement_ok = refine_color!(
        num_row,
        num_col,
        ctx.device_model::LP_info_gpu,
        ctx.workspace::FoldingWorkspace,
        tolerance,
        verbose,
    )
    CUDA.synchronize()
    if !refinement_ok
        verbose && println("Folding found no reducible structure.")
        return original_model, nothing
    end

    reduced_model = reduce_size(
        row_color_id,
        col_color_id,
        num_row_color,
        num_col_color,
        ctx.device_model::LP_info_gpu,
        tolerance,
    )
    if size(reduced_model.A) != size(model.A)
        device_map = _build_map(
            row_color_id,
            col_color_id,
            num_row_color,
            num_col_color,
        )
        model = reduced_model
        levels_applied += 1
    end

    if device_map === nothing || !folding_active(device_map)
        verbose && println("Folding found no reducible structure.")
        return original_model, nothing
    end

    if verbose
        println("Folding levels: ", levels_applied)
        println("Folding reduced size: $(size(original_model.A)) -> $(size(model.A))")
    end

    return model, _cpu_map(device_map)
end

# Initial vectors collapse by taking one representative value per color class.
# Refinement should only group entries that are equivalent for folding.
function fold_initial_vector(value::Union{Nothing,AbstractVector}, color_index::Vector{Int}, reduced_length::Int)
    value === nothing && return nothing
    folded = Vector{Float64}(undef, reduced_length)
    seen = falses(reduced_length)
    for (idx, color) in pairs(color_index)
        if !seen[color]
            folded[color] = Float64(value[idx])
            seen[color] = true
        end
    end
    return folded
end

# Primal values copy across column colors. Dual and slack values are scaled by
# inverse color size to match the averaged reduced bounds.
function unfold_solution(map::FoldingMap, x_red, y_red, z_red)
    x = Vector{Float64}(undef, map.original_ncol)
    z = Vector{Float64}(undef, map.original_ncol)
    y = Vector{Float64}(undef, map.original_nrow)
    for j in 1:map.original_ncol
        color = map.col_color_id[j]
        x[j] = x_red[color]
        z[j] = z_red[color] * map.col_scale[j]
    end
    for i in 1:map.original_nrow
        color = map.row_color_id[i]
        y[i] = y_red[color] * map.row_scale[i]
    end
    return x, y, z
end

function unfold_results!(
    results::HPRLP_results,
    original_model::LP_info_cpu,
    map::FoldingMap,
    params::HPRLP_parameters;
    log_kkt::Bool=true,
)
    x_red = results.x isa Vector{Float64} ? results.x : Vector(results.x)
    y_red = results.y isa Vector{Float64} ? results.y : Vector(results.y)
    z_red = results.z isa Vector{Float64} ? results.z : Vector(results.z)
    results.x, results.y, results.z = unfold_solution(map, x_red, y_red, z_red)

    results.original_nRows = map.original_nrow
    results.original_nCols = map.original_ncol
    if results.status == "OPTIMAL"
        p_obj, _, p_feas, d_feas, gap =
            compute_original_kkt_metrics(original_model, results.x, results.y, results.z)
        results.primal_obj = p_obj
        results.original_p_feas = p_feas
        results.original_d_feas = d_feas
        results.original_gap = gap
        if log_kkt && max(p_feas, d_feas, gap) > params.stoptol && params.verbose
            println("Warning: unfolded original KKT check failed")
            println("Primal Residual: ", @sprintf("%.6e", p_feas))
            println("Dual Residual: ", @sprintf("%.6e", d_feas))
            println("Relative Gap: ", @sprintf("%.6e", gap))
        end
    end
    return results
end

function unfold_results!(
    results::HPRLP_results,
    original_model::LP_info_gpu,
    map::FoldingMap,
    params::HPRLP_parameters;
    log_kkt::Bool=true,
)
    return unfold_results!(
        results,
        _cpu_validation_model(original_model),
        map,
        params;
        log_kkt=log_kkt,
    )
end
