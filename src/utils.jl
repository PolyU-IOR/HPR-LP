# The function to read the LP problem from the file and formulate the LP problem
function formulation(A, c, AL, AU, l, u, obj_constant)
    # Validate dimensions: A is m x n, c/l/u have length n, AL/AU have length m
    m, n = size(A)
    @assert length(c) == n "Dimension mismatch: size(A, 2) = $(n), but length(c) = $(length(c))."
    @assert length(l) == n "Dimension mismatch: size(A, 2) = $(n), but length(l) = $(length(l))."
    @assert length(u) == n "Dimension mismatch: size(A, 2) = $(n), but length(u) = $(length(u))."
    @assert length(AL) == m "Dimension mismatch: size(A, 1) = $(m), but length(AL) = $(length(AL))."
    @assert length(AU) == m "Dimension mismatch: size(A, 1) = $(m), but length(AU) = $(length(AU))."

    standard_lp = LP_info_cpu(A, transpose(A), c, AL, AU, l, u, obj_constant)
    return standard_lp
end

const VALID_PRESOLVE_BACKENDS = ("GPU", "PSLP", "CUSTOM", "NONE")

function normalize_presolve_backend(backend)
    backend_name = if backend isa Bool
        backend ? "GPU" : "NONE"
    else
        uppercase(String(backend))
    end
    backend_name in VALID_PRESOLVE_BACKENDS || throw(ArgumentError(
        "Unsupported presolve backend $(backend). Expected one of GPU, PSLP, CUSTOM, NONE."))
    return backend_name
end

function set_presolve_backend!(params::HPRLP_parameters, backend)
    params.presolve = normalize_presolve_backend(backend)
    return params
end

presolve_enabled(params::HPRLP_parameters) = normalize_presolve_backend(params.presolve) != "NONE"

function run_presolve_with_fallback(
    runner::Function,
    backend_name::AbstractString,
    model,
    params::HPRLP_parameters,
    ; fallback=(model, nothing),
)
    try
        return runner()
    catch err
        @warn "$(backend_name) presolve failed; falling back to the original model." exception=(err, catch_backtrace())
        if backend_name == "GPU" && params.use_gpu && CUDA.functional()
            try
                CUDA.synchronize()
                CUDA.reclaim()
            catch reclaim_err
                @warn "GPU presolve fallback cleanup failed after presolve error." exception=(reclaim_err, catch_backtrace())
            end
        end
        return fallback
    end
end

_host_vector(v::AbstractVector) = collect(v)
_host_sparse(A) = SparseMatrixCSC(A)

function _lp_data_equal(lhs::Union{LP_info_cpu,LP_info_gpu}, rhs::Union{LP_info_cpu,LP_info_gpu})
    size(lhs.A) == size(rhs.A) || return false
    nnz(_host_sparse(lhs.A)) == nnz(_host_sparse(rhs.A)) || return false
    lhs.obj_constant == rhs.obj_constant || return false

    A_lhs = _host_sparse(lhs.A)
    A_rhs = _host_sparse(rhs.A)
    A_lhs == A_rhs || return false

    return _host_vector(lhs.c) == _host_vector(rhs.c) &&
           _host_vector(lhs.AL) == _host_vector(rhs.AL) &&
           _host_vector(lhs.AU) == _host_vector(rhs.AU) &&
           _host_vector(lhs.l) == _host_vector(rhs.l) &&
           _host_vector(lhs.u) == _host_vector(rhs.u)
end

function _drop_noop_presolve_state!(
    original_model::Union{LP_info_cpu,LP_info_gpu},
    reduced_model::Union{LP_info_cpu,LP_info_gpu},
    presolve_state,
    params::HPRLP_parameters,
    backend_name::AbstractString,
)
    if presolve_state !== nothing && _lp_data_equal(original_model, reduced_model)
        params.verbose && println("$(backend_name) presolve made no model changes; skipping postsolve state.")
        release_presolve_state!(presolve_state)
        return original_model, nothing
    end
    return reduced_model, presolve_state
end

function apply_gpu_presolve(model::LP_info_gpu, params::HPRLP_parameters; presolve_params=nothing)
    return run_presolve_with_fallback("GPU", model, params; fallback=(model, nothing, 0.0)) do
        if params.verbose
            println("GPU PRESOLVE ...")
        end
        reduced_model, presolve_state, presolve_time = run_external_gpu_presolve(
            model,
            params;
            presolve_params=presolve_params,
        )

        if reduced_model === nothing || presolve_state === nothing
            println("GPU presolve failed or returned nothing.")
            if presolve_state !== nothing
                free_external_gpu_presolve_state!(presolve_state)
            end
            return model, nothing, 0.0
        end

        if params.verbose
            println("GPU presolve reduced size: $(size(model.A)) -> $(size(reduced_model.A))")
            println("GPU presolve objective offset: $(reduced_model.obj_constant - model.obj_constant)")
        end

        reduced_model, presolve_state = _drop_noop_presolve_state!(
            model,
            reduced_model,
            presolve_state,
            params,
            "GPU",
        )
        return reduced_model, presolve_state, presolve_time
    end
end

function apply_pslp_presolve(model::LP_info_cpu, params::HPRLP_parameters)
    if !PSLP.is_available()
        params.verbose && println("PSLP dynamic library not found at $(PSLP.LIB_PATH). Skipping PSLP presolve.")
        return model, nothing, 0.0
    end

    return run_presolve_with_fallback("PSLP", model, params; fallback=(model, nothing, 0.0)) do
        if params.verbose
            println("PSLP PRESOLVE ...")
        end

        settings = PSLP.Settings(verbose=params.verbose)
        presolver_info, reduced_data = PSLP.load_and_run_presolve(
            model.c,
            model.A,
            model.l,
            model.u,
            model.AL,
            model.AU;
            settings=settings,
        )

        if reduced_data === nothing || presolver_info === nothing
            println("PSLP presolve failed or returned nothing.")
            if presolver_info !== nothing
                PSLP.free_presolver_wrapper(presolver_info)
            end
            return model, nothing, 0.0
        end

        c_red, A_red, l_red, u_red, lhs_red, rhs_red, obj_offset = reduced_data

        if params.verbose
            println("PSLP reduced size: $(size(model.A)) -> $(size(A_red))")
            println("PSLP objective offset: $(obj_offset)")
        end

        reduced_model = formulation(A_red, c_red, lhs_red, rhs_red, l_red, u_red, obj_offset)
        return reduced_model, presolver_info, PSLP.get_presolve_time(presolver_info)
    end
end

function apply_custom_presolve(model::Union{LP_info_cpu,LP_info_gpu}, params::HPRLP_parameters; presolve_params=nothing)
    return run_presolve_with_fallback("CUSTOM", model, params; fallback=(model, nothing, 0.0)) do
        if params.verbose
            println("CUSTOM PRESOLVE ...")
        end
        reduced_model, state = run_custom_presolve(model, params; presolve_params=presolve_params)
        if reduced_model === nothing
            if state !== nothing
                free_custom_presolve_state!(state)
            end
            println("Custom presolve returned `nothing` as reduced model. Skipping to original model.")
            return model, nothing, 0.0
        end
        return reduced_model, state, 0.0
    end
end

function apply_presolve(model::LP_info_cpu, params::HPRLP_parameters; presolve_params=nothing)
    backend = normalize_presolve_backend(params.presolve)
    if backend == "GPU"
        return run_presolve_with_fallback("GPU", model, params; fallback=(model, nothing, 0.0)) do
            if params.verbose
                println("GPU PRESOLVE ...")
            end
            reduced_model, presolve_state, presolve_time = run_external_gpu_presolve(
                model,
                params;
                presolve_params=presolve_params,
            )
            if reduced_model === nothing || presolve_state === nothing
                println("GPU presolve failed or returned nothing.")
                if presolve_state !== nothing
                    free_external_gpu_presolve_state!(presolve_state)
                end
                return model, nothing, 0.0
            end
            if params.verbose
                println("GPU presolve reduced size: $(size(model.A)) -> $(size(reduced_model.A))")
                println("GPU presolve objective offset: $(reduced_model.obj_constant - model.obj_constant)")
            end
            reduced_model, presolve_state = _drop_noop_presolve_state!(
                model,
                reduced_model,
                presolve_state,
                params,
                "GPU",
            )
            return reduced_model, presolve_state, presolve_time
        end
    elseif backend == "PSLP"
        return apply_pslp_presolve(model, params)
    elseif backend == "CUSTOM"
        return apply_custom_presolve(model, params; presolve_params=presolve_params)
    end
    return model, nothing, 0.0
end

function apply_presolve(model::LP_info_gpu, params::HPRLP_parameters; presolve_params=nothing)
    backend = normalize_presolve_backend(params.presolve)
    if backend == "GPU"
        return apply_gpu_presolve(model, params; presolve_params=presolve_params)
    elseif backend == "PSLP"
        throw(ArgumentError("PSLP presolve expects an LP_info_cpu model."))
    elseif backend == "CUSTOM"
        return apply_custom_presolve(model, params; presolve_params=presolve_params)
    end
    return model, nothing, 0.0
end
# Helper function to create scaling info and apply scaling to the LP problem
const CURTIS_REID_SCALING_ITERS = 20

function curtis_reid_scaling!(
    lp::LP_info_cpu,
    row_norm::Vector{Float64},
    col_norm::Vector{Float64},
    niters::Int,
)
    m, n = size(lp.A)
    colptr = lp.A.colptr
    rowvals_A = rowvals(lp.A)
    nzvals_A = nonzeros(lp.A)
    nz = length(nzvals_A)

    logvals = log.(max.(abs.(nzvals_A), 1.0e-300))
    col_idx = Vector{Int32}(undef, nz)
    for j in 1:n
        for k in colptr[j]:(colptr[j + 1] - 1)
            col_idx[k] = Int32(j)
        end
    end

    row_count = zeros(Int, m)
    col_count = zeros(Int, n)
    for k in 1:nz
        row_count[rowvals_A[k]] += 1
        col_count[Int(col_idx[k])] += 1
    end
    row_count[row_count .== 0] .= 1
    col_count[col_count .== 0] .= 1

    r = zeros(m)
    c = zeros(n)
    row_sum = zeros(m)
    col_sum = zeros(n)

    for _ in 1:niters
        fill!(row_sum, 0.0)
        for k in 1:nz
            row_sum[rowvals_A[k]] += -logvals[k] - c[Int(col_idx[k])]
        end
        r .= row_sum ./ row_count

        fill!(col_sum, 0.0)
        for k in 1:nz
            col_sum[Int(col_idx[k])] += -logvals[k] - r[rowvals_A[k]]
        end
        c .= col_sum ./ col_count
    end

    row_scale = clamp.(exp.(r), 1.0e-30, 1.0e30)
    col_scale = clamp.(exp.(c), 1.0e-30, 1.0e30)

    for j in 1:n
        col_s = col_scale[j]
        for k in colptr[j]:(colptr[j + 1] - 1)
            nzvals_A[k] *= row_scale[rowvals_A[k]] * col_s
        end
    end

    lp.AL .*= row_scale
    lp.AU .*= row_scale
    lp.c .*= col_scale
    lp.l ./= col_scale
    lp.u ./= col_scale

    # HPRLP unscales with division by row_norm/col_norm, so row/column multipliers are stored inverted.
    row_norm ./= row_scale
    col_norm ./= col_scale
    return row_scale, col_scale
end

function curtis_reid_scaling!(
    lp::LP_info_gpu,
    row_norm::CuVector{Float64},
    col_norm::CuVector{Float64},
    niters::Int,
)
    m, n = size(lp.A)
    row_log_scale = CUDA.zeros(Float64, m)
    col_log_scale = CUDA.zeros(Float64, n)
    row_scale = CUDA.ones(Float64, m)
    col_scale = CUDA.ones(Float64, n)

    for _ in 1:niters
        @cuda threads = 256 blocks = ceil(Int, m / 256) compute_curtis_reid_row_log_scale_kernel!(
            row_log_scale, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, col_log_scale, m
        )
        CUDA.synchronize()
        @cuda threads = 256 blocks = ceil(Int, n / 256) compute_curtis_reid_col_log_scale_kernel!(
            col_log_scale, lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, row_log_scale, n
        )
        CUDA.synchronize()
    end

    @cuda threads = 256 blocks = ceil(Int, m / 256) curtis_reid_log_to_scale_kernel!(row_scale, row_log_scale, m)
    @cuda threads = 256 blocks = ceil(Int, n / 256) curtis_reid_log_to_scale_kernel!(col_scale, col_log_scale, n)
    CUDA.synchronize()

    @cuda threads = 256 blocks = ceil(Int, m / 256) scale_curtis_reid_rows_cols_csr_kernel!(
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, row_scale, col_scale, m
    )
    @cuda threads = 256 blocks = ceil(Int, n / 256) scale_curtis_reid_cols_rows_csr_kernel!(
        lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, row_scale, col_scale, n
    )
    CUDA.synchronize()

    @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_mul_kernel!(lp.AL, row_scale, m)
    @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_mul_kernel!(lp.AU, row_scale, m)
    @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(lp.c, col_scale, n)
    @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(lp.l, col_scale, n)
    @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(lp.u, col_scale, n)
    CUDA.synchronize()

    @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(row_norm, row_scale, m)
    @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(col_norm, col_scale, n)
    CUDA.synchronize()
    return row_scale, col_scale
end

function scaling!(lp::LP_info_cpu, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    return scaling!(lp, true, use_Ruiz_scaling, use_Pock_Chambolle_scaling, use_bc_scaling)
end

function scaling!(lp::LP_info_cpu, use_Curtis_Reid_scaling::Bool, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    m, n = size(lp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    # Preallocate temporary arrays
    temp_norm1 = zeros(m)
    temp_norm2 = zeros(n)
    DA = spdiagm(temp_norm1)
    EA = spdiagm(temp_norm2)
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    norm_b_org = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    norm_c_org = 1 + norm(lp.c)
    scaling_info = Scaling_info_cpu(copy(lp.l), copy(lp.u), row_norm, col_norm, 1, 1, 1, 1, norm_b_org, norm_c_org)
    if use_Curtis_Reid_scaling
        curtis_reid_scaling!(lp, row_norm, col_norm, CURTIS_REID_SCALING_ITERS)
    end

    # Ruiz scaling
    if use_Ruiz_scaling
        for _ in 1:10
            temp_norm1 .= sqrt.(maximum(abs, lp.A, dims=2)[:, 1])
            temp_norm1[iszero.(temp_norm1)] .= 1.0
            row_norm .*= temp_norm1
            DA .= spdiagm(1.0 ./ temp_norm1)
            temp_norm2 .= sqrt.(maximum(abs, lp.A, dims=1)[1, :])
            temp_norm2[iszero.(temp_norm2)] .= 1.0
            col_norm .*= temp_norm2
            EA .= spdiagm(1.0 ./ temp_norm2)
            lp.AL ./= temp_norm1
            lp.AU ./= temp_norm1
            lp.A .= DA * lp.A * EA
            lp.c ./= temp_norm2
            lp.l .*= temp_norm2
            lp.u .*= temp_norm2
        end
    end

    # Pock-Chambolle scaling
    if use_Pock_Chambolle_scaling
        temp_norm1 .= sqrt.(sum(abs, lp.A, dims=2)[:, 1])
        temp_norm1[iszero.(temp_norm1)] .= 1.0
        row_norm .*= temp_norm1
        DA .= spdiagm(1.0 ./ temp_norm1)
        temp_norm2 .= sqrt.(sum(abs, lp.A, dims=1)[1, :])
        temp_norm2[iszero.(temp_norm2)] .= 1.0
        col_norm .*= temp_norm2
        EA .= spdiagm(1.0 ./ temp_norm2)
        lp.AL ./= temp_norm1
        lp.AU ./= temp_norm1
        lp.A .= DA * lp.A * EA
        lp.c ./= temp_norm2
        lp.l .*= temp_norm2
        lp.u .*= temp_norm2
    end

    # scaling for b and c
    if use_bc_scaling
        AL_nInf = copy(lp.AL)
        AU_nInf = copy(lp.AU)
        AL_nInf[lp.AL.==-Inf] .= 0.0
        AU_nInf[lp.AU.==Inf] .= 0.0
        b_scale = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + norm(lp.c)
        lp.AL ./= b_scale
        lp.AU ./= b_scale
        lp.c ./= c_scale
        lp.l ./= b_scale
        lp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(lp.c)
    lp.AT = transpose(lp.A)
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm

    lp.AL[lp.AL .== -Inf] .= -1.0e100
    lp.AU[lp.AU .== Inf] .= 1.0e100
    lp.l[lp.l .== -Inf] .= -1.0e100
    lp.u[lp.u .== Inf] .= 1.0e100
    return scaling_info
end

# GPU-based scaling function for the LP problem
function scaling_gpu!(lp::LP_info_gpu, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    return scaling_gpu!(lp, true, use_Ruiz_scaling, use_Pock_Chambolle_scaling, use_bc_scaling)
end

function scaling_gpu!(lp::LP_info_gpu, use_Curtis_Reid_scaling::Bool, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    m = size(lp.A, 1)
    n = size(lp.A, 2)

    # Initialize scaling vectors on GPU
    row_norm = CUDA.ones(Float64, m)
    col_norm = CUDA.ones(Float64, n)

    # Compute original norms for scaling info
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    norm_b_org = 1 + CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    norm_c_org = 1 + CUDA.norm(lp.c)

    # Initialize scaling info
    scaling_info = Scaling_info_gpu(
        copy(lp.l), copy(lp.u),
        row_norm, col_norm,
        1.0, 1.0, 1.0, 1.0,
        norm_b_org, norm_c_org
    )

    # Get CSR matrix components
    A_rowPtr = lp.A.rowPtr
    A_colVal = lp.A.colVal
    A_nzVal = lp.A.nzVal
    AT_rowPtr = lp.AT.rowPtr
    AT_colVal = lp.AT.colVal
    AT_nzVal = lp.AT.nzVal

    # Temporary vectors for scaling
    temp_row_norm = CUDA.ones(Float64, m)
    temp_col_norm = CUDA.ones(Float64, n)

    if use_Curtis_Reid_scaling
        curtis_reid_scaling!(lp, row_norm, col_norm, CURTIS_REID_SCALING_ITERS)
    end

    # Ruiz scaling
    if use_Ruiz_scaling
        for _ in 1:10
            # Compute row-wise max of |A|
            @cuda threads = 256 blocks = ceil(Int, m / 256) compute_row_max_abs_kernel!(
                A_rowPtr, A_nzVal, temp_row_norm, m
            )
            CUDA.synchronize()

            # Compute column-wise max of |A| (via AT)
            @cuda threads = 256 blocks = ceil(Int, n / 256) compute_col_max_abs_kernel!(
                AT_rowPtr, AT_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            # Update cumulative norms
            row_norm .*= temp_row_norm
            col_norm .*= temp_col_norm

            # Scale A: A = DA * A * EA (rows by temp_row_norm, cols by temp_col_norm)
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_rows_csr_kernel!(
                A_rowPtr, A_nzVal, temp_row_norm, m
            )
            CUDA.synchronize()

            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_csr_cols_kernel!(
                A_rowPtr, A_colVal, A_nzVal, temp_col_norm, m
            )
            CUDA.synchronize()

            # Scale AT: AT = EA * AT * DA (rows by temp_col_norm, cols by temp_row_norm)
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
                AT_rowPtr, AT_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
                AT_rowPtr, AT_colVal, AT_nzVal, temp_row_norm, n
            )
            CUDA.synchronize()

            # Scale constraint bounds
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                lp.AL, temp_row_norm, m
            )
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                lp.AU, temp_row_norm, m
            )
            CUDA.synchronize()

            # Scale objective and variable bounds
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(
                lp.c, temp_col_norm, n
            )
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
                lp.l, temp_col_norm, n
            )
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
                lp.u, temp_col_norm, n
            )
            CUDA.synchronize()
        end
    end

    # Pock-Chambolle scaling
    if use_Pock_Chambolle_scaling
        # Compute row-wise sum of |A|
        @cuda threads = 256 blocks = ceil(Int, m / 256) compute_row_sum_abs_kernel!(
            A_rowPtr, A_nzVal, temp_row_norm, m
        )
        CUDA.synchronize()

        # Compute column-wise sum of |A| (via AT)
        @cuda threads = 256 blocks = ceil(Int, n / 256) compute_col_sum_abs_kernel!(
            AT_rowPtr, AT_nzVal, temp_col_norm, n
        )
        CUDA.synchronize()

        # Update cumulative norms
        row_norm .*= temp_row_norm
        col_norm .*= temp_col_norm

        # Scale A: A = DA * A * EA (rows by temp_row_norm, cols by temp_col_norm)
        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_rows_csr_kernel!(
            A_rowPtr, A_nzVal, temp_row_norm, m
        )
        CUDA.synchronize()

        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_csr_cols_kernel!(
            A_rowPtr, A_colVal, A_nzVal, temp_col_norm, m
        )
        CUDA.synchronize()

        # Scale AT: AT = EA * AT * DA (rows by temp_col_norm, cols by temp_row_norm)
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
            AT_rowPtr, AT_nzVal, temp_col_norm, n
        )
        CUDA.synchronize()

        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
            AT_rowPtr, AT_colVal, AT_nzVal, temp_row_norm, n
        )
        CUDA.synchronize()

        # Scale constraint bounds
        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
            lp.AL, temp_row_norm, m
        )
        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
            lp.AU, temp_row_norm, m
        )
        CUDA.synchronize()

        # Scale objective and variable bounds
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(
            lp.c, temp_col_norm, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
            lp.l, temp_col_norm, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
            lp.u, temp_col_norm, n
        )
        CUDA.synchronize()
    end

    # b and c scaling
    if use_bc_scaling
        AL_nInf = copy(lp.AL)
        AU_nInf = copy(lp.AU)
        AL_nInf[lp.AL.==-Inf] .= 0.0
        AU_nInf[lp.AU.==Inf] .= 0.0
        b_scale = 1 + CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + CUDA.norm(lp.c)

        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_scalar_div_kernel!(
            lp.AL, b_scale, m
        )
        @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_scalar_div_kernel!(
            lp.AU, b_scale, m
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            lp.c, c_scale, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            lp.l, b_scale, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            lp.u, b_scale, n
        )
        CUDA.synchronize()

        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    # Compute final norms
    AL_nInf = copy(lp.AL)
    AU_nInf = copy(lp.AU)
    AL_nInf[lp.AL.==-Inf] .= 0.0
    AU_nInf[lp.AU.==Inf] .= 0.0
    scaling_info.norm_b = CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = CUDA.norm(lp.c)

    # Store the cumulative scaling norms
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm

    lp.AL[lp.AL .== -Inf] .= -1.0e100
    lp.AU[lp.AU .== Inf] .= 1.0e100
    lp.l[lp.l .== -Inf] .= -1.0e100
    lp.u[lp.u .== Inf] .= 1.0e100

    return scaling_info
end

function power_iteration_gpu(
    ws::HPRLP_workspace_gpu,
    max_iterations::Int=5000,
    tolerance::Float64=1e-4,
    check_every::Int=10;
)
    spmv_A = ws.spmv_A
    spmv_AT = ws.spmv_AT

    z = ws.Ax      # length m
    q = ws.y       # length m (must be restored)
    # ATq = ws.ATy  # length n (written via descriptor spmv_AT.desc_ATy)

    # Backup ws.y (allocate once per call unless you add a preallocated backup in ws)
    copyto!(ws.dy, ws.y)

    error = Inf
    lambda_max = 1.0

    # GPU RNG init (avoid CPU->GPU transfer)
    CUDA.seed!(1)
    CUDA.randn!(z)
    @. z = z + 1e-8

    for i in 1:max_iterations
        # Normalize: q = z / ||z||  (1 reduction + 1 broadcast kernel)
        z2 = CUDA.dot(z, z)
        invn = inv(sqrt(z2 + eps(Float64)))
        @. q = z * invn

        # z = A * (A' * q)
        CUDA.CUSPARSE.cusparseSpMV(
            spmv_AT.handle, spmv_AT.operator,
            spmv_AT.alpha, spmv_AT.desc_AT, spmv_AT.desc_y,
            spmv_AT.beta, spmv_AT.desc_ATy,
            spmv_AT.compute_type, spmv_AT.alg, spmv_AT.buf
        )
        CUDA.CUSPARSE.cusparseSpMV(
            spmv_A.handle, spmv_A.operator,
            spmv_A.alpha, spmv_A.desc_A, spmv_AT.desc_ATy,
            spmv_A.beta, spmv_A.desc_Ax,
            spmv_A.compute_type, spmv_A.alg, spmv_A.buf
        )

        if (i % check_every == 0)
            lambda_max = CUDA.dot(q, z)
            @. q = z - lambda_max * q
            error = CUDA.norm(q)
            if error < tolerance
                copyto!(ws.y, ws.dy)
                return lambda_max
            end
        end
    end

    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    
    copyto!(ws.y, ws.dy)
    return lambda_max
end

function power_iteration_cpu(A::SparseMatrixCSC, AT::SparseMatrixCSC,
    max_iterations::Int=5000, tolerance::Float64=1e-4, check_every::Int=10;)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    error = Inf
    lambda_max = 1.0
    for i in 1:max_iterations
        z2 = dot(z, z)
        invn = inv(sqrt(z2 + eps(Float64)))
        @. q = z * invn
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        if (i % check_every == 0)
            lambda_max = dot(q, z)
            @. q = z - lambda_max * q
            error = norm(q)
            if error < tolerance
                return lambda_max
            end
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

"""
    validate_gpu_parameters!(params::HPRLP_parameters)

Validates GPU-related parameters and adjusts settings if GPU is requested but not available.

# Arguments
- `params::HPRLP_parameters`: The solver parameters to validate

# Behavior
- If `use_gpu=true` but CUDA is not functional, sets `use_gpu=false` and warns user
- If `use_gpu=true` but device_number is invalid, sets `use_gpu=false` and warns user
- Validates that device_number is within valid range [0, num_devices-1]
"""
function validate_gpu_parameters!(params::HPRLP_parameters)
    if params.use_gpu
        # Check if CUDA is functional
        if !CUDA.functional()
            @warn "GPU requested but CUDA is not functional. Falling back to CPU execution."
            params.use_gpu = false
            return
        end

        # Check if device_number is valid
        num_devices = length(CUDA.devices())
        if params.device_number < 0 || params.device_number >= num_devices
            @warn "Invalid GPU device number $(params.device_number). Valid range is [0, $(num_devices-1)]. Falling back to CPU execution."
            params.use_gpu = false
            return
        end
    end
end

"""
    build_from_mps(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)

Build an LP model from an MPS file.

# Arguments
- `filename::AbstractString`: Path to the `.mps` or `.mps.gz` file
- `verbose::Bool`: Enable verbose output (default: true)
- `mpsformat::Symbol`: MPS format hint passed to `MPSReader` (`:auto`, `:fixed`, `:free`)

# Returns
- `LP_info_cpu`: LP model ready to be solved

# Example
```julia
using HPRLP

model = build_from_mps("problem.mps")
params = HPRLP_parameters()
result = optimize(model, params)
```

See also: [`build_from_Abc`](@ref), [`optimize`](@ref)
"""
function build_from_mps(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)
    t_start = time()
    if verbose
        println("READING FILE ... ", filename)
    end
    lp = Logging.with_logger(Logging.NullLogger()) do
        MPSReader.read_mps(filename; keep_names=false, mpsformat=mpsformat)
    end
    read_time = time() - t_start
    if verbose
        println(@sprintf("READING FILE time: %.2f seconds", read_time))
    end

    t_start = time()
    if verbose
        println("FORMULATING LP ...")
    end
    A = sparse(lp.arows, lp.acols, lp.avals, lp.nrow, lp.ncol)
    standard_lp = formulation(A, lp.c, lp.lcon, lp.ucon, lp.lvar, lp.uvar, lp.obj_constant)
    if verbose
        println(@sprintf("FORMULATING LP time: %.2f seconds", time() - t_start))
    end

    return standard_lp
end

"""
    build_from_Abc(A, c, AL, AU, l, u, obj_constant=0.0)

Build an LP model from matrix form.

# Arguments
- `A::Union{SparseMatrixCSC, Matrix}`: Constraint matrix (m × n). Dense matrices will be automatically converted to sparse format with a warning.
- `c::Vector{Float64}`: Objective coefficients (length n)
- `AL::Vector{Float64}`: Lower bounds for constraints Ax (length m)
- `AU::Vector{Float64}`: Upper bounds for constraints Ax (length m)
- `l::Vector{Float64}`: Lower bounds for variables x (length n)
- `u::Vector{Float64}`: Upper bounds for variables x (length n)
- `obj_constant::Float64`: Constant term in objective function (default: 0.0)

# Returns
- `LP_info_cpu`: LP model ready to be solved

# Example
```julia
using SparseArrays, HPRLP

A = sparse([1.0 2.0; 3.0 1.0])
c = [-3.0, -5.0]
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]

model = build_from_Abc(A, c, AL, AU, l, u)
params = HPRLP_parameters()
result = optimize(model, params)
```

See also: [`optimize`](@ref)
"""
function build_from_Abc(A::Union{SparseMatrixCSC, Matrix},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64=0.0)

    # Convert dense matrix to sparse if needed
    if A isa Matrix
        @warn "Dense matrix detected. Converting to sparse format. For better performance, please provide a SparseMatrixCSC."
        A_sparse = sparse(A)
    else
        A_sparse = A
    end

    # Create copies to avoid modifying the input
    A_copy = copy(A_sparse)
    c_copy = copy(c)
    AL_copy = copy(AL)
    AU_copy = copy(AU)
    l_copy = copy(l)
    u_copy = copy(u)

    # Build the LP model
    standard_lp = formulation(A_copy, c_copy, AL_copy, AU_copy, l_copy, u_copy, obj_constant)

    return standard_lp
end

function release_solve_memory!(params::HPRLP_parameters)
    GC.gc(true)
    if params.use_gpu
        CUDA.synchronize()
        CUDA.reclaim()
    end
    return nothing
end

function _safe_shifted_geomean(values)
    numeric = Float64[]
    for value in values
        if value isa Number
            value_f = Float64(value)
            if isfinite(value_f)
                push!(numeric, value_f)
            end
        end
    end
    isempty(numeric) && return NaN
    return exp(mean(log.(numeric .+ 10.0))) - 10.0
end

_dataset_time_value(value) = value isa Number ? Float64(value) : NaN

function _dataset_solve_time(raw_solve_time, time_limit)
    return min(_dataset_time_value(raw_solve_time), Float64(time_limit))
end

function _dataset_total_time(solve_time, presolve_time, folding_time)
    return _dataset_time_value(solve_time) +
           _dataset_time_value(presolve_time) +
           _dataset_time_value(folding_time)
end

function _dataset_result_columns()
    return [
        :name,
        :iter,
        :solve_time,
        :total_time,
        :presolve_time,
        :folding_time,
        :res,
        :primal_obj,
        :status,
        :iter_4,
        :time_4,
        :iter_6,
        :time_6,
        :iter_8,
        :time_8,
    ]
end

function _snapshot_toml_quote(value::AbstractString)
    escaped = replace(value, "\\" => "\\\\", "\"" => "\\\"")
    return "\"" * escaped * "\""
end

function _snapshot_presolve_value(value)
    if value isa Nothing
        return _snapshot_toml_quote("nothing")
    elseif value isa Symbol
        return _snapshot_toml_quote(String(value))
    elseif value isa AbstractVector
        return "[" * join(_snapshot_presolve_value.(value), ", ") * "]"
    elseif value isa Tuple
        return "[" * join(_snapshot_presolve_value.(collect(value)), ", ") * "]"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa AbstractFloat
        if isfinite(value)
            return string(Float64(value))
        elseif isnan(value)
            return _snapshot_toml_quote("NaN")
        elseif value > 0
            return _snapshot_toml_quote("Inf")
        else
            return _snapshot_toml_quote("-Inf")
        end
    elseif value isa Integer
        return string(Int(value))
    elseif value isa AbstractString
        return _snapshot_toml_quote(String(value))
    end
    return _snapshot_toml_quote(string(value))
end

function _write_snapshot_line(io::IO, key::AbstractString, value; indent::Int=0)
    prefix = repeat(" ", indent)
    println(io, prefix, key, " = ", _snapshot_presolve_value(value))
end

function _tiered_bootstrap_snapshot_value(presolve_params)
    if hasproperty(presolve_params, :enable_tiered_bootstrap)
        return getproperty(presolve_params, :enable_tiered_bootstrap)
    end
    return getproperty(presolve_params, :gpu_presolve_scheduler) == :tiered
end

function _write_dataset_presolve_snapshot(
    snapshot_path::AbstractString,
    params::HPRLP_parameters,
    presolve_params;
    source_config_path=nothing,
)
    source_label = isnothing(source_config_path) ? (
        isnothing(presolve_params) ? "package_defaults" : "explicit_presolve_params"
    ) : String(source_config_path)
    open(snapshot_path, "w") do io
        println(io, "[meta]")
        _write_snapshot_line(io, "locked_at", string(now()))
        _write_snapshot_line(io, "source", source_label)
        println(io)

        println(io, "[hprlp]")
        _write_snapshot_line(io, "time_limit", Float64(params.time_limit))
        _write_snapshot_line(io, "stoptol", Float64(params.stoptol))
        _write_snapshot_line(io, "device_number", Int(params.device_number))
        _write_snapshot_line(io, "use_gpu", params.use_gpu)
        _write_snapshot_line(io, "warm_up", params.warm_up)
        _write_snapshot_line(io, "presolve", String(params.presolve))
        _write_snapshot_line(io, "use_postsolve", params.use_postsolve)
        _write_snapshot_line(io, "folding", String(params.folding))
        _write_snapshot_line(io, "folding_tolerance", Float64(params.folding_tolerance))
        _write_snapshot_line(io, "verbose", params.verbose)
        println(io)

        println(io, "[runtime]")
        _write_snapshot_line(io, "backend", "GPU")
        _write_snapshot_line(io, "device_number", Int(params.device_number))
        _write_snapshot_line(io, "verbose", params.verbose)
        if !isnothing(presolve_params)
            _write_snapshot_line(io, "scheduler_mode", presolve_params.gpu_presolve_scheduler)
            _write_snapshot_line(io, "tiered_bootstrap", _tiered_bootstrap_snapshot_value(presolve_params))
        end
        println(io)

        if isnothing(presolve_params)
            println(io, "# GPUPresolver parameters were not materialized in HPRLP; runtime defaults may have been used.")
            return snapshot_path
        end

        println(io, "[problem]")
        _write_snapshot_line(io, "type", "LP")
        println(io)

        println(io, "[limits]")
        _write_snapshot_line(io, "max_presolve_iters", presolve_params.max_iters)
        _write_snapshot_line(io, "max_presolve_time", presolve_params.max_time)
        println(io)

        println(io, "[tolerances]")
        _write_snapshot_line(io, "feasibility", presolve_params.feasibility_tol)
        _write_snapshot_line(io, "bound", presolve_params.bound_tol)
        _write_snapshot_line(io, "zero", presolve_params.zero_tol)
        _write_snapshot_line(io, "postsolve_tol", presolve_params.postsolve_tol)
        println(io)

        println(io, "[rules]")
        _write_snapshot_line(io, "close_bounds", presolve_params.enable_close_bounds)
        _write_snapshot_line(io, "empty_rows", presolve_params.enable_empty_rows)
        _write_snapshot_line(io, "singleton_rows", presolve_params.enable_singleton_rows)
        _write_snapshot_line(io, "activity_checks", presolve_params.enable_activity_checks)
        _write_snapshot_line(io, "primal_propagation", presolve_params.enable_primal_propagation)
        _write_snapshot_line(io, "parallel_rows", presolve_params.enable_parallel_rows)
        _write_snapshot_line(io, "empty_cols", presolve_params.enable_empty_cols)
        _write_snapshot_line(io, "singleton_cols_eq", presolve_params.enable_singleton_cols_eq)
        _write_snapshot_line(io, "singleton_cols_dual_infer", presolve_params.enable_singleton_cols_dual_infer)
        _write_snapshot_line(io, "doubleton_eq", presolve_params.enable_doubleton_eq)
        _write_snapshot_line(io, "linear_eq_agg", presolve_params.enable_linear_eq_agg)
        _write_snapshot_line(io, "dual_fix", presolve_params.enable_dual_fix)
        _write_snapshot_line(io, "parallel_cols", presolve_params.enable_parallel_cols)
        _write_snapshot_line(io, "fme_projection", presolve_params.enable_fme_projection)
        _write_snapshot_line(io, "structural_l1_substitution", presolve_params.enable_structural_l1_substitution)
        _write_snapshot_line(io, "redundant_bounds", presolve_params.enable_redundant_bounds)
        println(io)

        println(io, "[structural_l1]")
        _write_snapshot_line(io, "pattern", presolve_params.structural_l1_pattern)
        _write_snapshot_line(io, "allow_main_flow_without_tape", presolve_params.structural_l1_allow_main_flow_without_tape)
        _write_snapshot_line(io, "gpu_only", presolve_params.structural_l1_gpu_only)
        _write_snapshot_line(io, "residual_bound_as_free_min", presolve_params.structural_l1_residual_bound_as_free_min)
        println(io)

        println(io, "[fme]")
        _write_snapshot_line(io, "zero_objective_only", presolve_params.fme_zero_objective_only)
        _write_snapshot_line(io, "pair_limit", presolve_params.fme_pair_limit)
        _write_snapshot_line(io, "nnz_ratio_limit", presolve_params.fme_nnz_ratio_limit)
        _write_snapshot_line(io, "nnz_abs_slack", presolve_params.fme_nnz_abs_slack)
        _write_snapshot_line(io, "max_elims_per_call", presolve_params.fme_max_elims_per_call)
        _write_snapshot_line(io, "allow_main_flow_without_tape", presolve_params.fme_allow_main_flow_without_tape)
        _write_snapshot_line(io, "include_variable_bounds", presolve_params.fme_include_variable_bounds)
        _write_snapshot_line(io, "use_simple_screen", presolve_params.fme_use_simple_screen)
        _write_snapshot_line(io, "simple_side_limit", presolve_params.fme_simple_side_limit)
        _write_snapshot_line(io, "verbose", presolve_params.verbose_fme)
        println(io)

        println(io, "[scheduling]")
        _write_snapshot_line(io, "row_rule_order", presolve_params.row_rule_order)
        _write_snapshot_line(io, "col_rule_order", presolve_params.col_rule_order)
        _write_snapshot_line(io, "tiered_bootstrap", _tiered_bootstrap_snapshot_value(presolve_params))
        _write_snapshot_line(io, "tiered_cleanup_max_rounds", presolve_params.tiered_cleanup_max_rounds)
        _write_snapshot_line(io, "tiered_light_continue_ratio", presolve_params.tiered_light_continue_ratio)
        _write_snapshot_line(io, "tiered_cycle_stop_ratio", presolve_params.tiered_cycle_stop_ratio)
        _write_snapshot_line(io, "tiered_max_light_streak", presolve_params.tiered_max_light_streak)
        _write_snapshot_line(io, "tiered_global_period", presolve_params.tiered_global_period)
        println(io)

        println(io, "[doubleton]")
        _write_snapshot_line(io, "max_fill_in_proxy", presolve_params.doubleton_eq_max_fill_in_proxy)
        _write_snapshot_line(io, "scan", presolve_params.doubleton_eq_scan)
        _write_snapshot_line(io, "min_selected_per_batch", presolve_params.doubleton_eq_min_selected_per_batch)
        _write_snapshot_line(io, "min_selected_ratio", presolve_params.doubleton_eq_min_selected_ratio)
        _write_snapshot_line(io, "max_batch_rounds", presolve_params.doubleton_eq_max_batch_rounds)
        _write_snapshot_line(io, "max_time", presolve_params.doubleton_eq_max_time)
        println(io)

        println(io, "[qp]")
        _write_snapshot_line(io, "linear_eq_agg_max_support", presolve_params.qp_linear_eq_agg_max_support)
        _write_snapshot_line(io, "doubleton_max_q_fill_abs", presolve_params.qp_doubleton_max_q_fill_abs)
        _write_snapshot_line(io, "doubleton_max_q_fill_ratio", presolve_params.qp_doubleton_max_q_fill_ratio)
        _write_snapshot_line(io, "singleton_max_support", presolve_params.qp_singleton_max_support)
        _write_snapshot_line(io, "singleton_max_q_fill_abs", presolve_params.qp_singleton_max_q_fill_abs)
        _write_snapshot_line(io, "singleton_max_q_fill_ratio", presolve_params.qp_singleton_max_q_fill_ratio)
        _write_snapshot_line(io, "singleton_cols_eq_require_qdiag_zero", presolve_params.qp_singleton_cols_eq_require_qdiag_zero)
        _write_snapshot_line(io, "doubleton_eq_require_qdiag_zero", presolve_params.qp_doubleton_eq_require_qdiag_zero)
        println(io)

        println(io, "[advanced]")
        _write_snapshot_line(io, "record_postsolve_tape", presolve_params.record_postsolve_tape)
        _write_snapshot_line(io, "record_postsolve_tape_cpu", presolve_params.record_postsolve_tape_cpu)
        _write_snapshot_line(io, "debug_checks", presolve_params.debug_checks)
        _write_snapshot_line(io, "trace_enabled", presolve_params.trace_enabled)
        _write_snapshot_line(io, "trace_path", presolve_params.trace_path)
        _write_snapshot_line(io, "primal_propagation_min_tighten_abs", presolve_params.primal_propagation_min_tighten_abs)
    end
    return snapshot_path
end

function _env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    return default
end

function _env_int(name::AbstractString, default::Int; min_value::Int, max_value::Int)
    parsed = tryparse(Int, strip(get(ENV, name, string(default))))
    value = isnothing(parsed) ? default : parsed
    return clamp(value, min_value, max_value)
end

function _env_float(name::AbstractString, default::Float64; min_value::Float64, max_value::Float64)
    parsed = tryparse(Float64, strip(get(ENV, name, string(default))))
    value = isnothing(parsed) ? default : parsed
    return clamp(value, min_value, max_value)
end

function _maybe_set_dataset_warmup_coverage_mps!(
    data_path::String,
    params::HPRLP_parameters,
)
    uppercase(strip(params.presolve)) == "GPU" || return nothing
    params.warm_up || return nothing

    if haskey(ENV, "GPUPRESOLVER_WARMUP_COVERAGE_MPS")
        return nothing
    end

    if !_env_bool("GPUPRESOLVER_WARMUP_DATASET_MPS_AUTO", true)
        return nothing
    end

    count = _env_int("GPUPRESOLVER_WARMUP_DATASET_MPS_COUNT", 3; min_value=0, max_value=16)
    count == 0 && return nothing
    max_mb = _env_float("GPUPRESOLVER_WARMUP_DATASET_MPS_MAX_MB", 64.0; min_value=0.0, max_value=10240.0)
    max_bytes = round(Int, max_mb * 1024.0 * 1024.0)

    candidates = Tuple{String,Int64}[]
    for file in readdir(data_path)
        occursin(r"\.mps(\.gz)?$"i, file) || continue
        path = joinpath(data_path, file)
        isfile(path) || continue
        size_bytes = filesize(path)
        size_bytes <= max_bytes || continue
        push!(candidates, (path, size_bytes))
    end

    if isempty(candidates)
        return nothing
    end

    sort!(candidates; by=x -> (x[2], x[1]))
    selected = first(candidates, min(count, length(candidates)))
    ENV["GPUPRESOLVER_WARMUP_COVERAGE_MPS"] = join(first.(selected), ",")
    return nothing
end

# the function to test the HPR-LP algorithm on a dataset
function run_dataset(data_path::String, result_path::String, params::HPRLP_parameters; presolve_params=nothing)
    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = joinpath(result_path, "HPRLP_result.csv")

    # redirect the output to a file
    log_path = joinpath(result_path, "HPRLP_log.txt")

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")
    _maybe_set_dataset_warmup_coverage_mps!(data_path, params)

    dataset_presolve_params = presolve_params
    dataset_presolve_config_path = nothing
    if isnothing(dataset_presolve_params) && uppercase(strip(params.presolve)) == "GPU"
        dataset_presolve_config_path = _default_gpu_presolve_config_path()
        if !isnothing(dataset_presolve_config_path)
            dataset_presolve_params = load_gpu_presolve_setup(dataset_presolve_config_path).presolve_params
        end
    end

    if uppercase(strip(params.presolve)) == "GPU"
        snapshot_path = joinpath(result_path, "presolve_params_used.toml")
        _write_dataset_presolve_snapshot(
            snapshot_path,
            params,
            dataset_presolve_params;
            source_config_path=dataset_presolve_config_path,
        )
    end

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        solve_timelist = if hasproperty(result_table, :solve_time)
            Vector{Any}(result_table.solve_time[1:end-2])
        elseif hasproperty(result_table, :alg_time)
            Vector{Any}(result_table.alg_time[1:end-2])
        else
            fill(NaN, length(namelist))
        end
        presolve_timelist = if hasproperty(result_table, :presolve_time)
            Vector{Any}(result_table.presolve_time[1:end-2])
        else
            fill(NaN, length(namelist))
        end
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time4list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        iter8list = Vector{Any}(result_table.iter_8[1:end-2])
        time8list = Vector{Any}(result_table.time_8[1:end-2])
        folding_time_list = hasproperty(result_table, :folding_time) ? Vector{Any}(result_table.folding_time[1:end-2]) : Vector{Any}(fill(missing, length(namelist)))
        total_timelist = if hasproperty(result_table, :total_time)
            Vector{Any}(result_table.total_time[1:end-2])
        else
            Any[_dataset_total_time(solve_timelist[i], presolve_timelist[i], folding_time_list[i]) for i in eachindex(namelist)]
        end
    else
        namelist = []
        iterlist = []
        solve_timelist = []
        total_timelist = []
        presolve_timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time4list = []
        iter6list = []
        time6list = []
        iter8list = []
        time8list = []
        folding_time_list = []
    end


    for i = 1:length(files)
        file = files[i]
        if file in namelist
            println("The result of problem exists: ", file)
        end
        if occursin(".mps", file) && !(file in namelist)
            FILE_NAME = joinpath(data_path, file)
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                model = nothing
                results = nothing
                try
                    println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                    println("Solving: ----------------------------------------------------------------------------------------------------------")
                    t_start_all = time()

                    # Build and solve the model
                    model = build_from_mps(FILE_NAME, params.verbose)
                    case_presolve_params = isnothing(dataset_presolve_params) ? nothing : deepcopy(dataset_presolve_params)
                    results = optimize(model, params, model; presolve_params=case_presolve_params)

                    all_time = time() - t_start_all
                    println("Solve complete ----------------------------------------------------------------------------------------------------------")


                    println("iter = ", results.iter,
                        @sprintf("  time = %3.2e", results.time),
                        @sprintf("  residual = %3.2e", results.residuals),
                        @sprintf("  primal_obj = %3.15e", results.primal_obj),
                    )

                    push!(namelist, file)
                    push!(iterlist, results.iter)
                    solve_time = _dataset_solve_time(results.time, params.time_limit)
                    total_time = _dataset_total_time(solve_time, results.presolve_time, results.folding_time)
                    push!(solve_timelist, solve_time)
                    push!(total_timelist, total_time)
                    push!(presolve_timelist, results.presolve_time)
                    push!(reslist, results.residuals)
                    push!(objlist, results.primal_obj)
                    push!(statuslist, results.status)
                    push!(iter4list, results.iter_4)
                    push!(time4list, min(results.time_4, params.time_limit))
                    push!(iter6list, results.iter_6)
                    push!(time6list, min(results.time_6, params.time_limit))
                    push!(iter8list, results.iter_8)
                    push!(time8list, min(results.time_8, params.time_limit))
                    push!(folding_time_list, results.folding_time)
                finally
                    model = nothing
                    results = nothing
                    release_solve_memory!(params)
                end
            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                solve_time=solve_timelist,
                total_time=total_timelist,
                presolve_time=presolve_timelist,
                folding_time=folding_time_list,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time4list,
                iter_6=iter6list,
                time_6=time6list,
                iter_8=iter8list,
                time_8=time8list,
            )
            select!(result_table, _dataset_result_columns())

            # compute shifted geometric means and append them in the last rows
            geomean_solve_time = _safe_shifted_geomean(solve_timelist)
            geomean_total_time = _safe_shifted_geomean(total_timelist)
            geomean_presolve_time = _safe_shifted_geomean(presolve_timelist)
            geomean_folding_time = _safe_shifted_geomean(folding_time_list)
            geomean_time_4 = _safe_shifted_geomean(time4list)
            geomean_time_6 = _safe_shifted_geomean(time6list)
            geomean_time_8 = _safe_shifted_geomean(time8list)
            geomean_iter = _safe_shifted_geomean(iterlist)
            geomean_iter_4 = _safe_shifted_geomean(iter4list)
            geomean_iter_6 = _safe_shifted_geomean(iter6list)
            geomean_iter_8 = _safe_shifted_geomean(iter8list)
            push!(result_table, ["SGM10", geomean_iter, geomean_solve_time, geomean_total_time, geomean_presolve_time, geomean_folding_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, geomean_iter_8, geomean_time_8])
            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, solve_timelist)
            solved_total = count(x -> x < params.time_limit, total_timelist)
            solved_4 = count(x -> x < params.time_limit, time4list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            solved_8 = count(x -> x < params.time_limit, time8list)
            push!(result_table, ["solved", "", solved, solved_total, "", "", "", "", "", solved_4, "", solved_6, "", solved_8, ""])

            CSV.write(csv_file, result_table)
        end
    end
    println("The solver has finished running the dataset, total ", length(files), " problems")

    close(io)
end
