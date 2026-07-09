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

const VALID_PRESOLVE_BACKENDS = ("GPU", "PSLP", "NONE")

function normalize_presolve_backend(backend)
    backend_name = if backend isa Bool
        backend ? "GPU" : "NONE"
    else
        uppercase(String(backend))
    end
    backend_name in VALID_PRESOLVE_BACKENDS || throw(ArgumentError(
        "Unsupported presolve backend $(backend). Expected one of GPU, PSLP, NONE."))
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
        return model, nothing
    end
end

function apply_gpu_presolve(model::LP_info_gpu, params::HPRLP_parameters; presolve_params=nothing)
    return run_presolve_with_fallback("GPU", model, params) do
        if params.verbose
            println("GPU PRESOLVE ...")
        end
        t_start = time()

        settings = GPUPresolve.Settings(
            verbose=params.verbose,
            device_number=params.device_number,
            presolve_params=presolve_params,
        )
        presolve_state, reduced_model = GPUPresolve.run_presolve(model; settings=settings)

        if params.verbose
            println(@sprintf("GPU PRESOLVE time: %.2f seconds", time() - t_start))
        end

        if reduced_model === nothing || presolve_state === nothing
            println("GPU presolve failed or returned nothing.")
            if presolve_state !== nothing
                GPUPresolve.free_presolve_state!(presolve_state)
            end
            return model, nothing
        end

        if params.verbose
            println("GPU presolve reduced size: $(size(model.A)) -> $(size(reduced_model.A))")
            println("GPU presolve objective offset: $(reduced_model.obj_constant - model.obj_constant)")
        end

        return reduced_model, presolve_state
    end
end

function apply_pslp_presolve(model::LP_info_cpu, params::HPRLP_parameters)
    if !PSLP.is_available()
        params.verbose && println("PSLP dynamic library not found at $(PSLP.LIB_PATH). Skipping PSLP presolve.")
        return model, nothing
    end

    return run_presolve_with_fallback("PSLP", model, params) do
        if params.verbose
            println("PSLP PRESOLVE ...")
        end
        t_start = time()

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

        if params.verbose
            println(@sprintf("PSLP PRESOLVE time: %.2f seconds", time() - t_start))
        end

        if reduced_data === nothing || presolver_info === nothing
            println("PSLP presolve failed or returned nothing.")
            if presolver_info !== nothing
                PSLP.free_presolver_wrapper(presolver_info)
            end
            return model, nothing
        end

        c_red, A_red, l_red, u_red, lhs_red, rhs_red, obj_offset = reduced_data

        if params.verbose
            println("PSLP reduced size: $(size(model.A)) -> $(size(A_red))")
            println("PSLP objective offset: $(obj_offset)")
        end

        reduced_model = formulation(A_red, c_red, lhs_red, rhs_red, l_red, u_red, obj_offset)
        return reduced_model, presolver_info
    end
end

function apply_presolve(model::LP_info_cpu, params::HPRLP_parameters; presolve_params=nothing)
    backend = normalize_presolve_backend(params.presolve)
    if backend == "GPU"
        throw(ArgumentError("GPU presolve now requires an LP_info_gpu model. Route CPU->GPU transfer through optimize before calling GPU presolve."))
    elseif backend == "PSLP"
        return apply_pslp_presolve(model, params)
    end
    return model, nothing
end

function apply_presolve(model::LP_info_gpu, params::HPRLP_parameters; presolve_params=nothing)
    backend = normalize_presolve_backend(params.presolve)
    if backend == "GPU"
        return apply_gpu_presolve(model, params; presolve_params=presolve_params)
    elseif backend == "PSLP"
        throw(ArgumentError("PSLP presolve expects an LP_info_cpu model."))
    end
    return model, nothing
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

function release_solve_memory!(params::HPRLP_parameters)
    GC.gc(true)
    if params.use_gpu
        CUDA.synchronize()
        CUDA.reclaim()
    end
    return nothing
end
