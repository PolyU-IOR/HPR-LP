# Helper function to create scaling info and apply scaling to the LP problem
function scaling!(lp::LP_info_cpu, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
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
    standard_lp = LP_info_cpu(A_copy, transpose(A_copy), c_copy, AL_copy, AU_copy, l_copy, u_copy, obj_constant)

    return standard_lp
end

"""
    build_from_mps(filename::AbstractString; mpsformat::Symbol = :auto)

Read an `.mps` or `.mps.gz` file with `MPSReader` and convert it into an `LP_info_cpu`
model for HPR-LP.

# Arguments
- `filename::AbstractString`: Path to the MPS file.
- `mpsformat::Symbol`: `:auto`, `:fixed`, or `:free`.

# Returns
- `LP_info_cpu`: LP model ready for `optimize` or `solve`.
"""
function build_from_mps(filename::AbstractString; mpsformat::Symbol = :auto)
    data = MPSReader.read_mps(filename; keep_names = false, mpsformat = mpsformat)
    A = sparse(data.arows, data.acols, data.avals, data.nrow, data.ncol)
    return build_from_Abc(A, data.c, data.lcon, data.ucon, data.lvar, data.uvar, data.obj_constant)
end

