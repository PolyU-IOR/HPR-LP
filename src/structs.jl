#=

Data Structures
	•	HPRLP_results → Stores solver outputs (iterations, runtime, primal/dual objectives, residuals, status).
	•	HPRLP_workspace_gpu → Holds GPU-accelerated variables (x, y, z), matrices (A, AT), and intermediate computations.
	•	HPRLP_residuals → Tracks primal/dual feasibility and duality gap for convergence checks.
	•	HPRLP_restart → Manages adaptive/fixed restart strategies.
	•	LP_info_cpu / LP_info_gpu → Stores LP problem data in CPU/GPU memory.
	•	Scaling_info → Applies problem scaling for numerical stability.
	•	HPRLP_parameters → Defines solver settings (tolerance, max iterations, restart strategy).

=#

"""
    HPRLP_parameters

Parameters for the HPR-LP solver.

# Fields
- `stoptol::Float64`: Stopping tolerance (default: 1e-4)
- `max_iter::Int`: Maximum number of iterations (default: typemax(Int32))
- `time_limit::Float64`: Time limit in seconds (default: 3600.0)
- `check_iter::Int`: Interval for residual checks (default: 150)
- `use_Ruiz_scaling::Bool`: Enable Ruiz scaling (default: true)
- `use_Pock_Chambolle_scaling::Bool`: Enable Pock-Chambolle scaling (default: true)
- `use_bc_scaling::Bool`: Enable b/c scaling (default: true)
- `use_gpu::Bool`: Use GPU acceleration (default: true)
- `device_number::Int`: GPU device number (default: 0)
- `warm_up::Bool`: Enable warm-up phase (default: true)
- `print_frequency::Int`: Print log every N iterations, -1 for auto (default: -1)
- `verbose::Bool`: Enable verbose output (default: true)
- `initial_x::Union{Vector{Float64},Nothing}`: Initial primal solution (default: nothing)
- `initial_y::Union{Vector{Float64},Nothing}`: Initial dual solution (default: nothing)
- `auto_save::Bool`: Automatically save best x, y, and sigma during optimization (default: false)
- `save_filename::String`: Filename for auto-save HDF5 file (default: "hprlp_autosave.h5")

# Example
```julia
params = HPRLP_parameters()
params.stoptol = 1e-6
params.use_gpu = false
params.verbose = false
params.auto_save = true
params.save_filename = "my_optimization.h5"
# Provide initial point
params.initial_x = x0
params.initial_y = y0
```
"""
mutable struct HPRLP_parameters
    # the stopping tolerance, default is 1e-6
    stoptol::Float64

    # the maximum number of iterations, default is 1000
    max_iter::Int

    # the time limit in seconds, default is 3600.0
    time_limit::Float64

    # the check interval for the residuals, default is 150
    check_iter::Int

    # whether to use the Ruiz scaling, default is true
    use_Ruiz_scaling::Bool

    # whether to use the Pock-Chambolle scaling, default is true
    use_Pock_Chambolle_scaling::Bool

    # whether to use the scaling for b and c, default is true
    use_bc_scaling::Bool

    # use GPU or not, default is true
    use_gpu::Bool

    # GPU device number, default is 0
    device_number::Int

    # whether do warm up, default is false
    warm_up::Bool

    # print frequency, print the log every print_frequency iterations, default is -1 (auto)
    print_frequency::Int

    # whether to print verbose output, default is true
    verbose::Bool

    # initial primal solution, default is nothing
    initial_x::Union{Vector{Float64},Nothing}

    # initial dual solution, default is nothing
    initial_y::Union{Vector{Float64},Nothing}

    # whether to automatically save the best x, y, and sigma, default is false
    auto_save::Bool

    # filename for auto-save HDF5 file, default is "hprlp_autosave.h5"
    save_filename::String

    # Default constructor
    HPRLP_parameters() = new(1e-4, typemax(Int32), 3600.0, 150, true, true, true, true, 0, true, -1, true, nothing, nothing, false, "hprlp_autosave.h5")
end

"""
    HPRLP_results

Results from the HPR-LP solver.

# Fields
- `iter::Int`: Total number of iterations
- `iter_4::Int`: Iterations to reach 1e-4 accuracy
- `iter_6::Int`: Iterations to reach 1e-6 accuracy
- `iter_8::Int`: Iterations to reach 1e-8 accuracy
- `time::Float64`: Total solve time in seconds
- `time_4::Float64`: Time to reach 1e-4 accuracy
- `time_6::Float64`: Time to reach 1e-6 accuracy
- `time_8::Float64`: Time to reach 1e-8 accuracy
- `primal_obj::Float64`: Primal objective value
- `dual_obj::Float64`: Dual objective value
- `primal_residual::Float64`: Final primal feasibility residual
- `dual_residual::Float64`: Final dual feasibility residual
- `relative_duality_gap::Float64`: Final relative duality gap
- `x::Vector{Float64}`: Primal solution vector
- `status::String`: Termination status ("Optimal", "TimeLimit", "IterationLimit")

# Example
```julia
model = build_from_mps("problem.mps")
params = HPRLP_parameters()
results = solve(model, params)
println("Status: ", results.status)
println("Objective: ", results.primal_obj)
println("Time: ", results.time, " seconds")
```
"""
mutable struct HPRLP_results
    # Number of iterations
    iter::Int

    # Number of iterations for the 1e-4 accuracy
    iter_4::Int

    # Number of iterations for the 1e-6 accuracy
    iter_6::Int

    # Number of iterations for the 1e-8 accuracy
    iter_8::Int

    # Time in seconds
    time::Float64

    # Time in seconds for the 1e-4 accuracy
    time_4::Float64

    # Time in seconds for the 1e-6 accuracy
    time_6::Float64

    # Time in seconds for the 1e-8 accuracy
    time_8::Float64

    # Time used by power method
    power_time::Float64

    # Primal objective value
    primal_obj::Float64

    # Relative residuals of the primal feasibility, dual feasibility, and objective gap
    residuals::Float64

    # Objective gap
    gap::Float64


    # OPTIMAL, MAX_ITER or TIME_LIMIT
    # OPTIMAL: the algorithm finds the optimal solution
    # MAX_ITER: the algorithm reaches the maximum number of iterations
    # TIME_LIMIT: the algorithm reaches the time limit
    status::String

    # The vector x
    x::Vector{Float64}

    # The vector y
    y::Vector{Float64}

    # The vector z
    z::Vector{Float64}

    # Default constructor
    HPRLP_results() = new()
end

# Define the CUSPARSE SpMV structure
mutable struct CUSPARSE_spmv_A
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_A::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_x_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_x_hat::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_dx::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_Ax::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

mutable struct CUSPARSE_spmv_AT
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_AT::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_y_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_y::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_ATy::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# Define the saved state structure for auto_save feature
mutable struct HPRLP_saved_state_gpu
    # Best x found so far (GPU)
    save_x::CuVector{Float64}
    
    # Best y found so far (GPU)
    save_y::CuVector{Float64}
    
    # Best sigma value
    save_sigma::Float64
    
    # Iteration when best state was saved
    save_iter::Int
    
    # Primal residual at best state
    save_err_Rp::Float64
    
    # Dual residual at best state
    save_err_Rd::Float64
    
    # Primal objective at best state
    save_primal_obj::Float64
    
    # Dual objective at best state
    save_dual_obj::Float64
    
    # Relative gap at best state
    save_rel_gap::Float64
    
    # Default constructor
    HPRLP_saved_state_gpu() = new()
end

mutable struct HPRLP_saved_state_cpu
    # Best x found so far (CPU)
    save_x::Vector{Float64}
    
    # Best y found so far (CPU)
    save_y::Vector{Float64}
    
    # Best sigma value
    save_sigma::Float64
    
    # Iteration when best state was saved
    save_iter::Int
    
    # Primal residual at best state
    save_err_Rp::Float64
    
    # Dual residual at best state
    save_err_Rd::Float64
    
    # Primal objective at best state
    save_primal_obj::Float64
    
    # Dual objective at best state
    save_dual_obj::Float64
    
    # Relative gap at best state
    save_rel_gap::Float64
    
    # Default constructor
    HPRLP_saved_state_cpu() = new()
end

# Define the workspace for the HPR-LP algorithm
mutable struct HPRLP_workspace_gpu
    # The vector x
    x::CuVector{Float64}

    # The vector x_hat, corresponding to ̂x in the paper
    x_hat::CuVector{Float64}

    # The vector x_bar, corresponding to x̄ in the paper
    x_bar::CuVector{Float64}

    # The vector dx, mainly used to store the difference between x1 and x2
    dx::CuVector{Float64}

    # The vector y
    y::CuVector{Float64}

    # The vector y_hat, corresponding to ̂y in the paper
    y_hat::CuVector{Float64}

    # The vector y_bar, corresponding to ȳ in the paper
    y_bar::CuVector{Float64}

    # The vector y_obj, used for computing the dual objective function variable
    y_obj::CuVector{Float64}

    # The vector dy, mainly used to store the difference between y1 and y2
    dy::CuVector{Float64}

    # The vector z_bar, corresponding to z̄ in the paper
    z_bar::CuVector{Float64}

    # The sparse matrix A, corresponding to A in the paper, the constraints matrix
    A::CuSparseMatrixCSR{Float64,Int32}

    # The CUSPARSE SpMV descriptor for A
    spmv_A::CUSPARSE_spmv_A

    # The sparse matrix A^T, the transpose of A
    AT::CuSparseMatrixCSR{Float64,Int32}

    # The CUSPARSE SpMV descriptor for AT
    spmv_AT::CUSPARSE_spmv_AT

    # The vector AL, the coefficients of the lower bound of the constraints
    AL::CuVector{Float64}

    # The vector AU, the coefficients of the lower bound of the constraints
    AU::CuVector{Float64}

    # The vector c, the coefficients of the objective function
    c::CuVector{Float64}

    # The vector l, the lower bound of the variables
    l::CuVector{Float64}

    # The vector u, the upper bound of the variables
    u::CuVector{Float64}

    # The vector Rp, normally used to store the vector b-Ax
    Rp::CuVector{Float64}

    # The vector Rd, normally used to store the vector c-A^Ty-z
    Rd::CuVector{Float64}

    # The total number of constraints
    m::Int

    # The number of variables
    n::Int

    # The value of σ
    sigma::Float64

    # The value of λ_max(AA^T), the maximum eigenvalue of the matrix AA^T
    lambda_max::Float64

    # Normally used to store the vector Ax
    Ax::CuVector{Float64}

    # Normally used to store the vector ATy
    ATy::CuVector{Float64}

    # Normally used to store the vector x that the algorithm restarted last time
    last_x::CuVector{Float64}

    # Normally used to store the vector y that the algorithm restarted last time
    last_y::CuVector{Float64}

    # Normally used to indicate whether the termination conditions should be checked
    to_check::Bool

    # Saved state for auto_save feature
    saved_state::HPRLP_saved_state_gpu
    
    # Flag to indicate if backup file has been created in this session
    backup_created::Bool

    # Default constructor
    HPRLP_workspace_gpu() = new()
end

mutable struct HPRLP_workspace_cpu
    x::Vector{Float64}
    x_hat::Vector{Float64}
    x_bar::Vector{Float64}
    dx::Vector{Float64}
    y::Vector{Float64}
    y_hat::Vector{Float64}
    y_bar::Vector{Float64}
    y_obj::Vector{Float64}
    dy::Vector{Float64}
    z_bar::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    Rp::Vector{Float64}
    Rd::Vector{Float64}
    m::Int
    n::Int
    sigma::Float64
    lambda_max::Float64
    Ax::Vector{Float64}
    ATy::Vector{Float64}
    last_x::Vector{Float64}
    last_y::Vector{Float64}
    to_check::Bool
    saved_state::HPRLP_saved_state_cpu
    HPRLP_workspace_cpu() = new()
end

# Define the variables related to the residuals of the HPR-LP
mutable struct HPRLP_residuals
    # The relative residuals of the primal feasibility evaluated at x_bar
    err_Rp_org_bar::Float64

    # The relative residuals of the dual feasibility evaluated at y_bar and z_bar
    err_Rd_org_bar::Float64

    # The primal objective value evaluated at x_bar
    primal_obj_bar::Float64

    # The dual objective value evaluated at y_bar and z_bar
    dual_obj_bar::Float64

    # The relative gap evaluated at x_bar, y_bar, and z_bar
    rel_gap_bar::Float64

    # The maximum of the primal feasibility, dual feasibility, and duality gap
    KKTx_and_gap_org_bar::Float64

    # Define a default constructor
    HPRLP_residuals() = new()
end

# Define the variables related to the restart of the HPR-LP
mutable struct HPRLP_restart
    # indicate which restart condition is satisfied, 1: sufficient, 2: necessary, 3: long
    restart_flag::Int

    # indicate whether it is the first restart
    first_restart::Bool

    # the value \tilde{R}_{r,0}
    last_gap::Float64

    # the value \tilde{R}_{r,t}
    current_gap::Float64

    # the value \tilde{R}_{r,t-1}
    save_gap::Float64

    # the best value \tilde{R}_{best}
    best_gap::Float64

    # the  value of sigma at the best_gap
    best_sigma::Float64

    # the number of inner iterations, t in the paper
    inner::Int

    # the number of restart triggered by sufficient decrease
    sufficient::Int

    # the number of restart triggered by necessary decrease
    necessary::Int

    # the number of restart triggered by long iterations
    long::Int

    # the number of restart
    times::Int

    # Default constructor
    HPRLP_restart() = new()
end

# the space for the LP information on the CPU
mutable struct LP_info_cpu
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    obj_constant::Float64
end

# the space for the LP information on the GPU
mutable struct LP_info_gpu
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    c::CuVector{Float64}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    obj_constant::Float64
end

# the space for the scaling information on the CPU
mutable struct Scaling_info_cpu
    # the original vector l
    l_org::Vector{Float64}

    # the original vector u
    u_org::Vector{Float64}

    # the row norm of the matrix A
    row_norm::Vector{Float64}

    # the column norm of the matrix A
    col_norm::Vector{Float64}

    # the scaling factor for the vector b
    b_scale::Float64

    # the scaling factor for the vector c
    c_scale::Float64

    # the norm of the vector b
    norm_b::Float64

    # the norm of the vector c
    norm_c::Float64

    # the norm of the original vector b
    norm_b_org::Float64

    # the norm of the original vector c
    norm_c_org::Float64
end

# the space for the scaling information on the GPU
mutable struct Scaling_info_gpu
    l_org::CuVector{Float64}
    u_org::CuVector{Float64}
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
end