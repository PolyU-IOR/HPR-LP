# The function to read the LP problem from the file and formulate the LP problem
function formulation(lp, verbose::Bool=true)
    A = sparse(lp.arows, lp.acols, lp.avals, lp.ncon, lp.nvar)

    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall((sum(abs_A, dims=2)[:, 1] .== 0) .| ((lp.lcon .== -Inf) .& (lp.ucon .== Inf)))

    if length(del_row) > 0
        keep_rows = setdiff(1:size(A, 1), del_row)
        A = A[keep_rows, :]
        lp.lcon = lp.lcon[keep_rows]
        lp.ucon = lp.ucon[keep_rows]
        if verbose
            println("Deleted ", length(del_row), " rows of A that are all zeros.")
        end
    end

    # Get the index of the different types of constraints
    idxE = findall(lp.lcon .== lp.ucon)
    idxG = findall((lp.lcon .> -Inf) .& (lp.ucon .== Inf))
    idxL = findall((lp.lcon .== -Inf) .& (lp.ucon .< Inf))
    idxB = findall((lp.lcon .> -Inf) .& (lp.ucon .< Inf))
    idxB = setdiff(idxB, idxE)

    if verbose
        println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz A = ", nnz(A))
        println("                     number of equalities = ", length(idxE))
        println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))
    end

    @assert length(lp.lcon) == length(idxE) + length(idxG) + length(idxL) + length(idxB)

    standard_lp = LP_info_cpu(A, transpose(A), lp.c, lp.lcon, lp.ucon, lp.lvar, lp.uvar, lp.c0)

    # Return the modified lp
    return standard_lp
end

# the scaling function for the LP problem
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
    return scaling_info
end

function power_iteration_gpu(spmv_A::CUSPARSE_spmv_A, spmv_AT::CUSPARSE_spmv_AT, m::Int, n::Int,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    desc_q = CUDA.CUSPARSE.CuDenseVectorDescriptor(q)
    desc_z = CUDA.CUSPARSE.CuDenseVectorDescriptor(z)
    desc_ATq = CUDA.CUSPARSE.CuDenseVectorDescriptor(ATq)
    lambda_max = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.cusparseSpMV(spmv_AT.handle, spmv_AT.operator, spmv_AT.alpha, spmv_AT.desc_AT, desc_q, spmv_AT.beta, desc_ATq,
        spmv_AT.compute_type, spmv_AT.alg, spmv_AT.buf)
        CUDA.CUSPARSE.cusparseSpMV(spmv_A.handle, spmv_A.operator, spmv_A.alpha, spmv_A.desc_A, desc_ATq, spmv_A.beta, desc_z,
        spmv_A.compute_type, spmv_A.alg, spmv_A.buf)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q
        if CUDA.norm(q) < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", CUDA.norm(q))
    return lambda_max
end

function power_iteration_cpu(A::SparseMatrixCSC, AT::SparseMatrixCSC,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q
        if norm(q) < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", norm(q))
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

# the function to run the HPR-LP algorithm on a single file
function run_file(FILE_NAME::String, params::HPRLP_parameters)
    # Validate GPU parameters before proceeding
    validate_gpu_parameters!(params)
    
    t_start = time()
    if params.verbose
        println("READING FILE ... ", FILE_NAME)
    end
    io = open(FILE_NAME)
    lp = Logging.with_logger(Logging.NullLogger()) do
        readqps(io, mpsformat=:free)
    end
    close(io)
    read_time = time() - t_start
    if params.verbose
        println(@sprintf("READING FILE time: %.2f seconds", read_time))
    end

    t_start = time()
    setup_start = time()
    if params.verbose
        println("FORMULATING LP ...")
    end
    standard_lp = formulation(lp, params.verbose)
    if params.verbose
        println(@sprintf("FORMULATING LP time: %.2f seconds", time() - t_start))
    end

    if params.use_gpu
        CUDA.device!(params.device_number)
        t_start = time()
        if params.verbose
            println("SCALING LP ...")
        end
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        if params.verbose
            println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
        end

        CUDA.synchronize()
        t_start = time()
        if params.verbose
            println("COPY TO GPU ...")
        end
        standard_lp_gpu = LP_info_gpu(CuSparseMatrixCSR(standard_lp.A),
            CuSparseMatrixCSR(standard_lp.A'),
            CuVector(standard_lp.c),
            CuVector(standard_lp.AL),
            CuVector(standard_lp.AU),
            CuVector(standard_lp.l),
            CuVector(standard_lp.u),
            standard_lp.obj_constant,
        )
        scaling_info_gpu = Scaling_info_gpu(CuVector(scaling_info.l_org),
            CuVector(scaling_info.u_org),
            CuVector(scaling_info.row_norm),
            CuVector(scaling_info.col_norm),
            scaling_info.b_scale,
            scaling_info.c_scale,
            scaling_info.norm_b,
            scaling_info.norm_c,
            scaling_info.norm_b_org,
            scaling_info.norm_c_org)
        CUDA.synchronize()
        if params.verbose
            println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
        end
    else
        t_start = time()
        if params.verbose
            println("SCALING LP ...")
        end
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        if params.verbose
            println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
        end
    end

    setup_time = time() - setup_start
    if params.use_gpu
        results = solve(standard_lp_gpu, scaling_info_gpu, params)
    else
        results = solve(standard_lp, scaling_info, params)
    end
    if params.verbose
        println(@sprintf("Total time: %.2fs", read_time + setup_time + results.time),
            @sprintf("  read time = %.2fs", read_time),
            @sprintf("  setup time = %.2fs", setup_time),
            @sprintf("  solve time = %.2fs", results.time))
    end

    return results
end

"""
    run_lp(A, AL, AU, c, l, u, obj_constant, params)

Solve a linear program in standard form using the HPR-LP algorithm.

# Arguments
- `A::SparseMatrixCSC`: Constraint matrix (m × n)
- `AL::Vector{Float64}`: Lower bounds for constraints Ax (length m)
- `AU::Vector{Float64}`: Upper bounds for constraints Ax (length m)
- `c::Vector{Float64}`: Objective coefficients (length n)
- `l::Vector{Float64}`: Lower bounds for variables x (length n)
- `u::Vector{Float64}`: Upper bounds for variables x (length n)
- `obj_constant::Float64`: Constant term in objective function
- `params::HPRLP_parameters`: Solver parameters

# Returns
- `HPRLP_results`: Solution results including objective value, solution vector, and convergence info

# Example
```julia
using SparseArrays, HPRLP

# min -3x - 5y
# s.t. x + 2y ≤ 10
#      3x + y ≤ 12
#      x, y ≥ 0

A = sparse([1.0 2.0; 3.0 1.0])
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
c = [-3.0, -5.0]
l = [0.0, 0.0]
u = [Inf, Inf]

params = HPRLP_parameters()
params.verbose = false

results = run_lp(A, AL, AU, c, l, u, 0.0, params)
println("Optimal value: ", results.primal_obj)
println("Solution: ", results.x)
```
"""
function run_lp(A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    c::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64,
    params::HPRLP_parameters)
    
    # Validate GPU parameters before proceeding
    validate_gpu_parameters!(params)
    
    if params.warm_up
        if params.verbose
            println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
        end
        t_start_all = time()
        max_iter = params.max_iter
        params.max_iter = 200
        results = run_Abc(A, c, AL, AU, l, u, obj_constant, params)
        params.max_iter = max_iter
        all_time = time() - t_start_all
        if params.verbose
            println("warm up time: ", all_time)
            println("warm up ends ----------------------------------------------------------------------------------------------------------")
        end
    end
    if params.verbose
        println("main run starts: ----------------------------------------------------------------------------------------------------------")
    end
    results = run_Abc(A, c, AL, AU, l, u, obj_constant, params)
    if params.verbose
        println("main run ends----------------------------------------------------------------------------------------------------------")
    end
    return results
end

# the function to run the HPR-LP algorithm on a single LP problem with A, b, c, l, u, m1, obj_constant
function run_Abc(A::SparseMatrixCSC,
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64,
    params::HPRLP_parameters)
    A = copy(A)
    c = copy(c)
    AL = copy(AL)
    AU = copy(AU)
    l = copy(l)
    u = copy(u)
    setup_start = time()
    standard_lp = LP_info_cpu(A, transpose(A), c, AL, AU, l, u, obj_constant)
    if params.use_gpu
        CUDA.device!(params.device_number)
        t_start = time()
        if params.verbose
            println("SCALING LP ...")
        end
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        if params.verbose
            println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
        end

        CUDA.synchronize()
        t_start = time()
        if params.verbose
            println("COPY TO GPU ...")
        end
        standard_lp_gpu = LP_info_gpu(CuSparseMatrixCSR(standard_lp.A),
            CuSparseMatrixCSR(standard_lp.A'),
            CuVector(standard_lp.c),
            CuVector(standard_lp.AL),
            CuVector(standard_lp.AU),
            CuVector(standard_lp.l),
            CuVector(standard_lp.u),
            standard_lp.obj_constant,
        )
        scaling_info_gpu = Scaling_info_gpu(CuVector(scaling_info.l_org),
            CuVector(scaling_info.u_org),
            CuVector(scaling_info.row_norm),
            CuVector(scaling_info.col_norm),
            scaling_info.b_scale,
            scaling_info.c_scale,
            scaling_info.norm_b,
            scaling_info.norm_c,
            scaling_info.norm_b_org,
            scaling_info.norm_c_org)
        CUDA.synchronize()
        if params.verbose
            println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
        end
    else
        t_start = time()
        if params.verbose
            println("SCALING LP ...")
        end
        scaling_info = scaling!(standard_lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        if params.verbose
            println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
        end
    end
    setup_time = time() - setup_start

    if params.use_gpu
        results = solve(standard_lp_gpu, scaling_info_gpu, params)
    else
        results = solve(standard_lp, scaling_info, params)
    end
    if params.verbose
        println(@sprintf("Total time: %.2fs", setup_time + results.time),
            @sprintf("  setup time = %.2fs", setup_time),
            @sprintf("  solve time = %.2fs", results.time))
    end
    return results
end

# the function to test the HPR-LP algorithm on a dataset
function run_dataset(data_path::String, result_path::String, params::HPRLP_parameters)
    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = joinpath(result_path, "HPRLP_result.csv")

    # redirect the output to a file
    log_path = joinpath(result_path, "HPRLP_log.txt")

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time4list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        iter8list = Vector{Any}(result_table.iter_8[1:end-2])
        time8list = Vector{Any}(result_table.time_8[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time4list = []
        iter6list = []
        time6list = []
        iter8list = []
        time8list = []
    end


    warm_up_done = false
    for i = 1:length(files)
        file = files[i]
        if file in namelist
            println("The result of problem exists: ", file)
        end
        if occursin(".mps", file) && !(file in namelist)
            FILE_NAME = joinpath(data_path, file)
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
            # println(file)

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                if params.warm_up && !warm_up_done
                    warm_up_done = true
                    println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                    t_start_all = time()
                    max_iter = params.max_iter
                    params.max_iter = 200
                    results = run_file(FILE_NAME, params)
                    params.max_iter = max_iter
                    all_time = time() - t_start_all
                    println("warm up time: ", all_time)
                    println("warm up ends ----------------------------------------------------------------------------------------------------------")
                end


                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter4list, results.iter_4)
                push!(time4list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
                push!(iter8list, results.iter_8)
                push!(time8list, min(results.time_8, params.time_limit))
            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time4list,
                iter_6=iter6list,
                time_6=time6list,
                iter_8=iter8list,
                time_8=time8list
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time4list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_time_8 = exp(mean(log.(time8list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            geomean_iter_8 = exp(mean(log.(iter8list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, geomean_iter_8, geomean_time_8])
            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_4 = count(x -> x < params.time_limit, time4list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            solved_8 = count(x -> x < params.time_limit, time8list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_4, "", solved_6, "", solved_8])

            CSV.write(csv_file, result_table)
        end
    end
    println("The solver has finished running the dataset, total ", length(files), " problems")

    close(io)
end

"""
    run_single(file_name, params)

Solve a linear program from an MPS file using the HPR-LP algorithm.

# Arguments
- `file_name::String`: Path to the .mps file
- `params::HPRLP_parameters`: Solver parameters

# Returns
- `HPRLP_results`: Solution results including objective value, solution vector, and convergence info

# Example
```julia
using HPRLP

params = HPRLP_parameters()
params.stoptol = 1e-6
params.verbose = true

results = run_single("problem.mps", params)

println("Status: ", results.status)
println("Objective: ", results.primal_obj)
println("Time: ", results.time, " seconds")
println("Iterations: ", results.iter)
```

See also: [`run_lp`](@ref), [`HPRLP_parameters`](@ref), [`HPRLP_results`](@ref)
"""
function run_single(file_name::String, params::HPRLP_parameters)
    if params.verbose
        println("data path: ", file_name)
    end

    if occursin(".mps", file_name)
        if params.warm_up
            if params.verbose
                println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
            end
            t_start_all = time()
            max_iter = params.max_iter
            params.max_iter = 200
            results = run_file(file_name, params)
            params.max_iter = max_iter
            all_time = time() - t_start_all
            if params.verbose
                println("warm up time: ", all_time)
                println("warm up ends ----------------------------------------------------------------------------------------------------------")
            end
        end

        if params.verbose
            println("main run starts: ----------------------------------------------------------------------------------------------------------")
        end
        results = run_file(file_name, params)
        if params.verbose
            println("main run ends----------------------------------------------------------------------------------------------------------")
        end
    else
        error("The file is not in the correct format, please provide a .mps file")
    end

    return results
end


# Define the product of sets at module level (required for struct definition)
MOI.Utilities.@product_of_sets(
    LPSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
)

# Define the cache type with MatrixOfConstraints (same approach as Clp.jl)
const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Float64},
        LPSets{Float64},
    },
}
