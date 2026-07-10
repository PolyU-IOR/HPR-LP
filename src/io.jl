const LP_HDF5_SCHEMA_VERSION = Int32(1)

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

    # Build the LP model
    standard_lp = formulation(A_sparse, c, AL, AU, l, u, obj_constant)

    return standard_lp
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
function build_from_mps(filename::AbstractString; verbose::Bool=true, mpsformat::Symbol=:auto)
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
    save_lp_to_hdf5(filename, model; overwrite=true)
    save_lp_to_hdf5(filename, A, c, AL, AU, l, u, obj_constant=0.0; overwrite=true)

Save an LP in HPRLP's matrix form to an HDF5 cache file.

The file stores `A` in compressed sparse column form under `A/`, plus `c`, `AL`,
`AU`, `l`, `u`, and `obj_constant`.
"""
function save_lp_to_hdf5(filename::AbstractString, model::LP_info_cpu; overwrite::Bool=true)
    return save_lp_to_hdf5(
        filename,
        model.A,
        model.c,
        model.AL,
        model.AU,
        model.l,
        model.u,
        model.obj_constant;
        overwrite=overwrite,
    )
end

function save_lp_to_hdf5(filename::AbstractString,
    A::Union{SparseMatrixCSC, Matrix},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64=0.0;
    overwrite::Bool=true)

    if !overwrite && isfile(filename)
        throw(ArgumentError("Refusing to overwrite existing HDF5 file: $filename"))
    end

    A_sparse = A isa Matrix ? sparse(A) : A
    m, n = size(A_sparse)
    length(c) == n || throw(DimensionMismatch("length(c) must equal size(A, 2)"))
    length(l) == n || throw(DimensionMismatch("length(l) must equal size(A, 2)"))
    length(u) == n || throw(DimensionMismatch("length(u) must equal size(A, 2)"))
    length(AL) == m || throw(DimensionMismatch("length(AL) must equal size(A, 1)"))
    length(AU) == m || throw(DimensionMismatch("length(AU) must equal size(A, 1)"))

    h5open(filename, "w") do file
        write(file, "schema_version", LP_HDF5_SCHEMA_VERSION)
        write(file, "obj_constant", obj_constant)
        write(file, "c", c)
        write(file, "AL", AL)
        write(file, "AU", AU)
        write(file, "l", l)
        write(file, "u", u)

        A_group = create_group(file, "A")
        write(A_group, "size", Int64[m, n])
        write(A_group, "colptr", A_sparse.colptr)
        write(A_group, "rowval", rowvals(A_sparse))
        write(A_group, "nzval", nonzeros(A_sparse))
    end

    return filename
end

function save_lp_to_hdf5(filename::AbstractString;
    A,
    AL,
    AU,
    c,
    l,
    u,
    obj_constant=0.0,
    overwrite::Bool=true)

    return save_lp_to_hdf5(
        filename,
        A,
        c,
        AL,
        AU,
        l,
        u,
        Float64(obj_constant);
        overwrite=overwrite,
    )
end

"""
    save_mps_as_hdf5(mps_filename, hdf5_filename; verbose=true, mpsformat=:auto, overwrite=true)

Read an MPS file once and save the resulting `(A, c, AL, AU, l, u, obj_constant)`
data to an HDF5 cache for faster repeated experiments.
"""
function save_mps_as_hdf5(
    mps_filename::AbstractString,
    hdf5_filename::AbstractString;
    verbose::Bool=true,
    mpsformat::Symbol=:auto,
    overwrite::Bool=true,
)
    model = build_from_mps(mps_filename, verbose; mpsformat=mpsformat)
    save_lp_to_hdf5(hdf5_filename, model; overwrite=overwrite)
    return hdf5_filename
end

"""
    read_from_hdf5(filename; verbose=true)

Read an LP saved by [`save_lp_to_hdf5`](@ref) or [`save_mps_as_hdf5`](@ref) and
return an `LP_info_cpu` model.
"""
function read_from_hdf5(filename::AbstractString; verbose::Bool=true)
    t_start = time()
    if verbose
        println("READING HDF5 FILE ... ", filename)
    end

    data = h5open(filename, "r") do file
        version = Int32(read(file, "schema_version"))
        version == LP_HDF5_SCHEMA_VERSION ||
            throw(ArgumentError("Unsupported LP HDF5 schema version $version"))

        A_size = Vector{Int64}(read(file, "A/size"))
        length(A_size) == 2 || throw(ArgumentError("Invalid A/size dataset in $filename"))
        colptr = Vector{Int32}(read(file, "A/colptr"))
        rowval = Vector{Int32}(read(file, "A/rowval"))
        nzval = Vector{Float64}(read(file, "A/nzval"))
        A = SparseMatrixCSC{Float64,Int32}(Int(A_size[1]), Int(A_size[2]), colptr, rowval, nzval)

        c = Vector{Float64}(read(file, "c"))
        AL = Vector{Float64}(read(file, "AL"))
        AU = Vector{Float64}(read(file, "AU"))
        l = Vector{Float64}(read(file, "l"))
        u = Vector{Float64}(read(file, "u"))
        obj_constant = Float64(read(file, "obj_constant"))
        return A, c, AL, AU, l, u, obj_constant
    end

    if verbose
        println(@sprintf("READING HDF5 FILE time: %.2f seconds", time() - t_start))
    end

    return build_from_Abc(data...)
end

read_from_h5(filename::AbstractString; kwargs...) = read_from_hdf5(filename; kwargs...)

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

    try
        for i = 1:length(files)
            file = files[i]
            if file in namelist
                println("The result of problem exists: ", file)
            end
            if (occursin(".mps", file) || occursin(".h5", file)) && !(file in namelist)
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
                        if occursin(".mps", file)
                            model = build_from_mps(FILE_NAME; verbose=params.verbose)
                        elseif occursin(".h5", file)
                            model = read_from_hdf5(FILE_NAME; verbose=params.verbose)
                        else
                            throw(ArgumentError("Unsupported file format: $file"))
                        end
                        results = optimize(model, params)

                        all_time = time() - t_start_all
                        println("Solve complete ----------------------------------------------------------------------------------------------------------")

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
                        push!(statuslist, results.status)
                        push!(iter4list, results.iter_4)
                        push!(time4list, min(results.time_4, params.time_limit))
                        push!(iter6list, results.iter_6)
                        push!(time6list, min(results.time_6, params.time_limit))
                        push!(iter8list, results.iter_8)
                        push!(time8list, min(results.time_8, params.time_limit))
                    finally
                        model = nothing
                        results = nothing
                        release_solve_memory!(params)
                    end
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
    finally
        close(io)
    end
end
