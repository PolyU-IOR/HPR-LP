function repeat_batched_columns(v::AbstractVector{<:Real}, batch_size::Int)
    return repeat(reshape(Float64.(v), :, 1), 1, batch_size)
end

function make_batched_objectives(
    c::AbstractVector{<:Real},
    batch_size::Int,
    mode::Symbol,
    scale::Float64,
    rng::AbstractRNG,
)
    c_float = Float64.(c)
    C = repeat_batched_columns(c_float, batch_size)
    if mode == :copy
        return C
    elseif mode == :perturbed_obj
        for k in 2:batch_size
            C[:, k] .= c_float .* (1.0 .+ scale .* randn(rng, length(c_float)))
        end
        return C
    else
        throw(ArgumentError("Unsupported instance_mode $(mode). Use :copy or :perturbed_obj."))
    end
end

function _read_existing_batched_results(csv_file::String, detail_csv_file::String)
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        keep = .!in.(String.(result_table.name), Ref(["SGM10", "solved"]))
        result_table = result_table[keep, :]
        done_names = Set(String.(result_table.name))
    else
        result_table = DataFrame(
            name=String[],
            batch_size=Int[],
            instance_mode=String[],
            alg_time=Float64[],
            setup_time=Float64[],
            solve_time=Float64[],
            power_time=Float64[],
            wall_time=Float64[],
            iter=Int[],
            res=Float64[],
            gap=Float64[],
            primal_obj_min=Float64[],
            primal_obj_max=Float64[],
            status=String[],
            warm_up=Bool[],
        )
        done_names = Set{String}()
    end

    if isfile(detail_csv_file)
        detail_table = CSV.read(detail_csv_file, DataFrame)
    else
        detail_table = DataFrame(
            name=String[],
            batch_index=Int[],
            instance_mode=String[],
            iter=Int[],
            alg_time=Float64[],
            setup_time=Float64[],
            solve_time=Float64[],
            res=Float64[],
            gap=Float64[],
            primal_obj=Float64[],
            status=String[],
        )
    end

    return result_table, detail_table, done_names
end

function _batched_summary_table(result_table::DataFrame, params::HPRLP_parameters)
    summary = copy(result_table)
    if nrow(result_table) == 0
        return summary
    end

    alg_time = Float64.(result_table.alg_time)
    setup_time = Float64.(result_table.setup_time)
    solve_time = Float64.(result_table.solve_time)
    wall_time = Float64.(result_table.wall_time)
    iter = Float64.(result_table.iter)

    geomean_alg = exp(mean(log.(alg_time .+ 10.0))) - 10.0
    geomean_setup = exp(mean(log.(setup_time .+ 10.0))) - 10.0
    geomean_solve = exp(mean(log.(solve_time .+ 10.0))) - 10.0
    geomean_wall = exp(mean(log.(wall_time .+ 10.0))) - 10.0
    geomean_iter = exp(mean(log.(iter .+ 10.0))) - 10.0
    solved = count(==("OPTIMAL"), String.(result_table.status))

    push!(summary, (
        name="SGM10",
        batch_size=missing,
        instance_mode="",
        alg_time=geomean_alg,
        setup_time=geomean_setup,
        solve_time=geomean_solve,
        power_time=missing,
        wall_time=geomean_wall,
        iter=geomean_iter,
        res=missing,
        gap=missing,
        primal_obj_min=missing,
        primal_obj_max=missing,
        status="",
        warm_up=missing,
    ), promote=true)
    push!(summary, (
        name="solved",
        batch_size=missing,
        instance_mode="",
        alg_time=solved,
        setup_time=missing,
        solve_time=missing,
        power_time=missing,
        wall_time=missing,
        iter=missing,
        res=missing,
        gap=missing,
        primal_obj_min=missing,
        primal_obj_max=missing,
        status="",
        warm_up=missing,
    ), promote=true)
    return summary
end

function _batched_error_status(err)
    msg = sprint(showerror, err)
    msg = replace(msg, '\n' => ' ')
    if ncodeunits(msg) > 160
        msg = first(msg, 160) * "..."
    end
    return "ERROR: " * msg
end

function _record_batched_failure!(
    result_table::DataFrame,
    detail_table::DataFrame,
    file::String,
    batch_size::Int,
    instance_mode::Symbol,
    params::HPRLP_parameters,
    all_time::Float64,
    status::String,
)
    push!(result_table, (
        name=file,
        batch_size=batch_size,
        instance_mode=String(instance_mode),
        alg_time=params.time_limit,
        setup_time=0.0,
        solve_time=0.0,
        power_time=0.0,
        wall_time=all_time,
        iter=params.max_iter,
        res=Inf,
        gap=Inf,
        primal_obj_min=NaN,
        primal_obj_max=NaN,
        status=status,
        warm_up=params.warm_up,
    ))

    for k in 1:batch_size
        push!(detail_table, (
            name=file,
            batch_index=k,
            instance_mode=String(instance_mode),
            iter=params.max_iter,
            alg_time=params.time_limit,
            setup_time=0.0,
            solve_time=0.0,
            res=Inf,
            gap=Inf,
            primal_obj=NaN,
            status=status,
        ))
    end
    return nothing
end

function _safe_release_batched_solve_memory!(params::HPRLP_parameters)
    try
        release_solve_memory!(params)
    catch err
        println("Warning: failed to release memory after batched dataset instance failure.")
        showerror(stdout, err, catch_backtrace())
        println()
    end
    return nothing
end

function run_batched_dataset(
    data_path::String,
    result_path::String,
    params::HPRLP_parameters;
    batch_size::Int=4,
    instance_mode::Symbol=:copy,
    obj_perturbation_scale::Float64=1.0e-3,
    random_seed::Int=1234,
)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    params.use_gpu || throw(ArgumentError("run_batched_dataset requires params.use_gpu=true."))
    normalize_presolve_backend(params.presolve) == "NONE" || throw(ArgumentError("run_batched_dataset currently supports presolve = \"NONE\" only."))

    files = sort(readdir(data_path))
    mps_files = filter(file -> occursin(".mps", file), files)

    if !isdir(result_path)
        mkpath(result_path)
    end

    csv_file = joinpath(result_path, "HPRLP_batched_result.csv")
    detail_csv_file = joinpath(result_path, "HPRLP_batched_columns.csv")
    log_path = joinpath(result_path, "HPRLP_batched_log.txt")

    result_table, detail_table, done_names = _read_existing_batched_results(csv_file, detail_csv_file)
    rng = MersenneTwister(random_seed)

    io = open(log_path, "a")
    try
        for (i, file) in enumerate(mps_files)
            if file in done_names
                println("The batched result of problem exists: ", file)
                continue
            end

            file_name = joinpath(data_path, file)
            println(@sprintf("solving the batched problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                t_start_all = time()
                model = nothing
                result = nothing
                try
                    println(@sprintf("solving the batched problem %d", i), @sprintf(": %s", file))
                    println("Solving batched: ----------------------------------------------------------------------------------------------------------")

                    model = build_from_mps(file_name, params.verbose)
                    C = make_batched_objectives(model.c, batch_size, instance_mode, obj_perturbation_scale, rng)
                    AL = repeat_batched_columns(model.AL, batch_size)
                    AU = repeat_batched_columns(model.AU, batch_size)
                    L = repeat_batched_columns(model.l, batch_size)
                    U = repeat_batched_columns(model.u, batch_size)
                    obj_constants = fill(model.obj_constant, batch_size)

                    result = optimize_batched_gpu(
                        model.A,
                        C,
                        AL,
                        AU,
                        L,
                        U,
                        params;
                        obj_constants=obj_constants,
                    )

                    all_time = time() - t_start_all
                    aggregate_status = all(==("OPTIMAL"), result.status) ? "OPTIMAL" : join(unique(result.status), ";")

                    println("Batched solve complete ----------------------------------------------------------------------------------------------------------")
                    println("batch_size = ", batch_size,
                        "  instance_mode = ", instance_mode,
                        "  iter_max = ", maximum(result.iter),
                        @sprintf("  time = %3.2e", result.time),
                        @sprintf("  wall = %3.2e", all_time),
                        @sprintf("  residual_max = %3.2e", maximum(result.residuals)),
                        @sprintf("  gap_max = %3.2e", maximum(result.gap)),
                    )

                    push!(result_table, (
                        name=file,
                        batch_size=batch_size,
                        instance_mode=String(instance_mode),
                        alg_time=min(result.time, params.time_limit),
                        setup_time=result.setup_time,
                        solve_time=result.solve_time,
                        power_time=result.power_time,
                        wall_time=all_time,
                        iter=maximum(result.iter),
                        res=maximum(result.residuals),
                        gap=maximum(result.gap),
                        primal_obj_min=minimum(result.primal_obj),
                        primal_obj_max=maximum(result.primal_obj),
                        status=aggregate_status,
                        warm_up=params.warm_up,
                    ))

                    for k in eachindex(result.status)
                        push!(detail_table, (
                            name=file,
                            batch_index=k,
                            instance_mode=String(instance_mode),
                            iter=result.iter[k],
                            alg_time=min(result.time, params.time_limit),
                            setup_time=result.setup_time,
                            solve_time=result.solve_time,
                            res=result.residuals[k],
                            gap=result.gap[k],
                            primal_obj=result.primal_obj[k],
                            status=result.status[k],
                        ))
                    end
                catch err
                    err isa InterruptException && rethrow()

                    all_time = time() - t_start_all
                    status = _batched_error_status(err)

                    println("Batched solve failed ----------------------------------------------------------------------------------------------------------")
                    println("file = ", file)
                    println("status = ", status)
                    println(@sprintf("wall = %3.2e", all_time))
                    println("Stacktrace:")
                    showerror(stdout, err, catch_backtrace())
                    println()

                    _record_batched_failure!(
                        result_table,
                        detail_table,
                        file,
                        batch_size,
                        instance_mode,
                        params,
                        all_time,
                        status,
                    )
                finally
                    model = nothing
                    result = nothing
                    _safe_release_batched_solve_memory!(params)
                end
            end

            CSV.write(csv_file, _batched_summary_table(result_table, params))
            CSV.write(detail_csv_file, detail_table)
        end
    finally
        close(io)
    end

    println("The batched solver has finished running the dataset, total ", length(mps_files), " MPS problems")
    println("Aggregate results: ", csv_file)
    println("Column results: ", detail_csv_file)
    println("Log: ", log_path)
    return nothing
end