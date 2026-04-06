import HPRLP
using CSV
using DataFrames
using JuMP
using Printf
using Statistics

data_path = "xxx" # Replace with the actual path to your dataset
result_path = "xxx" # Replace with the actual path where you want to save the results

function set_hprlp_attributes!(model::Model, params::HPRLP.HPRLP_parameters)
	set_attribute(model, "stoptol", params.stoptol)
	set_attribute(model, "max_iter", params.max_iter)
	set_attribute(model, "time_limit", Float64(params.time_limit))
	set_attribute(model, "check_iter", params.check_iter)
	set_attribute(model, "use_Ruiz_scaling", params.use_Ruiz_scaling)
	set_attribute(model, "use_Pock_Chambolle_scaling", params.use_Pock_Chambolle_scaling)
	set_attribute(model, "use_bc_scaling", params.use_bc_scaling)
	set_attribute(model, "use_gpu", params.use_gpu)
	set_attribute(model, "CUSPARSE_spmv", params.CUSPARSE_spmv)
	set_attribute(model, "device_number", params.device_number)
	set_attribute(model, "warm_up", params.warm_up)
	set_attribute(model, "print_frequency", params.print_frequency)
	set_attribute(model, "verbose", params.verbose)
	return nothing
end

function initialize_result_lists(csv_file::String)
	if !isfile(csv_file)
		return (
			namelist=Any[],
			iterlist=Any[],
			timelist=Any[],
			reslist=Any[],
			objlist=Any[],
			statuslist=Any[],
			iter4list=Any[],
			time4list=Any[],
			iter6list=Any[],
			time6list=Any[],
			iter8list=Any[],
			time8list=Any[],
		)
	end

	result_table = CSV.read(csv_file, DataFrame)
	data_rows = max(nrow(result_table) - 2, 0)
	return (
		namelist=Vector{Any}(result_table.name[1:data_rows]),
		iterlist=Vector{Any}(result_table.iter[1:data_rows]),
		timelist=Vector{Any}(result_table.alg_time[1:data_rows]),
		reslist=Vector{Any}(result_table.res[1:data_rows]),
		objlist=Vector{Any}(result_table.primal_obj[1:data_rows]),
		statuslist=Vector{Any}(result_table.status[1:data_rows]),
		iter4list=Vector{Any}(result_table.iter_4[1:data_rows]),
		time4list=Vector{Any}(result_table.time_4[1:data_rows]),
		iter6list=Vector{Any}(result_table.iter_6[1:data_rows]),
		time6list=Vector{Any}(result_table.time_6[1:data_rows]),
		iter8list=Vector{Any}(result_table.iter_8[1:data_rows]),
		time8list=Vector{Any}(result_table.time_8[1:data_rows]),
	)
end

function build_result_table(state, params::HPRLP.HPRLP_parameters)
	result_table = DataFrame(
		name=state.namelist,
		iter=state.iterlist,
		alg_time=state.timelist,
		res=state.reslist,
		primal_obj=state.objlist,
		status=state.statuslist,
		iter_4=state.iter4list,
		time_4=state.time4list,
		iter_6=state.iter6list,
		time_6=state.time6list,
		iter_8=state.iter8list,
		time_8=state.time8list,
	)

	if isempty(state.timelist)
		return result_table
	end

	geomean_time = exp(mean(log.(state.timelist .+ 10.0))) - 10.0
	geomean_time_4 = exp(mean(log.(state.time4list .+ 10.0))) - 10.0
	geomean_time_6 = exp(mean(log.(state.time6list .+ 10.0))) - 10.0
	geomean_time_8 = exp(mean(log.(state.time8list .+ 10.0))) - 10.0
	geomean_iter = exp(mean(log.(state.iterlist .+ 10.0))) - 10.0
	geomean_iter_4 = exp(mean(log.(state.iter4list .+ 10.0))) - 10.0
	geomean_iter_6 = exp(mean(log.(state.iter6list .+ 10.0))) - 10.0
	geomean_iter_8 = exp(mean(log.(state.iter8list .+ 10.0))) - 10.0

	push!(result_table, [
		"SGM10", geomean_iter, geomean_time, "", "", "",
		geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, geomean_iter_8, geomean_time_8,
	])

	solved = count(x -> x < params.time_limit, state.timelist)
	solved_4 = count(x -> x < params.time_limit, state.time4list)
	solved_6 = count(x -> x < params.time_limit, state.time6list)
	solved_8 = count(x -> x < params.time_limit, state.time8list)
	push!(result_table, ["solved", "", solved, "", "", "", "", solved_4, "", solved_6, "", solved_8])

	return result_table
end

function run_dataset(data_path::String, result_path::String, params::HPRLP.HPRLP_parameters)
	mkpath(result_path)
	files = sort(readdir(data_path))
	csv_file = joinpath(result_path, "HPRLP_result.csv")
	log_path = joinpath(result_path, "HPRLP_log.txt")

	state = initialize_result_lists(csv_file)

	open(log_path, "a") do io
		for (index, file) in enumerate(files)
			if file in state.namelist
				println("The result of problem exists: ", file)
				continue
			end
			if !endswith(lowercase(file), ".mps") && !endswith(lowercase(file), ".mps.gz")
				continue
			end

			file_name = joinpath(data_path, file)
			println(@sprintf("solving the problem %d", index), @sprintf(": %s", file))

			result = redirect_stdout(io) do
				println(@sprintf("solving the problem %d", index), @sprintf(": %s", file))
				println("Solving: ----------------------------------------------------------------------------------------------------------")

				model = read_from_file(file_name)
				set_optimizer(model, HPRLP.Optimizer)
				set_hprlp_attributes!(model, params)
				optimize!(model)

				solver_result = unsafe_backend(model).results

				println("Solve complete ----------------------------------------------------------------------------------------------------------")
				println(
					"iter = ", solver_result.iter,
					@sprintf("  time = %3.2e", solver_result.time),
					@sprintf("  residual = %3.2e", solver_result.residuals),
					@sprintf("  primal_obj = %3.15e", solver_result.primal_obj),
				)

				solver_result
			end

			push!(state.namelist, file)
			push!(state.iterlist, result.iter)
			push!(state.timelist, min(result.time, params.time_limit))
			push!(state.reslist, result.residuals)
			push!(state.objlist, result.primal_obj)
			push!(state.statuslist, result.status)
			push!(state.iter4list, result.iter_4)
			push!(state.time4list, min(result.time_4, params.time_limit))
			push!(state.iter6list, result.iter_6)
			push!(state.time6list, min(result.time_6, params.time_limit))
			push!(state.iter8list, result.iter_8)
			push!(state.time8list, min(result.time_8, params.time_limit))

			CSV.write(csv_file, build_result_table(state, params))
		end
	end

	println("The solver has finished running the dataset, total ", length(files), " problems")
	return nothing
end

params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.warm_up = true
params.use_gpu = true

run_dataset(data_path, result_path, params)

# The results consist of the following files:
# - HPRLP_result.csv: a CSV file containing the results of the experiment
# - HPRLP_log.txt: a log file containing the output of HPRLP