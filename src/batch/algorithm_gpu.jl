function allocate_batched_workspace_gpu(
	shared::BatchedSharedMatrix_gpu,
	batch::BatchedLPData_gpu,
	scaling::BatchedScalingInfo_gpu,
)
	B = batch.batch_size
	m = shared.m
	n = shared.n
	sigma = ifelse.(
		(scaling.norm_b .> 1.0e-8) .& (scaling.norm_c .> 1.0e-8),
		scaling.norm_b ./ scaling.norm_c,
		1.0,
	)

	return BatchedWorkspace_gpu(
		m,
		n,
		B,
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, m, B),
		CUDA.zeros(Float64, n, B),
		CUDA.zeros(Float64, m, B),
		sigma,
		shared.lambda_max,
		CUDA.fill(true, B),
		CUDA.fill(false, B),
		CUDA.zeros(Float64, B),
		CUDA.ones(Float64, B),
	)
end

function initialize_batched_restart_gpu(sigma::CuVector{Float64})
	B = length(sigma)
	sigma_host = Array(sigma)
	return BatchedRestartInfo_gpu(
		zeros(Int, B),
		fill(true, B),
		fill(Inf, B),
		fill(Inf, B),
		fill(Inf, B),
		fill(Inf, B),
		copy(sigma_host),
		zeros(Int, B),
		zeros(Int, B),
		zeros(Int, B),
		zeros(Int, B),
		zeros(Int, B),
		zeros(Float64, B),
		ones(Float64, B),
		fill(false, B),
		sigma_host,
	)
end

function upload_batched_halpern_factors!(ws::BatchedWorkspace_gpu, restart_info::BatchedRestartInfo_gpu)
	for k in 1:ws.B
		restart_info.halpern_fact1_host[k] = 1.0 / (restart_info.inner[k] + 2.0)
		restart_info.halpern_fact2_host[k] = 1.0 - restart_info.halpern_fact1_host[k]
	end
	copyto!(ws.halpern_fact1, restart_info.halpern_fact1_host)
	copyto!(ws.halpern_fact2, restart_info.halpern_fact2_host)
	return nothing
end

function upload_batched_restart_flags!(ws::BatchedWorkspace_gpu, restart_info::BatchedRestartInfo_gpu)
	for k in 1:ws.B
		restart_info.restart_flags_host[k] = restart_info.restart_flag[k] > 0
	end
	copyto!(ws.restart_flags, restart_info.restart_flags_host)
	return any(restart_info.restart_flags_host)
end

function check_batched_restart!(
	restart_info::BatchedRestartInfo_gpu,
	iter::Int,
	check_iter::Int,
	sigma::Vector{Float64},
	active::Vector{Bool},
)
	for k in eachindex(active)
		active[k] || continue
		if restart_info.first_restart[k]
			if iter == check_iter
				restart_info.first_restart[k] = false
				restart_info.restart_flag[k] = 1
				restart_info.best_gap[k] = restart_info.current_gap[k]
				restart_info.best_sigma[k] = sigma[k]
			end
		elseif rem(iter, check_iter) == 0
			if restart_info.current_gap[k] < 0.0
				restart_info.current_gap[k] = 1.0e-6
			end

			if restart_info.current_gap[k] <= 0.2 * restart_info.last_gap[k]
				restart_info.sufficient[k] += 1
				restart_info.restart_flag[k] = 1
			end

			if (restart_info.current_gap[k] <= 0.6 * restart_info.last_gap[k]) && (restart_info.current_gap[k] > restart_info.save_gap[k])
				restart_info.necessary[k] += 1
				restart_info.restart_flag[k] = 2
			end

			if restart_info.inner[k] >= 0.2 * iter
				restart_info.long[k] += 1
				restart_info.restart_flag[k] = 3
			end

			if restart_info.best_gap[k] > restart_info.current_gap[k]
				restart_info.best_gap[k] = restart_info.current_gap[k]
				restart_info.best_sigma[k] = sigma[k]
			end

			restart_info.save_gap[k] = restart_info.current_gap[k]
		end
	end
	return nothing
end

function update_batched_sigma!(
	restart_info::BatchedRestartInfo_gpu,
	ws::BatchedWorkspace_gpu,
	residuals::BatchedResiduals_gpu,
	active::Vector{Bool},
)
	if !any(flag -> 1 <= flag <= 3, restart_info.restart_flag)
		return nothing
	end
	primal_move, dual_move = compute_batched_restart_movement_norms_gpu!(ws)
	err_Rd = Array(residuals.err_Rd)
	err_Rp = Array(residuals.err_Rp)
	rel_gap = Array(residuals.rel_gap)
	sqrt_lambda = sqrt(ws.lambda_max)
	for k in eachindex(active)
		active[k] || continue
		if 1 <= restart_info.restart_flag[k] <= 3
			if primal_move[k] > 1.0e-16 && dual_move[k] > 1.0e-16 && primal_move[k] < 1.0e12 && dual_move[k] < 1.0e12
				ratio = (primal_move[k] / dual_move[k]) / sqrt_lambda
				fact = exp(-0.05 * (restart_info.current_gap[k] / restart_info.best_gap[k]))
				temp_1 = max(min(err_Rd[k], err_Rp[k]), min(rel_gap[k], restart_info.current_gap[k]))
				sigma_cand = exp(fact * log(ratio) + (1.0 - fact) * log(restart_info.best_sigma[k]))
				if temp_1 > 9.0e-10
					κ = 1.0
				elseif temp_1 > 5.0e-10
					κ = clamp(sqrt(err_Rd[k] / err_Rp[k]), 1.0e-2, 100.0)
				else
					κ = clamp(err_Rd[k] / err_Rp[k], 1.0e-2, 100.0)
				end
				restart_info.sigma_host[k] = κ * sigma_cand
			else
				restart_info.sigma_host[k] = 1.0
			end
		end
	end
	copyto!(ws.sigma, restart_info.sigma_host)
	return nothing
end

function do_batched_restart!(restart_info::BatchedRestartInfo_gpu, ws::BatchedWorkspace_gpu, active::Vector{Bool})
	any_restart = upload_batched_restart_flags!(ws, restart_info)
	if any_restart
		do_batched_restart_gpu!(ws)
		for k in eachindex(active)
			if active[k] && restart_info.restart_flag[k] > 0
				restart_info.times[k] += 1
				restart_info.inner[k] = 0
				restart_info.save_gap[k] = Inf
			end
		end
	end
	return any_restart
end

function advance_batched_inner!(restart_info::BatchedRestartInfo_gpu, active::Vector{Bool})
	for k in eachindex(active)
		if active[k]
			restart_info.inner[k] += 1
		end
	end
	return nothing
end

function batched_warmup_params(params::HPRLP_parameters)
	warmup_params = deepcopy(params)
	warmup_params.max_iter = 200
	warmup_params.verbose = false
	warmup_params.warm_up = false
	return warmup_params
end

function print_batched_warmup_start(params::HPRLP_parameters)
	if params.verbose
		println("="^80)
		println("BATCHED WARM UP PHASE")
		println("  Running batched warmup to avoid Julia/CUDA JIT overhead in reported timings")
		println("="^80)
	end
	return nothing
end

function print_batched_warmup_finish(params::HPRLP_parameters, warmup_time::Float64)
	if params.verbose
		println(@sprintf("Batched warmup time: %.2f seconds", warmup_time))
		println("="^80)
		println()
	end
	return nothing
end

function prepare_batched_gpu_problem(
	A::Union{SparseMatrixCSC,Matrix},
	C::AbstractMatrix{<:Real},
	AL::AbstractMatrix{<:Real},
	AU::AbstractMatrix{<:Real},
	L::AbstractMatrix{<:Real},
	U::AbstractMatrix{<:Real},
	params::HPRLP_parameters;
	obj_constants::AbstractVector{<:Real}=zeros(size(C, 2)),
)
	shared = build_batched_shared_matrix_gpu(A, params)
	batch, scaling = build_batched_lp_gpu(
		shared,
		C,
		AL,
		AU,
		L,
		U;
		obj_constants=obj_constants,
		use_bc_scaling=params.use_bc_scaling,
	)
	ws = allocate_batched_workspace_gpu(shared, batch, scaling)
	return shared, batch, scaling, ws
end

function _optimize_batched_gpu_solve(
	shared::BatchedSharedMatrix_gpu,
	batch::BatchedLPData_gpu,
	scaling::BatchedScalingInfo_gpu,
	params::HPRLP_parameters,
)
	params.use_gpu || throw(ArgumentError("optimize_batched_gpu requires params.use_gpu=true."))
	normalize_presolve_backend(params.presolve) == "NONE" || throw(ArgumentError("optimize_batched_gpu currently supports presolve = \"NONE\" only."))

	ws = allocate_batched_workspace_gpu(shared, batch, scaling)
	residuals = allocate_batched_residuals_gpu(batch.batch_size)
	restart_info = initialize_batched_restart_gpu(ws.sigma)
	status = fill("CONTINUE", batch.batch_size)
	final_iter = fill(params.max_iter, batch.batch_size)
	active = fill(true, batch.batch_size)
	check_iter = max(params.check_iter, 1)
	t_start = time()
	setup_time = 0.0
	power_time = 0.0

	for iter in 0:params.max_iter
		periodic_check = rem(iter, check_iter) == 0
		elapsed = time() - t_start
		print_yes = should_print_log(iter, params.max_iter, params.print_frequency, elapsed, params.time_limit)
		residual_check = periodic_check || print_yes
		if residual_check
			if periodic_check && iter > 0
				restart_info.current_gap .= compute_batched_weighted_norm_gpu!(ws, shared)
			end
			compute_batched_residuals_gpu!(residuals, ws, shared, batch, scaling, iter)
			kkt_error = Array(residuals.kkt_error)
			for k in eachindex(kkt_error)
				if active[k] && kkt_error[k] <= params.stoptol
					status[k] = "OPTIMAL"
					final_iter[k] = iter
					active[k] = false
				end
			end
			copyto!(ws.active, active)
		end
		if all(!=("CONTINUE"), status)
			return collect_batched_results_gpu(ws, residuals, scaling, status, final_iter, t_start, setup_time, power_time)
		end

		if iter >= params.max_iter || elapsed >= params.time_limit
			final_status = elapsed >= params.time_limit ? "TIME_LIMIT" : "ITER_LIMIT"
			for k in eachindex(status)
				if status[k] == "CONTINUE"
					status[k] = final_status
					final_iter[k] = iter
					active[k] = false
				end
			end
			copyto!(ws.active, active)
			return collect_batched_results_gpu(ws, residuals, scaling, status, final_iter, t_start, setup_time, power_time)
		end

		restart_info.restart_flag .= 0
		if periodic_check
			copyto!(restart_info.sigma_host, Array(ws.sigma))
			check_batched_restart!(restart_info, iter, check_iter, restart_info.sigma_host, active)
		end

		update_batched_sigma!(restart_info, ws, residuals, active)
		restarted = do_batched_restart!(restart_info, ws, active)

		to_check = rem(iter + 1, check_iter) == 0 || restarted
		if params.print_frequency == -1
			to_check = to_check || rem(iter + 1, print_step(iter + 1)) == 0
		elseif params.print_frequency > 0
			to_check = to_check || rem(iter + 1, params.print_frequency) == 0
		end
		upload_batched_halpern_factors!(ws, restart_info)
		if to_check
			update_x_z_check_batched_gpu!(ws, shared, batch)
			update_y_check_batched_gpu!(ws, shared, batch)
		else
			update_x_z_normal_batched_gpu!(ws, shared, batch)
			update_y_normal_batched_gpu!(ws, shared, batch)
		end
		advance_batched_inner!(restart_info, active)
		if restarted
			last_gap = compute_batched_weighted_norm_gpu!(ws, shared)
			for k in eachindex(active)
				if restart_info.restart_flag[k] > 0
					restart_info.last_gap[k] = last_gap[k]
				end
			end
		end
	end

	return collect_batched_results_gpu(ws, residuals, scaling, status, final_iter, t_start, setup_time, power_time)
end

function optimize_batched_gpu(
	shared::BatchedSharedMatrix_gpu,
	batch::BatchedLPData_gpu,
	scaling::BatchedScalingInfo_gpu,
	params::HPRLP_parameters,
)
	params.use_gpu || throw(ArgumentError("optimize_batched_gpu requires params.use_gpu=true."))
	normalize_presolve_backend(params.presolve) == "NONE" || throw(ArgumentError("optimize_batched_gpu currently supports presolve = \"NONE\" only."))

	if params.warm_up
		print_batched_warmup_start(params)
		original_lambda_max = shared.lambda_max
		warmup_params = batched_warmup_params(params)
		warmup_time = @elapsed _optimize_batched_gpu_solve(shared, batch, scaling, warmup_params)
		shared.lambda_max = original_lambda_max
		print_batched_warmup_finish(params, warmup_time)
	end

	return _optimize_batched_gpu_solve(shared, batch, scaling, params)
end

function optimize_batched_gpu(
	A::Union{SparseMatrixCSC,Matrix},
	C::AbstractMatrix{<:Real},
	AL::AbstractMatrix{<:Real},
	AU::AbstractMatrix{<:Real},
	L::AbstractMatrix{<:Real},
	U::AbstractMatrix{<:Real},
	params::HPRLP_parameters;
	obj_constants::AbstractVector{<:Real}=zeros(size(C, 2)),
)
	if params.warm_up
		print_batched_warmup_start(params)
		warmup_params = batched_warmup_params(params)
		warmup_time = @elapsed begin
			warmup_shared, warmup_batch, warmup_scaling, _ = prepare_batched_gpu_problem(
				A,
				C,
				AL,
				AU,
				L,
				U,
				warmup_params;
				obj_constants=obj_constants,
			)
			_optimize_batched_gpu_solve(warmup_shared, warmup_batch, warmup_scaling, warmup_params)
		end
		CUDA.reclaim()
		print_batched_warmup_finish(params, warmup_time)
	end

	setup_time = @elapsed shared, batch, scaling, _ = prepare_batched_gpu_problem(
		A,
		C,
		AL,
		AU,
		L,
		U,
		params;
		obj_constants=obj_constants,
	)
	results = _optimize_batched_gpu_solve(shared, batch, scaling, params)
	results.setup_time += setup_time
	results.time += setup_time
	return results
end
