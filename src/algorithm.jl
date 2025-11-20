
# This file is the main file for the package HPR-LP (release version).

# The package is used to solve linear programming (LP) with HPR method in the paper HPR-LP: An implementation of an HPR method for
# solving linear programming.
# The package is developed by Kaihuang Chen · Defeng Sun · Yancheng Yuan · Guojun Zhang · Xinyuan Zhao.
# The format of the linear programming problem is as follows:
# min <c,x>
# s.t. AL <= Ax <= AU
#       l <= x <= u



#=
HPR-LP Solver: Code Overview

This solver is designed for GPU-accelerated large-scale LP problems, using adaptive restarts, dynamic step-size updates, 
    and efficient memory management to improve convergence speed.

Key Components:

1. Main Functions
	Problem Setup (in utils.jl)
		•   formulation(lp): Converts LP problem into the form described in the paper.
		•   scaling!(lp, ...): Applies scaling techniques for stability.
		•   power_iteration_gpu(A, AT): Estimates largest eigenvalue (λ_max).
	Solver Updates (GPU Kernels)
		•   update_x_z_gpu!(), update_y_gpu!(): Updates primal/dual variables.
		•   compute_Rd_gpu!(), compute_err_Rp_gpu!(): Computes residuals.
		•   Halpern_update_gpu!(): Halpern iteration.
	Convergence & Restart
		•   check_break(): Stop conditions (OPTIMAL, TIME_LIMIT).
		•   check_restart(), do_restart(): Adaptive restart mechanism.
		•   update_sigma(): Dynamic penalty paramter adjustment (σ).
	Result Collection
		•   collect_results_gpu!(): Saves final solution values.

2. Execution Flow (solve(lp, scaling_info, params))
The solve function is the core solver, executing the following steps:
	1.	Power iteration to estimate λ_max.
	2.	Initialize residuals, restart conditions, and GPU workspace.
	3.	Iterative solver updates:
	    •	Compute residuals (compute_residuals)
	    •	Check stopping criteria (check_break)
	    •	Apply restart conditions (check_restart) and adjust penalty parameter (update_sigma)
	    •	Update primal (update_x_z_gpu!) and dual (update_y_gpu!) variables
	    •	Perform Halpern iteration (Halpern_update_gpu!)
	4.	Check convergence and return results.

=#

# the function to compute the M norm 
function compute_weighted_norm_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_dx, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
        ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    dot_prod = 2 * CUDA.dot(ws.Ax, ws.dy)
    dy_squarenorm = CUDA.dot(ws.dy, ws.dy)
    dx_squarenorm = CUDA.dot(ws.dx, ws.dx)
    weighted_norm = ws.sigma * (ws.lambda_max * dy_squarenorm) + (dx_squarenorm) / ws.sigma + dot_prod
    if weighted_norm < 0
        println("The estimated maximum eigenvalue is too small! Current value is ", ws.lambda_max)
        ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
        println("The new estimated maximum eigenvalue is ", ws.lambda_max)
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    return weighted_norm
end

function compute_weighted_norm_cpu!(ws::HPRLP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.dx)
    dot_prod = 2 * dot(ws.Ax, ws.dy)
    dy_squarenorm = dot(ws.dy, ws.dy)
    dx_squarenorm = dot(ws.dx, ws.dx)
    weighted_norm = ws.sigma * (ws.lambda_max * dy_squarenorm) + (dx_squarenorm) / ws.sigma + dot_prod
    if weighted_norm < 0
        println("The estimated value of lambda_max is too small! Please increase params.lambda_factor!")
        ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    return weighted_norm
end

# the Halpern iteration, Step 10 in Algorithm 2
function Halpern_update_gpu!(ws::HPRLP_workspace_gpu, restart_info::HPRLP_restart)
    fact1 = 1.0 / (restart_info.inner + 2.0)
    fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
    axpby_gpu!(fact1, ws.last_x, fact2, ws.x_hat, ws.x, ws.n)
    axpby_gpu!(fact1, ws.last_y, fact2, ws.y_hat, ws.y, ws.m)
    restart_info.inner += 1
end

function Halpern_update_cpu!(ws::HPRLP_workspace_cpu, restart_info::HPRLP_restart)
    fact1 = 1.0 / (restart_info.inner + 2.0)
    fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
    ws.x .= fact1 .* ws.last_x .+ fact2 .* ws.x_hat
    ws.y .= fact1 .* ws.last_y .+ fact2 .* ws.y_hat
    restart_info.inner += 1
end

# the function to compute the residuals for the original LP problem
function compute_residuals_gpu!(ws::HPRLP_workspace_gpu,
    lp::LP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRLP_residuals,
    iter::Int,
)
    ### obj
    scbc = sc.b_scale * sc.c_scale
    res.primal_obj_bar = scbc * CUDA.dot(ws.c, ws.x_bar) + lp.obj_constant
    res.dual_obj_bar = scbc * (CUDA.dot(ws.y_obj, ws.y_bar) + CUDA.dot(ws.x_bar, ws.z_bar)) + lp.obj_constant
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0 + abs(res.primal_obj_bar) + abs(res.dual_obj_bar))

    ### Rd
    compute_Rd_gpu!(ws, sc)
    res.err_Rd_org_bar = sc.c_scale * CUDA.norm(ws.Rd) / sc.norm_c_org

    ### Rp
    compute_err_Rp_gpu!(ws, sc)
    res.err_Rp_org_bar = sc.b_scale * CUDA.norm(ws.Rp) / sc.norm_b_org

    if iter == 0
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_err_lu_kernel!(sc.col_norm, ws.dx, ws.x_bar, ws.l, ws.u, ws.n)
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, sc.b_scale * CUDA.norm(ws.dx))
    end
    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
end

# the function to compute the residuals for the original LP problem
function compute_residuals_cpu!(ws::HPRLP_workspace_cpu,
    lp::LP_info_cpu,
    sc::Scaling_info_cpu,
    res::HPRLP_residuals,
    iter::Int
)
    ### obj
    scbc = sc.b_scale * sc.c_scale
    res.primal_obj_bar = scbc * dot(ws.c, ws.x_bar) + lp.obj_constant
    res.dual_obj_bar = scbc * (dot(ws.y_obj, ws.y_bar) + dot(ws.x_bar, ws.z_bar)) + lp.obj_constant
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0 + abs(res.primal_obj_bar) + abs(res.dual_obj_bar))

    ### Rd
    compute_err_Rd_cpu!(ws, sc)
    res.err_Rd_org_bar = sc.c_scale * norm(ws.Rd) / sc.norm_c_org

    ### Rp
    compute_err_Rp_cpu!(ws, sc)
    res.err_Rp_org_bar = sc.b_scale * norm(ws.Rp) / sc.norm_b_org

    if iter == 0
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, sc.b_scale * norm((ws.x_bar - max.(min.(ws.x_bar, ws.u), ws.l)) ./ sc.col_norm))
    end

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)
end


# the function to update the value of sigma
function update_sigma_gpu!(
    restart_info::HPRLP_restart,
    ws::HPRLP_workspace_gpu,
    residuals::HPRLP_residuals,
)
    if restart_info.restart_flag >= 1 && restart_info.restart_flag <= 3
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
        axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
        primal_move = CUDA.norm(ws.dx)
        dual_move = CUDA.norm(ws.dy)
        if primal_move > 1e-16 && dual_move > 1e-16 &&
           primal_move < 1e12 && dual_move < 1e12
            pm_over_dm = primal_move / dual_move
            sqrtλ = sqrt(ws.lambda_max)
            ratio = pm_over_dm / sqrtλ
            fact = exp(-0.05 * (restart_info.current_gap / restart_info.best_gap))
            temp_1 = max(min(residuals.err_Rd_org_bar, residuals.err_Rp_org_bar), min(residuals.rel_gap_bar, restart_info.current_gap))
            sigma_cand = exp(fact * log(ratio) + (1 - fact) * log(restart_info.best_sigma))
            if temp_1 > 9e-10
                κ = 1.0
            elseif temp_1 > 5e-10
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp(sqrt(ratio_infeas_org), 1e-2, 100.0)
            else
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp((ratio_infeas_org), 1e-2, 100.0)
            end
            ws.sigma = κ * sigma_cand
        else
            ws.sigma = 1.0
        end
    end
end

function update_sigma_cpu!(
    restart_info::HPRLP_restart,
    ws::HPRLP_workspace_cpu,
    residuals::HPRLP_residuals,
)
    if restart_info.restart_flag >= 1 && restart_info.restart_flag <= 3
        ws.dx .= ws.x_bar .- ws.last_x
        ws.dy .= ws.y_bar .- ws.last_y
        primal_move = norm(ws.dx)
        dual_move = norm(ws.dy)
        if primal_move > 1e-16 && dual_move > 1e-16 &&
           primal_move < 1e12 && dual_move < 1e12
            pm_over_dm = primal_move / dual_move
            sqrtλ = sqrt(ws.lambda_max)
            ratio = pm_over_dm / sqrtλ
            fact = exp(-0.05 * (restart_info.current_gap / restart_info.best_gap))
            temp_1 = max(min(residuals.err_Rd_org_bar, residuals.err_Rp_org_bar), min(residuals.rel_gap_bar, restart_info.current_gap))
            sigma_cand = exp(fact * log(ratio) + (1 - fact) * log(restart_info.best_sigma))
            if temp_1 > 9e-10
                κ = 1.0
            elseif temp_1 > 5e-10
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp(sqrt(ratio_infeas_org), 1e-2, 100.0)
            else
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp((ratio_infeas_org), 1e-2, 100.0)
            end
            ws.sigma = κ * sigma_cand
        else
            ws.sigma = 1.0
        end
    end
end


# the function to check whether to restart the algorithm
function check_restart(restart_info::HPRLP_restart,
    iter::Int,
    check_iter::Int, sigma::Float64,
)

    restart_info.restart_flag = 0
    # adaptive restart
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.best_gap = restart_info.current_gap
            restart_info.best_sigma = sigma
        end
    else
        if rem(iter, check_iter) == 0
            if restart_info.current_gap < 0
                restart_info.current_gap = 1e-6
                println("current_gap < 0")
            end

            # sufficient decrease
            if restart_info.current_gap <= 0.2 * restart_info.last_gap
                restart_info.sufficient += 1
                restart_info.restart_flag = 1
            end

            # necessary decrease
            if (restart_info.current_gap <= 0.6 * restart_info.last_gap) && (restart_info.current_gap > 1.00 * restart_info.save_gap)
                restart_info.necessary += 1
                restart_info.restart_flag = 2
            end

            # long iterations
            if restart_info.inner >= 0.2 * iter
                restart_info.long += 1
                restart_info.restart_flag = 3
            end

            if restart_info.best_gap > restart_info.current_gap
                restart_info.best_gap = restart_info.current_gap
                restart_info.best_sigma = sigma
            end

            restart_info.save_gap = restart_info.current_gap
        end
    end
end

# the function to do the restart
function do_restart!(restart_info::HPRLP_restart, ws::Union{HPRLP_workspace_gpu,HPRLP_workspace_cpu})
    if restart_info.restart_flag > 0
        ws.x .= ws.x_bar
        ws.y .= ws.y_bar
        ws.last_x .= ws.x_bar
        ws.last_y .= ws.y_bar
        restart_info.times += 1
        restart_info.inner = 0
        restart_info.save_gap = Inf
    end
end

# the function to check whether to stop the algorithm
function check_break(residuals::HPRLP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRLP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

# the function to collect the results
function collect_results_gpu!(
    ws::HPRLP_workspace_gpu,
    residuals::HPRLP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
    status::String,
    tolerance_times::Vector{Float64},
    tolerance_iters::Vector{Int}
)
    results = HPRLP_results()
    results.x = CuVector{Float64}(undef, ws.n)
    results.y = CuVector{Float64}(undef, ws.m)
    results.z = CuVector{Float64}(undef, ws.n)
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    ### copy the results to the CPU ### 
    results.x .= Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y .= Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z .= Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))

    results.output_type = status
    # Set tolerance results, using final values if threshold not reached
    results.time_4 = tolerance_times[1] == 0.0 ? results.time : tolerance_times[1]
    results.iter_4 = tolerance_iters[1] == 0 ? iter : tolerance_iters[1]
    results.time_6 = tolerance_times[2] == 0.0 ? results.time : tolerance_times[2]
    results.iter_6 = tolerance_iters[2] == 0 ? iter : tolerance_iters[2]
    results.time_8 = tolerance_times[3] == 0.0 ? results.time : tolerance_times[3]
    results.iter_8 = tolerance_iters[3] == 0 ? iter : tolerance_iters[3]
    return results
end

function collect_results_cpu!(
    ws::HPRLP_workspace_cpu,
    residuals::HPRLP_residuals,
    sc::Scaling_info_cpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
    status::String,
    tolerance_times::Vector{Float64},
    tolerance_iters::Vector{Int}
)
    results = HPRLP_results()
    results.x = Vector{Float64}(undef, ws.n)
    results.y = Vector{Float64}(undef, ws.m)
    results.z = Vector{Float64}(undef, ws.n)
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    results.x .= sc.b_scale * (ws.x_bar ./ sc.col_norm)
    results.y .= sc.c_scale * (ws.y_bar ./ sc.row_norm)
    results.z .= sc.c_scale * (ws.z_bar .* sc.col_norm)

    results.output_type = status
    # Set tolerance results, using final values if threshold not reached
    results.time_4 = tolerance_times[1] == 0.0 ? results.time : tolerance_times[1]
    results.iter_4 = tolerance_iters[1] == 0 ? iter : tolerance_iters[1]
    results.time_6 = tolerance_times[2] == 0.0 ? results.time : tolerance_times[2]
    results.iter_6 = tolerance_iters[2] == 0 ? iter : tolerance_iters[2]
    results.time_8 = tolerance_times[3] == 0.0 ? results.time : tolerance_times[3]
    results.iter_8 = tolerance_iters[3] == 0 ? iter : tolerance_iters[3]
    return results
end


# the function to prepare the spmv for a given sparse matrix A
function prepare_spmv!(A::CuSparseMatrixCSR{Float64,Int32}, AT::CuSparseMatrixCSR{Float64,Int32},
    x_bar::CuVector{Float64}, x_hat::CuVector{Float64}, dx::CuVector{Float64}, Ax::CuVector{Float64},
    y_bar::CuVector{Float64}, y::CuVector{Float64}, ATy::CuVector{Float64})
    desc_A = CUDA.CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    desc_x_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_bar)
    desc_x_hat = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_hat)
    desc_dx = CUDA.CUSPARSE.CuDenseVectorDescriptor(dx)
    desc_Ax = CUDA.CUSPARSE.CuDenseVectorDescriptor(Ax)
    desc_AT = CUDA.CUSPARSE.CuSparseMatrixDescriptor(AT, 'O')
    desc_y_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(y_bar)
    desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    desc_ATy = CUDA.CUSPARSE.CuDenseVectorDescriptor(ATy)
    CUSPARSE_handle = CUDA.CUSPARSE.handle()
    sz_A = Ref{Csize_t}(0)
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero,
        desc_Ax, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_A)

    buf_A = CUDA.CuArray{UInt8}(undef, sz_A[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero, desc_Ax,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)
    end

    spmv_A = CUSPARSE_spmv_A(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, desc_x_hat, desc_dx, ref_zero, desc_Ax,
        Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)

    sz_AT = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero,
        desc_ATy, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_AT)
    buf_AT = CUDA.CuArray{UInt8}(undef, sz_AT[])
    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero, desc_ATy,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)
    end
    spmv_AT = CUSPARSE_spmv_AT(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, desc_y, ref_zero, desc_ATy,
        Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)

    return spmv_A, spmv_AT
end

# the function to allocate the workspace for the HPR-LP algorithm
function allocate_workspace_gpu(lp::LP_info_gpu, scaling_info::Scaling_info_gpu)
    ws = HPRLP_workspace_gpu()
    m, n = size(lp.A)
    ws.m = m
    ws.n = n
    ws.x = CUDA.zeros(Float64, n)
    ws.x_hat = CUDA.zeros(Float64, n)
    ws.x_bar = CUDA.zeros(Float64, n)
    ws.dx = CUDA.zeros(Float64, n)
    ws.y = CUDA.zeros(Float64, m)
    ws.y_hat = CUDA.zeros(Float64, m)
    ws.y_bar = CUDA.zeros(Float64, m)
    ws.dy = CUDA.zeros(Float64, m)
    ws.y_obj = CUDA.zeros(Float64, m)
    ws.z_bar = CUDA.zeros(Float64, n)
    ws.A = lp.A
    ws.AT = lp.AT
    ws.AL = lp.AL
    ws.AU = lp.AU
    ws.c = lp.c
    ws.l = lp.l
    ws.u = lp.u
    ws.Rp = CUDA.zeros(Float64, m)
    ws.Rd = CUDA.zeros(Float64, n)
    ws.ATy = CUDA.zeros(Float64, n)
    ws.Ax = CUDA.zeros(Float64, m)
    ws.last_x = CUDA.zeros(Float64, n)
    ws.last_y = CUDA.zeros(Float64, m)
    ws.to_check = false
    ws.spmv_A, ws.spmv_AT = prepare_spmv!(lp.A, lp.AT, ws.x_bar, ws.x_hat, ws.dx, ws.Ax,
        ws.y_bar, ws.y, ws.ATy)
    if scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8
        ws.sigma = scaling_info.norm_b / scaling_info.norm_c
    else
        ws.sigma = 1.0
    end
    return ws
end

function allocate_workspace_cpu(lp::LP_info_cpu, scaling_info::Scaling_info_cpu)
    ws = HPRLP_workspace_cpu()
    m, n = size(lp.A)
    ws.m = m
    ws.n = n
    ws.x = Vector(zeros(n))
    ws.x_hat = Vector(zeros(n))
    ws.x_bar = Vector(zeros(n))
    ws.dx = Vector(zeros(n))
    ws.y = Vector(zeros(m))
    ws.y_hat = Vector(zeros(m))
    ws.y_bar = Vector(zeros(m))
    ws.y_obj = Vector(zeros(m))
    ws.dy = Vector(zeros(m))
    ws.z_bar = Vector(zeros(n))
    ws.A = lp.A
    ws.AT = lp.AT
    ws.AL = lp.AL
    ws.AU = lp.AU
    ws.c = lp.c
    ws.l = lp.l
    ws.u = lp.u
    ws.Rp = Vector(zeros(m))
    ws.Rd = Vector(zeros(n))
    ws.ATy = Vector(zeros(n))
    ws.Ax = Vector(zeros(m))
    ws.last_x = Vector(zeros(n))
    ws.last_y = Vector(zeros(m))
    ws.to_check = false
    if scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8
        ws.sigma = scaling_info.norm_b / scaling_info.norm_c
    else
        ws.sigma = 1.0
    end
    return ws
end

# the function to initialize the restart information
function initialize_restart(sigma::Float64)
    restart_info = HPRLP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.best_gap = Inf
    restart_info.best_sigma = sigma
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.restart_flag = 0
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end

function compute_maximum_eigenvalue!(lp::Union{LP_info_gpu,LP_info_cpu},
    ws::Union{HPRLP_workspace_gpu,HPRLP_workspace_cpu},
    params::HPRLP_parameters)
    t_start_power = time()
    if params.verbose
        println("ESTIMATING MAXIMUM EIGENVALUE ...")
    end
    if params.use_gpu
        CUDA.synchronize()
        lambda_max = power_iteration_gpu(ws.spmv_A, ws.spmv_AT, ws.m, ws.n) * 1.01
        CUDA.synchronize()
    else
        lambda_max = power_iteration_cpu(lp.A, lp.AT) * 1.01
    end
    power_time = time() - t_start_power
    if params.verbose
        println(@sprintf("ESTIMATING MAXIMUM EIGENVALUE time = %.2f seconds", power_time))
        println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max))
    end
    ws.lambda_max = lambda_max

    return power_time
end

# The main function for the HPR-LP algorithm
function solve(lp::Union{LP_info_gpu,LP_info_cpu},
    scaling_info::Union{Scaling_info_gpu,Scaling_info_cpu},
    params::HPRLP_parameters)

    if params.verbose
        println("HPR-LP version v0.1.3")
    end
    t_start_alg = time()

    ### Initialization ###
    residuals = HPRLP_residuals()
    ws = params.use_gpu ? allocate_workspace_gpu(lp, scaling_info) : allocate_workspace_cpu(lp, scaling_info)
    restart_info = initialize_restart(ws.sigma)

    ### power iteration to estimate lambda_max ###
    power_time = compute_maximum_eigenvalue!(lp, ws, params)

    if params.verbose
        println(" iter     errRp        errRd         p_obj            d_obj          gap         sigma       time")
    end

    # Track when tolerance thresholds are reached
    tolerance_levels = [1e-4, 1e-6, 1e-8]
    tolerance_times = zeros(Float64, length(tolerance_levels))
    tolerance_iters = zeros(Int, length(tolerance_levels))
    tolerance_reached = falses(length(tolerance_levels))

    if params.use_gpu
        compute_residuals! = compute_residuals_gpu!
        update_sigma! = update_sigma_gpu!
        collect_results! = collect_results_gpu!
        update_x_z! = update_x_z_gpu!
        update_y! = update_y_gpu!
        compute_weighted_norm! = compute_weighted_norm_gpu!
    else
        compute_residuals! = compute_residuals_cpu!
        update_sigma! = update_sigma_cpu!
        collect_results! = collect_results_cpu!
        update_x_z! = update_x_z_cpu!
        update_y! = update_y_cpu!
        compute_weighted_norm! = compute_weighted_norm_cpu!
    end

    for iter = 0:params.max_iter
        ### whether to print the log ###
        if params.print_frequency == -1
            print_yes = ((rem(iter, print_step(iter)) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        elseif params.print_frequency > 0
            print_yes = ((rem(iter, params.print_frequency) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        else
            error("Invalid print_frequency: ", params.print_frequency, ". It should be a positive integer or -1 for automatic printing.")
        end

        ### compute residuals ###
        if rem(iter, params.check_iter) == 0 || print_yes
            compute_residuals!(ws, lp, scaling_info, residuals, iter)
        end

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        check_restart(restart_info, iter, params.check_iter, ws.sigma)

        ### print the log ###
        if print_yes || (status != "CONTINUE")
            if params.verbose
                println(@sprintf("%5.0f    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f",
                    iter,
                    residuals.err_Rp_org_bar,
                    residuals.err_Rd_org_bar,
                    residuals.primal_obj_bar,
                    residuals.dual_obj_bar,
                    residuals.rel_gap_bar,
                    ws.sigma,
                    time() - t_start_alg))
            end
        end

        ### collect results and return ###
        # Check tolerance thresholds
        for i in eachindex(tolerance_levels)
            if !tolerance_reached[i] && residuals.KKTx_and_gap_org_bar < tolerance_levels[i]
                tolerance_times[i] = time() - t_start_alg
                tolerance_iters[i] = iter
                tolerance_reached[i] = true
                if params.verbose
                    println("KKT < ", tolerance_levels[i], " at iter = ", iter)
                end
            end
        end

        if status != "CONTINUE"
            if params.verbose
                println("Termination reason: ", status, ", accuracy = ", residuals.KKTx_and_gap_org_bar)
            end
            results = collect_results!(ws, residuals, scaling_info, iter, t_start_alg, power_time, status, tolerance_times, tolerance_iters)
            return results
        end

        ### update sigma ###
        update_sigma!(restart_info, ws, residuals)

        ### restart if needed ###
        do_restart!(restart_info, ws)

        ## whether to compute bar points for residuals ##
        ws.to_check = (rem(iter + 1, params.check_iter) == 0) || (restart_info.restart_flag > 0)
        if params.print_frequency == -1
            ws.to_check = ws.to_check || (rem(iter + 1, print_step(iter + 1)) == 0)
        elseif params.print_frequency > 0
            ws.to_check = ws.to_check || (rem(iter + 1, params.print_frequency) == 0)
        end

        ### update x, y,  and z ###
        fact1 = 1.0 / (restart_info.inner + 2.0)
        fact2 = 1.0 - fact1
        update_x_z!(ws, fact1, fact2)
        update_y!(ws, fact1, fact2)
        restart_info.inner += 1 

        ### compute weighted norm ###
        if rem(iter + 1, params.check_iter) == 0
            restart_info.current_gap = compute_weighted_norm!(ws)
        end
        if restart_info.restart_flag > 0
            restart_info.last_gap = compute_weighted_norm!(ws)
        end
    end
end
