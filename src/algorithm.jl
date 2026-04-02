
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
@inline function queue_dot!(buf::CuVector{Float64}, idx::Int, x::CuVector{Float64}, y::CuVector{Float64})
    CUDA.CUBLAS.dot(length(x), x, y, CUDA.CuRefArray(buf, idx))
    return nothing
end

@inline function queue_nrm2!(buf::CuVector{Float64}, idx::Int, x::CuVector{Float64})
    CUDA.CUBLAS.nrm2(length(x), x, CUDA.CuRefArray(buf, idx))
    return nothing
end

@inline function fetch_reduction_scalars!(ws::HPRLP_workspace_gpu)
    host = ws.reduction_scalars_host
    dev = ws.reduction_scalars
    n = length(host)
    GC.@preserve host dev begin
        # Keep identical host-visible semantics (values ready on return) while
        # avoiding the extra pre-sync used by generic copyto!(Array, CuArray).
        unsafe_copyto!(pointer(host), pointer(dev), n; async=false)
    end
    return host
end

@inline function compute_restart_movement_norms_gpu!(ws::HPRLP_workspace_gpu)
    axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
    axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
    queue_nrm2!(ws.reduction_scalars, 9, ws.dx)
    queue_nrm2!(ws.reduction_scalars, 10, ws.dy)
    fetch_reduction_scalars!(ws)
    return nothing
end

function compute_weighted_norm_gpu!(ws::HPRLP_workspace_gpu)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_dx, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
        ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    queue_dot!(ws.reduction_scalars, 1, ws.Ax, ws.dy)
    queue_dot!(ws.reduction_scalars, 2, ws.dy, ws.dy)
    queue_dot!(ws.reduction_scalars, 3, ws.dx, ws.dx)
    reduction_scalars_host = fetch_reduction_scalars!(ws)
    dot_prod = 2 * reduction_scalars_host[1]
    dy_squarenorm = reduction_scalars_host[2]
    dx_squarenorm = reduction_scalars_host[3]
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

@inline function queue_pending_last_gap_gpu!(ws::HPRLP_workspace_gpu)
    # Reuse reduction slots 11:13 for deferred restart-gap terms.
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_dx, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
        ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    queue_dot!(ws.reduction_scalars, 11, ws.Ax, ws.dy)
    queue_dot!(ws.reduction_scalars, 12, ws.dy, ws.dy)
    queue_dot!(ws.reduction_scalars, 13, ws.dx, ws.dx)
    ws.pending_last_gap_sigma = ws.sigma
    ws.pending_last_gap_lambda_max = ws.lambda_max
    ws.pending_last_gap = true
    return nothing
end

@inline function consume_pending_last_gap_gpu!(ws::HPRLP_workspace_gpu, restart_info::HPRLP_restart)
    if !ws.pending_last_gap
        return nothing
    end
    reduction_scalars_host = ws.reduction_scalars_host
    sigma = ws.pending_last_gap_sigma
    lambda_max = ws.pending_last_gap_lambda_max
    dot_prod = 2 * reduction_scalars_host[11]
    dy_squarenorm = reduction_scalars_host[12]
    dx_squarenorm = reduction_scalars_host[13]
    weighted_norm = sigma * (lambda_max * dy_squarenorm) + (dx_squarenorm) / sigma + dot_prod
    if weighted_norm < 0
        println("The estimated maximum eigenvalue is too small! Current value is ", lambda_max)
        ws.lambda_max = -(dot_prod + (dx_squarenorm) / sigma) / (sigma * (dy_squarenorm)) * 1.05
        println("The new estimated maximum eigenvalue is ", ws.lambda_max)
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    restart_info.last_gap = weighted_norm
    ws.pending_last_gap = false
    return nothing
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

function compute_original_kkt_metrics(
    model::LP_info_cpu,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    z::AbstractVector{<:Real},
)
    xh = x isa Vector{Float64} ? x : Array(x)
    yh = y isa Vector{Float64} ? copy(y) : Array(y)
    zh = z isa Vector{Float64} ? copy(z) : Array(z)

    ALh = copy(model.AL)
    AUh = copy(model.AU)
    lh = copy(model.l)
    uh = copy(model.u)

    ALh[ALh.==-Inf] .= -1.0e100
    AUh[AUh.==Inf] .= 1.0e100
    lh[lh.==-Inf] .= -1.0e100
    uh[uh.==Inf] .= 1.0e100

    @. yh = ifelse((AUh .== 1e100) & (ALh .== -1e100), 0.0,
        ifelse(AUh .== 1e100, max(yh, 0.0),
            ifelse(ALh .== -1e100, min(yh, 0.0), yh)))

    @. zh = ifelse((uh .== 1e100) & (lh .== -1e100), 0.0,
        ifelse(uh .== 1e100, max(zh, 0.0),
            ifelse(lh .== -1e100, min(zh, 0.0), zh)))

    Ax = model.A * xh
    ATy = model.AT * yh

    sum_sq_b = 0.0
    for val in model.AL
        if isfinite(val)
            sum_sq_b += val^2
        end
    end
    for val in model.AU
        if isfinite(val)
            sum_sq_b += val^2
        end
    end
    norm_b = 1.0 + sqrt(sum_sq_b)

    sum_sq_c = 0.0
    for val in model.c
        if isfinite(val)
            sum_sq_c += val^2
        end
    end
    norm_c = 1.0 + sqrt(sum_sq_c)

    err_Ax_sq = 0.0
    for i in eachindex(Ax)
        val = Ax[i]
        lower = ALh[i]
        upper = AUh[i]
        err_Ax_sq += max(0.0, lower - val, val - upper)^2
    end
    err_Ax = sqrt(err_Ax_sq)

    err_x_sq = 0.0
    for j in eachindex(xh)
        val = xh[j]
        lower = lh[j]
        upper = uh[j]
        err_x_sq += max(0.0, lower - val, val - upper)^2
    end
    err_x = sqrt(err_x_sq)
    primal_feas = max(err_Ax, err_x) / norm_b

    dual_residual = model.c .- ATy .- zh
    dual_feas = norm(dual_residual) / norm_c

    p_lin = dot(model.c, xh)
    delta_y = sum(((yb, al, au),) -> yb >= 0 ? yb * al : yb * au, zip(yh, ALh, AUh))
    delta_z = sum(((zb, lb, ub),) -> zb >= 0 ? zb * lb : zb * ub, zip(zh, lh, uh))
    d_lin = delta_y + delta_z

    gap = abs(d_lin - p_lin) / (1.0 + abs(d_lin) + abs(p_lin))
    p_obj = p_lin + model.obj_constant
    d_obj = d_lin + model.obj_constant

    return p_obj, d_obj, primal_feas, dual_feas, gap
end

function check_org_recovery_failures(
    p_feas::Real,
    d_feas::Real,
    gap::Real,
    stoptol::Real,
)
    failures = String[]
    if p_feas > stoptol
        push!(failures, "primal recover failed")
    end
    if d_feas > stoptol || gap > stoptol
        push!(failures, "dual recover failed")
    end
    return failures
end

function postsolve_and_validate_original_kkt!(
    results::HPRLP_results,
    original_model::LP_info_cpu,
    presolver_info,
    params::HPRLP_parameters,
)
    if params.verbose
        println("\n", "="^80)
        println("PSLP POSTSOLVE")
        println("="^80)
    end

    try
        x_red = results.x isa Vector{Float64} ? results.x : Vector(results.x)
        y_red = results.y isa Vector{Float64} ? results.y : Vector(results.y)
        z_red = results.z isa Vector{Float64} ? results.z : Vector(results.z)
        x_org, y_org, z_org = PSLP.postsolve(presolver_info, x_red, y_red, z_red)
        results.x = x_org
        results.y = y_org
        results.z = z_org
    finally
        PSLP.free_presolver_wrapper(presolver_info)
    end

    if results.status == "OPTIMAL"
        p_obj, d_obj, p_feas, d_feas, gap =
            compute_original_kkt_metrics(original_model, results.x, results.y, results.z)
        original_kkt_error = max(p_feas, d_feas, gap)
        original_kkt_passed = original_kkt_error <= params.stoptol

        if !original_kkt_passed
            failure_reasons = check_org_recovery_failures(
                p_feas, d_feas, gap, params.stoptol)
            if params.verbose
                println("Postsolve original KKT check failed (but the primal solution and objective are reliable): $(join(failure_reasons, "; ")). stop_tolerance = $(params.stoptol) primal_objective = $(p_obj) dual_objective = $(d_obj) primal_feasibility = $(p_feas) dual_feasibility = $(d_feas) relative_gap = $(gap)")
            end
        elseif params.verbose
            println("Postsolve original KKT check passed")
        end
    else
        if params.verbose
            println("Skipping postsolve original KKT check since the reduced solution is not optimal")
        end
    end

    return nothing
end

# the function to compute the residuals for the original LP problem
function compute_residuals_gpu!(ws::HPRLP_workspace_gpu,
    lp::LP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRLP_residuals,
    iter::Int,
    params::HPRLP_parameters,
    restart_info::HPRLP_restart,
    compute_gap::Bool=false,
)
    ### obj
    scbc = sc.b_scale * sc.c_scale
    queue_dot!(ws.reduction_scalars, 1, ws.c, ws.x_bar)

    use_support_dual = iter == 0 && (params.initial_y !== nothing || params.initial_x !== nothing)
    support_y = 0.0
    support_z = 0.0
    if use_support_dual
        # Project z_bar so support(-z_bar, [l,u]) is finite.
        # If u_i = +Inf, then z_i must be >= 0; if l_i = -Inf, then z_i must be <= 0; if both infinite, z_i = 0.
        @. ws.z_bar = ifelse((lp.u .== 1e100) & (lp.l .== -1e100), 0.0,
            ifelse(lp.u .== 1e100, max(ws.z_bar, 0.0),
                ifelse(lp.l .== -1e100, min(ws.z_bar, 0.0), ws.z_bar)))

        # Compute dual objective as negative of support function of -y_bar on [AL, AU] and -z_bar on [l, u]
        support_y = CUDA.mapreduce((yb, al, au) -> yb >= 0 ? yb * al : yb * au, +, ws.y_bar, lp.AL, lp.AU)
        support_z = CUDA.mapreduce((zb, lb, ub) -> zb >= 0 ? zb * lb : zb * ub, +, ws.z_bar, lp.l, lp.u)
    else
        queue_dot!(ws.reduction_scalars, 2, ws.y_obj, ws.y_bar)
        queue_dot!(ws.reduction_scalars, 3, ws.x_bar, ws.z_bar)
    end

    # Periodic restart gap computation:
    # keep identical arithmetic/reduction order, but batch this with the
    # residual readback to avoid an extra device->host sync.
    if compute_gap
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_dx, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
            ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
        queue_dot!(ws.reduction_scalars, 6, ws.Ax, ws.dy)
        queue_dot!(ws.reduction_scalars, 7, ws.dy, ws.dy)
        queue_dot!(ws.reduction_scalars, 8, ws.dx, ws.dx)
        # Also precompute restart movement norms here so update_sigma! can reuse
        # the same fetch without an extra restart-time synchronization.
        # axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
        # axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
        # queue_nrm2!(ws.reduction_scalars, 9, ws.dx)
        # queue_nrm2!(ws.reduction_scalars, 10, ws.dy)
    end

    ### Rd
    compute_Rd_gpu!(ws, sc)
    queue_nrm2!(ws.reduction_scalars, 4, ws.Rd)

    ### Rp
    compute_err_Rp_gpu!(ws, sc)
    queue_nrm2!(ws.reduction_scalars, 5, ws.Rp)

    if iter == 0
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_err_lu_kernel!(sc.col_norm, ws.dx, ws.x_bar, ws.l, ws.u, ws.n)
        # queue_nrm2!(ws.reduction_scalars, 6, ws.dx)
    end

    reduction_scalars_host = fetch_reduction_scalars!(ws)
    res.primal_obj_bar = scbc * reduction_scalars_host[1] + lp.obj_constant
    if use_support_dual
        res.dual_obj_bar = scbc * (support_y + support_z) + lp.obj_constant
    else
        res.dual_obj_bar = scbc * (reduction_scalars_host[2] + reduction_scalars_host[3]) + lp.obj_constant
    end
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0 + abs(res.primal_obj_bar) + abs(res.dual_obj_bar))

    res.err_Rd_org_bar = sc.c_scale * reduction_scalars_host[4] / sc.norm_c_org
    res.err_Rp_org_bar = sc.b_scale * reduction_scalars_host[5] / sc.norm_b_org

    if iter == 0
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, sc.b_scale * CUDA.norm(ws.dx))
    end
    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)

    if compute_gap
        dot_prod = 2 * reduction_scalars_host[6]
        dy_squarenorm = reduction_scalars_host[7]
        dx_squarenorm = reduction_scalars_host[8]
        weighted_norm = ws.sigma * (ws.lambda_max * dy_squarenorm) + (dx_squarenorm) / ws.sigma + dot_prod
        if weighted_norm < 0
            println("The estimated maximum eigenvalue is too small! Current value is ", ws.lambda_max)
            ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
            println("The new estimated maximum eigenvalue is ", ws.lambda_max)
            weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
        else
            weighted_norm = sqrt(weighted_norm)
        end
        restart_info.current_gap = weighted_norm
    end

    # Save best values if auto_save is enabled
    if params.auto_save
        if iter == 0 || res.KKTx_and_gap_org_bar < max(ws.saved_state.save_err_Rp, ws.saved_state.save_err_Rd, ws.saved_state.save_rel_gap)
            ws.saved_state.save_x .= ws.x_bar
            ws.saved_state.save_y .= ws.y_bar
            ws.saved_state.save_sigma = ws.sigma
            ws.saved_state.save_iter = iter
            ws.saved_state.save_err_Rp = res.err_Rp_org_bar
            ws.saved_state.save_err_Rd = res.err_Rd_org_bar
            ws.saved_state.save_primal_obj = res.primal_obj_bar
            ws.saved_state.save_dual_obj = res.dual_obj_bar
            ws.saved_state.save_rel_gap = res.rel_gap_bar
        end
    end
end

# the function to compute the residuals for the original LP problem
function compute_residuals_cpu!(ws::HPRLP_workspace_cpu,
    lp::LP_info_cpu,
    sc::Scaling_info_cpu,
    res::HPRLP_residuals,
    iter::Int,
    params::HPRLP_parameters,
    restart_info::HPRLP_restart,
    compute_gap::Bool=false,
)
    ### obj
    scbc = sc.b_scale * sc.c_scale
    res.primal_obj_bar = scbc * dot(ws.c, ws.x_bar) + lp.obj_constant
    if iter == 0 && (params.initial_y !== nothing || params.initial_x !== nothing)
        # Project z_bar so support(-z_bar, [l,u]) is finite.
        # If u_i = +Inf, then z_i must be >= 0; if l_i = -Inf, then z_i must be <= 0; if both infinite, z_i = 0.
        @. ws.z_bar = ifelse((lp.u .== 1e100) & (lp.l .== -1e100), 0.0,
            ifelse(lp.u .== 1e100, max(ws.z_bar, 0.0),
                ifelse(lp.l .== -1e100, min(ws.z_bar, 0.0), ws.z_bar)))
        # Compute dual objective as negative of support function of -y_bar on [AL, AU] and -z_bar on [l, u]
        support_y = sum(((yb, al, au),) -> yb >= 0 ? yb * al : yb * au, zip(ws.y_bar, lp.AL, lp.AU))
        support_z = sum(((zb, lb, ub),) -> zb >= 0 ? zb * lb : zb * ub, zip(ws.z_bar, lp.l, lp.u))
        res.dual_obj_bar = scbc * (support_y + support_z) + lp.obj_constant
    else
        res.dual_obj_bar = scbc * (dot(ws.y_obj, ws.y_bar) + dot(ws.x_bar, ws.z_bar)) + lp.obj_constant
    end
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

    if compute_gap
        restart_info.current_gap = compute_weighted_norm_cpu!(ws)
    end

    # Save best values if auto_save is enabled
    if params.auto_save
        if iter == 0 || res.KKTx_and_gap_org_bar < restart_info.best_gap
            ws.saved_state.save_x .= ws.x_bar
            ws.saved_state.save_y .= ws.y_bar
            ws.saved_state.save_sigma = ws.sigma
            ws.saved_state.save_iter = iter
            ws.saved_state.save_err_Rp = res.err_Rp_org_bar
            ws.saved_state.save_err_Rd = res.err_Rd_org_bar
            ws.saved_state.save_primal_obj = res.primal_obj_bar
            ws.saved_state.save_dual_obj = res.dual_obj_bar
            ws.saved_state.save_rel_gap = res.rel_gap_bar
        end
    end
end

# the function to update the value of sigma
function update_sigma_gpu!(
    restart_info::HPRLP_restart,
    ws::HPRLP_workspace_gpu,
    residuals::HPRLP_residuals,
)
    if restart_info.restart_flag >= 1 && restart_info.restart_flag <= 3
        # Movement norms are populated during periodic compute_residuals_gpu!.
        compute_restart_movement_norms_gpu!(ws)
        primal_move = ws.reduction_scalars_host[9]
        dual_move = ws.reduction_scalars_host[10]
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
    elapsed_time::Float64,
    params::HPRLP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "ITER_LIMIT"
    end

    if elapsed_time > params.time_limit
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

    results.status = status
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

    results.status = status
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
    spmv_alg = CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    sz_A = Ref{Csize_t}(0)
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_hat, ref_zero,
        desc_Ax, Float64, spmv_alg, sz_A)

    buf_A = CUDA.CuArray{UInt8}(undef, sz_A[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_hat, ref_zero, desc_Ax,
            Float64, spmv_alg, buf_A)
    end

    spmv_A = CUSPARSE_spmv_A(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, desc_x_hat, desc_dx, ref_zero, desc_Ax,
        Float64, spmv_alg, buf_A)

    sz_AT = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y, ref_zero,
        desc_ATy, Float64, spmv_alg, sz_AT)
    buf_AT = CUDA.CuArray{UInt8}(undef, sz_AT[])
    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y, ref_zero, desc_ATy,
            Float64, spmv_alg, buf_AT)
    end
    spmv_AT = CUSPARSE_spmv_AT(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, desc_y, ref_zero, desc_ATy,
        Float64, spmv_alg, buf_AT)

    return spmv_A, spmv_AT
end

@inline deterministic_probe_iterations() = 150

@inline function deterministic_probe_candidates()
    candidates = [(use_x=false, use_y=false)]
    push!(candidates, (use_x=true, use_y=false))
    push!(candidates, (use_x=false, use_y=true))
    push!(candidates, (use_x=true, use_y=true))
    return candidates
end

@inline function choose_deterministic_probe_backend(
    ref_metric::Float64,
    ref_time_ns::Integer,
    candidate_results::AbstractVector{<:NamedTuple};
    quality_rel_tol::Float64=0.01,
    quality_abs_tol::Float64=1e-12,
    speedup_margin::Float64=0.05)

    if !isfinite(ref_metric) || ref_time_ns <= 0
        return (use_x=false, use_y=false, reason=:default_cusparse)
    end

    allowed_metric = ref_metric + max(quality_abs_tol, abs(ref_metric) * quality_rel_tol)
    ref_time = Float64(ref_time_ns)
    best_choice = (use_x=false, use_y=false, reason=:default_cusparse)
    best_time = typemax(Float64)
    for candidate in candidate_results
        candidate_metric = candidate.metric
        candidate_time = Float64(candidate.time_ns)
        if !isfinite(candidate_metric) || candidate_metric > allowed_metric
            continue
        end
        if candidate_time <= 0.0
            continue
        end
        if candidate_time <= ref_time * (1.0 - speedup_margin) && candidate_time < best_time
            best_time = candidate_time
            best_choice = (use_x=candidate.use_x, use_y=candidate.use_y, reason=:speed)
        end
    end
    return best_choice
end

@inline function build_csr_row_buckets(row_ptr::CuVector{Int32}; short_max::Int32=Int32(16))
    row_ptr_h = Vector(row_ptr)
    nrows = length(row_ptr_h) - 1
    rows_short = Int32[]
    rows_medium = Int32[]
    sizehint!(rows_short, nrows)
    sizehint!(rows_medium, nrows ÷ 2)
    @inbounds for i in 1:nrows
        nnz_i = row_ptr_h[i+1] - row_ptr_h[i]
        if nnz_i <= short_max
            push!(rows_short, Int32(i))
        else
            push!(rows_medium, Int32(i))
        end
    end
    return CuArray(rows_short), CuArray(rows_medium)
end

# the function to allocate the workspace for the HPR-LP algorithm
function allocate_workspace_gpu(lp::LP_info_gpu, scaling_info::Scaling_info_gpu, params::HPRLP_parameters)
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
    ws.use_custom_deterministic_fused = !params.CUSPARSE_spmv
    ws.use_custom_fused_x = false
    ws.use_custom_fused_y = false
    ws.x_bound_type = CuArray(UInt8[])
    ws.y_bound_type = CuArray(UInt8[])
    ws.to_check = false
    ws.spmv_A, ws.spmv_AT = prepare_spmv!(lp.A, lp.AT, ws.x_bar, ws.x_hat, ws.dx, ws.Ax,
        ws.y_bar, ws.y, ws.ATy)
    if scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8
        ws.sigma = scaling_info.norm_b / scaling_info.norm_c
    else
        ws.sigma = 1.0
    end

    if ws.use_custom_deterministic_fused
        ws.x_bound_type = CUDA.zeros(UInt8, n)
        ws.y_bound_type = CUDA.zeros(UInt8, m)
        @. ws.x_bound_type = ifelse((ws.l <= -1e90) & (ws.u >= 1e90), UInt8(0),
            ifelse(ws.u >= 1e90, UInt8(1), ifelse(ws.l <= -1e90, UInt8(2), UInt8(3))))
        @. ws.y_bound_type = ifelse((ws.AL <= -1e90) & (ws.AU >= 1e90), UInt8(0),
            ifelse(ws.AU >= 1e90, UInt8(1), ifelse(ws.AL <= -1e90, UInt8(2), UInt8(3))))

        ws.A_rows_short, ws.A_rows_medium =
            build_csr_row_buckets(lp.A.rowPtr; short_max=Int32(16))
        ws.AT_rows_short, ws.AT_rows_medium =
            build_csr_row_buckets(lp.AT.rowPtr; short_max=Int32(16))
    else
        ws.A_rows_short = CuArray(Int32[])
        ws.A_rows_medium = CuArray(Int32[])
        ws.AT_rows_short = CuArray(Int32[])
        ws.AT_rows_medium = CuArray(Int32[])
    end

    # Initialize with user-provided initial x if available
    if params.initial_x !== nothing
        # Copy initial_x to GPU first
        ws.x .= CuArray(params.initial_x)

        # Scale x on GPU: inverse of x_result = b_scale * (x_bar / col_norm)
        # So x_bar = x_input * col_norm / b_scale
        ws.x .= ws.x .* scaling_info.col_norm ./ scaling_info.b_scale
        ws.x_bar .= ws.x
        ws.last_x .= ws.x
    end

    # Initialize with user-provided initial y if available
    if params.initial_y !== nothing
        # Copy initial_y to GPU first
        ws.y .= CuArray(params.initial_y)

        # Scale y on GPU: inverse of y_result = c_scale * (y_bar / row_norm)
        # So y_bar = y_input * row_norm / c_scale
        ws.y .= ws.y .* scaling_info.row_norm ./ scaling_info.c_scale
        # Project y_bar so support(-y_bar, [AL,AU]) is finite.
        # If AU_i = +Inf, then y_i must be >= 0; if AL_i = -Inf, then y_i must be <= 0; if both infinite, y_i = 0.
        @. ws.y = ifelse((lp.AU .== 1e100) & (lp.AL .== -1e100), 0.0,
            ifelse(lp.AU .== 1e100, max(ws.y, 0.0),
                ifelse(lp.AL .== -1e100, min(ws.y, 0.0), ws.y)))
        ws.y_bar .= ws.y
        ws.last_y .= ws.y

        # Compute z_bar = c - AT * y_bar
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha,
            ws.spmv_AT.desc_AT, ws.spmv_AT.desc_y_bar, ws.spmv_AT.beta, ws.spmv_AT.desc_ATy,
            ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
        ws.z_bar .= ws.c .- ws.ATy
    end

    # Initialize saved_state for auto_save feature
    ws.saved_state = HPRLP_saved_state_gpu()
    ws.saved_state.save_x = CUDA.zeros(Float64, n)
    ws.saved_state.save_y = CUDA.zeros(Float64, m)
    ws.saved_state.save_sigma = ws.sigma
    ws.saved_state.save_iter = 0
    ws.saved_state.save_err_Rp = Inf
    ws.saved_state.save_err_Rd = Inf
    ws.saved_state.save_primal_obj = 0.0
    ws.saved_state.save_dual_obj = 0.0
    ws.saved_state.save_rel_gap = Inf

    # Initialize backup flag
    ws.backup_created = false

    ws.Halpern_params = CUDA.zeros(Float64, 4)
    ws.halpern_inner = CUDA.zeros(Int64, 1)
    ws.halpern_factors = CuArray([0.5, 0.5])
    ws.reduction_scalars = CUDA.zeros(Float64, 13)
    ws.reduction_scalars_host = CUDA.pin(Vector{Float64}(undef, 13))
    ws.pending_last_gap = false
    ws.pending_last_gap_sigma = NaN
    ws.pending_last_gap_lambda_max = NaN

    return ws
end

function allocate_workspace_cpu(lp::LP_info_cpu, scaling_info::Scaling_info_cpu, params::HPRLP_parameters)
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

    # Initialize with user-provided initial x if available
    if params.initial_x !== nothing
        # Scale x: inverse of x_result = b_scale * (x_bar / col_norm)
        # So x_bar = x_input * col_norm / b_scale
        scaled_x = params.initial_x .* scaling_info.col_norm ./ scaling_info.b_scale
        ws.x .= scaled_x
        ws.x_bar .= scaled_x
        ws.last_x .= scaled_x
    end

    # Initialize with user-provided initial y if available
    if params.initial_y !== nothing
        # Scale y: inverse of y_result = c_scale * (y_bar / row_norm)
        # So y_bar = y_input * row_norm / c_scale
        scaled_y = params.initial_y .* scaling_info.row_norm ./ scaling_info.c_scale
        # Project y_bar so support(-y_bar, [AL,AU]) is finite.
        # If AU_i = +Inf, then y_i must be >= 0; if AL_i = -Inf, then y_i must be <= 0; if both infinite, y_i = 0.
        @. scaled_y = ifelse((lp.AU .== 1e100) & (lp.AL .== -1e100), 0.0,
            ifelse(lp.AU .== 1e100, max(scaled_y, 0.0),
                ifelse(lp.AL .== -1e100, min(scaled_y, 0.0), scaled_y)))
        ws.y .= scaled_y
        ws.y_bar .= scaled_y
        ws.last_y .= scaled_y

        # Compute z_bar = c - AT * y_bar
        mul!(ws.ATy, ws.AT, ws.y_bar)
        ws.z_bar .= ws.c .- ws.ATy
    end

    # Initialize saved_state for auto_save feature
    ws.saved_state = HPRLP_saved_state_cpu()
    ws.saved_state.save_x = Vector(zeros(n))
    ws.saved_state.save_y = Vector(zeros(m))
    ws.saved_state.save_sigma = ws.sigma
    ws.saved_state.save_iter = 0
    ws.saved_state.save_err_Rp = Inf
    ws.saved_state.save_err_Rd = Inf
    ws.saved_state.save_primal_obj = 0.0
    ws.saved_state.save_dual_obj = 0.0
    ws.saved_state.save_rel_gap = Inf

    return ws
end

# the function to save current state to HDF5 file
# This function is called whenever the log is printed (if auto_save is enabled)
# It saves:
#   - Current solution (x_bar, y_bar) - scaled to original problem
#   - Best solution so far (save_x, save_y) - scaled to original problem
#   - Current and best sigma values
#   - Current and best residuals, objectives, and iteration numbers
#   - Current iteration number and elapsed time
#   - All solver parameters including initial solutions
function save_state_to_hdf5(
    filename::String,
    ws::Union{HPRLP_workspace_gpu,HPRLP_workspace_cpu},
    sc::Union{Scaling_info_gpu,Scaling_info_cpu},
    residuals::HPRLP_residuals,
    params::HPRLP_parameters,
    iter::Int,
    t_start_alg::Float64,
)
    # Convert GPU arrays to CPU if needed
    if ws isa HPRLP_workspace_gpu
        x_bar = Vector(ws.x_bar)
        y_bar = Vector(ws.y_bar)
        save_x = Vector(ws.saved_state.save_x)
        save_y = Vector(ws.saved_state.save_y)
        col_norm = Vector(sc.col_norm)
        row_norm = Vector(sc.row_norm)
    else
        x_bar = ws.x_bar
        y_bar = ws.y_bar
        save_x = ws.saved_state.save_x
        save_y = ws.saved_state.save_y
        col_norm = sc.col_norm
        row_norm = sc.row_norm
    end

    # Scale the variables (same as in collect_results)
    x_bar_scaled = sc.b_scale * (x_bar ./ col_norm)
    y_bar_scaled = sc.c_scale * (y_bar ./ row_norm)
    save_x_scaled = sc.b_scale * (save_x ./ col_norm)
    save_y_scaled = sc.c_scale * (save_y ./ row_norm)

    # Create or open HDF5 file
    if isfile(filename)
        rm(filename, force=true)
    end
    h5open(filename, "w") do file
        # Save current iteration info
        file["current/iteration"] = iter
        file["current/time_elapsed"] = time() - t_start_alg
        file["current/timestamp"] = string(Dates.now())

        # Save current solution (scaled)
        file["current/x_org"] = x_bar_scaled
        file["current/y_org"] = y_bar_scaled
        file["current/sigma"] = ws.sigma

        # Save current residuals
        file["current/err_Rp"] = residuals.err_Rp_org_bar
        file["current/err_Rd"] = residuals.err_Rd_org_bar
        file["current/primal_obj"] = residuals.primal_obj_bar
        file["current/dual_obj"] = residuals.dual_obj_bar
        file["current/rel_gap"] = residuals.rel_gap_bar

        # Save best solution so far (scaled)
        file["best/x_org"] = save_x_scaled
        file["best/y_org"] = save_y_scaled
        file["best/sigma"] = ws.saved_state.save_sigma
        file["best/iteration"] = ws.saved_state.save_iter

        # Save best residuals
        file["best/err_Rp"] = ws.saved_state.save_err_Rp
        file["best/err_Rd"] = ws.saved_state.save_err_Rd
        file["best/primal_obj"] = ws.saved_state.save_primal_obj
        file["best/dual_obj"] = ws.saved_state.save_dual_obj
        file["best/rel_gap"] = ws.saved_state.save_rel_gap

        # Save parameters
        file["parameters/stoptol"] = params.stoptol
        file["parameters/max_iter"] = params.max_iter
        file["parameters/time_limit"] = params.time_limit
        file["parameters/check_iter"] = params.check_iter
        file["parameters/use_Ruiz_scaling"] = params.use_Ruiz_scaling
        file["parameters/use_Pock_Chambolle_scaling"] = params.use_Pock_Chambolle_scaling
        file["parameters/use_bc_scaling"] = params.use_bc_scaling
        file["parameters/use_gpu"] = params.use_gpu
        file["parameters/CUSPARSE_spmv"] = params.CUSPARSE_spmv
        file["parameters/device_number"] = params.device_number
        file["parameters/warm_up"] = params.warm_up
        file["parameters/print_frequency"] = params.print_frequency
        file["parameters/verbose"] = params.verbose
        file["parameters/auto_save"] = params.auto_save

        # Save initial solutions if provided
        if params.initial_x !== nothing
            file["parameters/initial_x"] = params.initial_x
        end
        if params.initial_y !== nothing
            file["parameters/initial_y"] = params.initial_y
        end
    end
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

# Helper function to transfer model to GPU
function setup_gpu_model(model::LP_info_cpu, params::HPRLP_parameters)
    CUDA.device!(params.device_number)
    t_start = time()
    if params.verbose
        println("COPY TO GPU ...")
    end

    model_gpu = LP_info_gpu(
        CuSparseMatrixCSR(model.A),
        CuSparseMatrixCSR(model.AT),
        CuVector(model.c),
        CuVector(model.AL),
        CuVector(model.AU),
        CuVector(model.l),
        CuVector(model.u),
        model.obj_constant,
    )
    CUDA.synchronize()

    if params.verbose
        println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
    end

    return model_gpu
end

# Helper function to apply scaling (CPU or GPU)
function setup_scaling(lp::Union{LP_info_cpu,LP_info_gpu}, params::HPRLP_parameters)
    t_start = time()

    if params.use_gpu
        if params.verbose
            println("SCALING LP ON GPU ...")
        end
        scaling_info = scaling_gpu!(lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        CUDA.synchronize()
        if params.verbose
            println(@sprintf("SCALING LP ON GPU time: %.2f seconds", time() - t_start))
        end
    else
        if params.verbose
            println("SCALING LP ...")
        end
        scaling_info = scaling!(lp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        if params.verbose
            println(@sprintf("SCALING LP time: %.2f seconds", time() - t_start))
        end
    end

    return scaling_info
end

# Helper function to print solver parameters
function print_solver_parameters(params::HPRLP_parameters, lp::Union{LP_info_cpu,LP_info_gpu})
    m, n = size(lp.A)

    # Count constraint types
    AL = lp.AL isa CuArray ? Vector(lp.AL) : lp.AL
    AU = lp.AU isa CuArray ? Vector(lp.AU) : lp.AU

    num_equalities = count(AL .== AU)
    num_inequalities = m - num_equalities
    nnz_A = nnz(lp.A isa CuSparseMatrixCSR ? SparseMatrixCSC(lp.A) : lp.A)

    println("="^80)
    println("PROBLEM INFORMATION:")
    println("  Rows (constraints): m = ", m)
    println("  Columns (variables): n = ", n)
    println("  Non-zeros in A: ", nnz_A)
    println("  Equalities: ", num_equalities)
    println("  Inequalities: ", num_inequalities)
    println()
    println("SOLVER PARAMETERS:")
    println("  Device: ", params.use_gpu ? "GPU (device $(params.device_number))" : "CPU")
    println("  Stop tolerance: ", params.stoptol)
    println("  Max iterations: ", params.max_iter)
    println("  Time limit: ", params.time_limit, " seconds")
    println("  Check interval: ", params.check_iter)
    println("  Print frequency: ", params.print_frequency == -1 ? "Adaptive" : params.print_frequency)
    println("  Scaling options:")
    println("    Ruiz scaling: ", params.use_Ruiz_scaling ? "Enabled" : "Disabled")
    println("    Pock-Chambolle scaling: ", params.use_Pock_Chambolle_scaling ? "Enabled" : "Disabled")
    println("    b/c scaling: ", params.use_bc_scaling ? "Enabled" : "Disabled")
    println("  CUSPARSE_spmv: ", params.CUSPARSE_spmv ? "Enabled (force cuSPARSE, no autotune)" : "Disabled")

    if params.warm_up
        println("  Warm-up: Enabled (avoids JIT compilation overhead)")
    else
        println("  Warm-up: Disabled")
        println("    ⚠ WARNING: First run of each function may be slower due to JIT compilation.")
        println("    ⚠ Consider enabling warm_up for more accurate timing measurements.")
    end

    if params.initial_x !== nothing
        println("  Initial x: Provided (length ", length(params.initial_x), ")")
    end
    if params.initial_y !== nothing
        println("  Initial y: Provided (length ", length(params.initial_y), ")")
    end

    if params.auto_save
        # Calculate estimated memory for auto_save
        memory_bytes = (n + m) * 16  # 8 bytes per Float64
        memory_mb = memory_bytes / (1024 * 1024)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)

        println("  Auto-save: ENABLED")
        println("    ⚠ WARNING: Auto-save will write to disk at each print iteration.")
        println("    ⚠ This may consume significant I/O bandwidth and slightly reduce speed.")
        if memory_gb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f GB", memory_gb))
        elseif memory_mb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f MB", memory_mb))
        elseif memory_bytes >= 1024.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f KB", memory_bytes / 1024))
        else
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f bytes", memory_bytes))
        end
        println("    Save file: ", params.save_filename)
    else
        println("  Auto-save: Disabled")
    end
    println("="^80)
end

# Helper function to select GPU or CPU function implementations
function setup_solver_functions(use_gpu::Bool)
    if use_gpu
        return (
            compute_residuals_gpu!,
            update_sigma_gpu!,
            collect_results_gpu!,
            update_x_z_check_gpu!,
            update_x_z_normal_gpu!,
            update_y_check_gpu!,
            update_y_normal_gpu!,
            compute_weighted_norm_gpu!
        )
    else
        return (
            compute_residuals_cpu!,
            update_sigma_cpu!,
            collect_results_cpu!,
            update_x_z_check_cpu!,
            update_x_z_normal_cpu!,
            update_y_check_cpu!,
            update_y_normal_cpu!,
            compute_weighted_norm_cpu!
        )
    end
end

@inline function reset_halpern_runtime_params!(ws::HPRLP_workspace_gpu)
    fact1 = ws.lambda_max * ws.sigma
    inv_fact1 = 1.0 / fact1
    inv_sigma = 1.0 / ws.sigma
    copyto!(ws.Halpern_params, [ws.sigma, fact1, inv_fact1, inv_sigma])
    copyto!(ws.halpern_inner, Int64[0])
    copyto!(ws.halpern_factors, [0.5, 0.5])
    return
end

@inline function upload_halpern_iter_params_if_needed!(
    ws::HPRLP_workspace_gpu,
    iter_params_host::Vector{Float64},
    uploaded_sigma::Float64,
    uploaded_lambda_max::Float64)
    if ws.sigma != uploaded_sigma || ws.lambda_max != uploaded_lambda_max
        y_fact1 = ws.lambda_max * ws.sigma
        @inbounds begin
            iter_params_host[1] = ws.sigma
            iter_params_host[2] = y_fact1
            iter_params_host[3] = 1.0 / y_fact1
            iter_params_host[4] = 1.0 / ws.sigma
        end
        copyto!(ws.Halpern_params, iter_params_host)
        uploaded_sigma = ws.sigma
        uploaded_lambda_max = ws.lambda_max
    end
    return uploaded_sigma, uploaded_lambda_max
end

@inline function upload_halpern_restart_params!(
    ws::HPRLP_workspace_gpu,
    restart_info::HPRLP_restart,
    halpern_inner_host::Vector{Int64},
    halpern_factors_host::Vector{Float64})
    if restart_info.restart_flag > 0
        @inbounds halpern_inner_host[1] = restart_info.inner
        copyto!(ws.halpern_inner, halpern_inner_host)
        current_fact1 = 1.0 / (restart_info.inner + 2.0)
        @inbounds begin
            halpern_factors_host[1] = current_fact1
            halpern_factors_host[2] = 1.0 - current_fact1
        end
        copyto!(ws.halpern_factors, halpern_factors_host)
    end
    return nothing
end

function autotune_custom_update_backends!(ws::HPRLP_workspace_gpu, lp::LP_info_gpu, sc::Scaling_info_gpu, params::HPRLP_parameters)
    if !ws.use_custom_deterministic_fused
        ws.use_custom_fused_x = false
        ws.use_custom_fused_y = false
        return
    end
    candidates = deterministic_probe_candidates()

    bench_iters = min(params.max_iter, deterministic_probe_iterations())
    if params.autotune_verbose
        println("AUTO-SELECT custom backends (", bench_iters, " deterministic iterations per candidate) ...")
        n_A_short = length(ws.A_rows_short)
        n_A_medium = length(ws.A_rows_medium)
        n_A_total = n_A_short + n_A_medium
        n_AT_short = length(ws.AT_rows_short)
        n_AT_medium = length(ws.AT_rows_medium)
        n_AT_total = n_AT_short + n_AT_medium
        p_A_short = n_A_total > 0 ? 100.0 * n_A_short / n_A_total : 0.0
        p_A_medium = n_A_total > 0 ? 100.0 * n_A_medium / n_A_total : 0.0
        p_AT_short = n_AT_total > 0 ? 100.0 * n_AT_short / n_AT_total : 0.0
        p_AT_medium = n_AT_total > 0 ? 100.0 * n_AT_medium / n_AT_total : 0.0
        println("  A row buckets: short=", n_A_short,
            ", medium=", n_A_medium,
            ", total=", n_A_total,
            " [",
            @sprintf("%.1f%%", p_A_short), ", ",
            @sprintf("%.1f%%", p_A_medium), "]")
        println("  AT row buckets: short=", n_AT_short,
            ", medium=", n_AT_medium,
            ", total=", n_AT_total,
            " [",
            @sprintf("%.1f%%", p_AT_short), ", ",
            @sprintf("%.1f%%", p_AT_medium), "]")
    end

    x_save = copy(ws.x)
    x_hat_save = copy(ws.x_hat)
    x_bar_save = copy(ws.x_bar)
    dx_save = copy(ws.dx)
    y_save = copy(ws.y)
    y_hat_save = copy(ws.y_hat)
    y_bar_save = copy(ws.y_bar)
    y_obj_save = copy(ws.y_obj)
    dy_save = copy(ws.dy)
    z_bar_save = copy(ws.z_bar)
    ATy_save = copy(ws.ATy)
    Ax_save = copy(ws.Ax)

    function restore_state!()
        ws.x .= x_save
        ws.x_hat .= x_hat_save
        ws.x_bar .= x_bar_save
        ws.dx .= dx_save
        ws.y .= y_save
        ws.y_hat .= y_hat_save
        ws.y_bar .= y_bar_save
        ws.y_obj .= y_obj_save
        ws.dy .= dy_save
        ws.z_bar .= z_bar_save
        ws.ATy .= ATy_save
        ws.Ax .= Ax_save
        reset_halpern_runtime_params!(ws)
        CUDA.synchronize()
    end

    function run_candidate_probe!(use_x::Bool, use_y::Bool, iters::Int)
        ws.use_custom_fused_x = use_x
        ws.use_custom_fused_y = use_y
        for _ in 1:iters
            update_x_z_normal_gpu!(ws)
            update_y_normal_gpu!(ws)
        end
        update_x_z_check_gpu!(ws)
        update_y_check_gpu!(ws)
        tmp_res = HPRLP_residuals()
        tmp_restart = initialize_restart(ws.sigma)
        compute_residuals_gpu!(ws, lp, sc, tmp_res, iters, params, tmp_restart)
        CUDA.synchronize()
        return tmp_res.KKTx_and_gap_org_bar
    end

    function eval_candidate(use_x::Bool, use_y::Bool, iters::Int)
        restore_state!()
        run_candidate_probe!(use_x, use_y, iters)
        restore_state!()
        CUDA.synchronize()
        t0 = time_ns()
        metric = run_candidate_probe!(use_x, use_y, iters)
        CUDA.synchronize()
        dt = time_ns() - t0
        return metric, dt
    end

    candidate_results = Dict{Tuple{Bool,Bool},Tuple{Float64,Int64}}()
    for candidate in candidates
        metric, dt = eval_candidate(candidate.use_x, candidate.use_y, bench_iters)
        candidate_results[(candidate.use_x, candidate.use_y)] = (metric, dt)
        if params.autotune_verbose
            println("  candidate x=", candidate.use_x ? "fused" : "cusparse",
                ", y=", candidate.use_y ? "fused" : "cusparse",
                " -> ", round(dt / 1e6; digits=3), " ms",
                ", merit=", metric)
        end
    end

    ref_metric, ref_t = candidate_results[(false, false)]
    fused_candidates = [
        (use_x=candidate.use_x,
            use_y=candidate.use_y,
            metric=candidate_results[(candidate.use_x, candidate.use_y)][1],
            time_ns=candidate_results[(candidate.use_x, candidate.use_y)][2])
        for candidate in candidates if candidate != (use_x=false, use_y=false)
    ]
    decision = choose_deterministic_probe_backend(ref_metric, ref_t, fused_candidates)

    restore_state!()
    ws.use_custom_fused_x = decision.use_x
    ws.use_custom_fused_y = decision.use_y

    if params.autotune_verbose
        selected_metric, selected_t = candidate_results[(decision.use_x, decision.use_y)]
        println("AUTO-SELECT selected x=", decision.use_x ? "fused" : "cusparse",
            ", y=", decision.use_y ? "fused" : "cusparse",
            " (", round(selected_t / 1e6; digits=3), " ms, merit=", selected_metric,
            ", reason=", decision.reason, ")")
    end
    return
end

# Helper function to determine if log should be printed
function should_print_log(iter::Int, max_iter::Int, print_frequency::Int, elapsed_time::Float64, time_limit::Float64)
    if print_frequency == -1
        return (rem(iter, print_step(iter)) == 0) || (iter == max_iter) || (elapsed_time > time_limit)
    elseif print_frequency > 0
        return (rem(iter, print_frequency) == 0) || (iter == max_iter) || (elapsed_time > time_limit)
    else
        error("Invalid print_frequency: ", print_frequency, ". It should be a positive integer or -1 for automatic printing.")
    end
end

@inline iteration_logging_enabled(verbose::Bool) = verbose

@inline function next_iteration_log_iter(iter::Int, print_frequency::Int)
    if print_frequency == -1
        step = trunc(Int, print_step(max(iter + 1, 1)))
    elseif print_frequency > 0
        step = print_frequency
    else
        error("Invalid print_frequency: ", print_frequency,
            ". It should be a positive integer or -1 for automatic printing.")
    end
    return ((iter ÷ step) + 1) * step
end

# Helper function to print iteration log
function print_iteration_log(iter::Int, residuals::HPRLP_residuals, sigma::Float64, t_start_alg::Float64)
    println(@sprintf("%5.0f    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f",
        iter,
        residuals.err_Rp_org_bar,
        residuals.err_Rd_org_bar,
        residuals.primal_obj_bar,
        residuals.dual_obj_bar,
        residuals.rel_gap_bar,
        sigma,
        time() - t_start_alg))
end

# Helper function to check and record tolerance thresholds
function check_and_record_tolerance!(
    residuals::HPRLP_residuals,
    iter::Int,
    t_start_alg::Float64,
    tolerance_levels::Vector{Float64},
    tolerance_times::Vector{Float64},
    tolerance_iters::Vector{Int},
    tolerance_reached::BitVector,
    verbose::Bool
)
    for i in eachindex(tolerance_levels)
        if !tolerance_reached[i] && residuals.KKTx_and_gap_org_bar < tolerance_levels[i]
            tolerance_times[i] = time() - t_start_alg
            tolerance_iters[i] = iter
            tolerance_reached[i] = true
            if verbose
                println("KKT < ", tolerance_levels[i], " at iter = ", iter)
            end
        end
    end
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
        lambda_max = power_iteration_gpu(ws) * 1.01
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
"""
    optimize(model::LP_info_cpu, params::HPRLP_parameters)

Optimize a linear program using the HPR-LP algorithm.

This function handles GPU transfer, scaling, and optional warmup internally based on the parameters.

# Arguments
- `model::LP_info_cpu`: LP model built from `build_from_mps` or `build_from_Abc`
- `params::HPRLP_parameters`: Solver parameters

# Returns
- `HPRLP_results`: Solution results including objective value, solution vector, and convergence info

# Example
```julia
using HPRLP

model = build_from_mps("problem.mps")
params = HPRLP_parameters()
params.stoptol = 1e-6
params.use_gpu = true
params.warm_up = true

result = optimize(model, params)
println("Status: ", result.status)
println("Objective: ", result.primal_obj)
```

See also: [`build_from_mps`](@ref), [`build_from_Abc`](@ref), [`HPRLP_parameters`](@ref)
"""
function optimize(model::LP_info_cpu, params::HPRLP_parameters)
    # Presolve if requested
    if params.use_presolve
        original_model = model
        model, presolver_info = apply_pslp_presolve(original_model, params.verbose)
    end

    # Handle warmup if requested
    if params.warm_up
        if params.verbose
            println("="^80)
            println("WARM UP PHASE")
            println("  ℹ Running warmup to avoid JIT compilation overhead in main solve")
            println("="^80)
        end
        t_start_warmup = time()

        # Save original max_iter and verbose
        original_max_iter = params.max_iter
        original_verbose = params.verbose
        params.max_iter = 200
        params.verbose = false

        # Create a copy of the model for warmup
        warmup_model = LP_info_cpu(
            copy(model.A), copy(model.AT), copy(model.c),
            copy(model.AL), copy(model.AU), copy(model.l), copy(model.u),
            model.obj_constant
        )

        # Run warmup solve
        solve(warmup_model, params)

        # Restore original parameters
        params.max_iter = original_max_iter
        params.verbose = original_verbose

        warmup_time = time() - t_start_warmup
        if params.verbose
            println(@sprintf("Warmup time: %.2f seconds", warmup_time))
            println("="^80)
            println()
        end
    end

    # Main solve
    if params.verbose
        println("="^80)
        println("MAIN SOLVE")
        println("="^80)
    end

    setup_start = time()
    results = solve(model, params)
    setup_time = time() - setup_start - results.time

    if params.verbose
        println(@sprintf("Total time: %.2fs", setup_time + results.time),
            @sprintf("  setup time = %.2fs", setup_time),
            @sprintf("  solve time = %.2fs", results.time))
        println("="^80)
    end

    if presolver_info !== nothing
        postsolve_and_validate_original_kkt!(results, original_model, presolver_info, params)
    end

    return results
end

"""
    solve(model::LP_info_cpu, params::HPRLP_parameters)

Solve a CPU-based LP model (with GPU transfer and scaling applied internally).

This is the core solver function that:
1. Transfers to GPU if needed (based on params.use_gpu)
2. Applies scaling
3. Runs the HPR-LP algorithm

# Arguments
- `model::LP_info_cpu`: LP model on CPU
- `params::HPRLP_parameters`: Solver parameters

# Returns
- `HPRLP_results`: Solution results

# Example
```julia
using HPRLP

model = build_from_Abc(A, c, AL, AU, l, u)
params = HPRLP_parameters()
params.use_gpu = true
result = solve(model, params)
```

See also: [`build_from_mps`](@ref), [`build_from_Abc`](@ref), [`optimize`](@ref)
"""
function solve(model::LP_info_cpu, params::HPRLP_parameters)
    # Validate GPU parameters before attempting GPU operations
    if params.use_gpu
        validate_gpu_parameters!(params)
    end

    # Setup: GPU transfer and scaling
    if params.use_gpu
        lp = setup_gpu_model(model, params)
        scaling_info = setup_scaling(lp, params)
    else
        scaling_info = setup_scaling(model, params)
        lp = model
    end

    # Main optimization algorithm
    if params.verbose
        println("HPR-LP version v0.1.4")
    end
    t_start_alg = time()

    # Print solver parameters
    if params.verbose
        print_solver_parameters(params, lp)
    end

    # Initialization
    residuals = HPRLP_residuals()
    log_residuals = HPRLP_residuals()
    ws = params.use_gpu ? allocate_workspace_gpu(lp, scaling_info, params) : allocate_workspace_cpu(lp, scaling_info, params)
    restart_info = initialize_restart(ws.sigma)

    # Power iteration to estimate lambda_max
    power_time = compute_maximum_eigenvalue!(lp, ws, params)
    if params.use_gpu
        autotune_custom_update_backends!(ws, lp, scaling_info, params)
    end


    if params.verbose
        println(" iter     errRp        errRd         p_obj            d_obj          gap         sigma       time")
    end

    # Track when tolerance thresholds are reached
    tolerance_levels = [1e-4, 1e-6, 1e-8]
    tolerance_times = zeros(Float64, length(tolerance_levels))
    tolerance_iters = zeros(Int, length(tolerance_levels))
    tolerance_reached = falses(length(tolerance_levels))

    # Select GPU or CPU function implementations
    compute_residuals!, update_sigma!, collect_results!, update_x_z_check!, update_x_z_normal!, update_y_check!, update_y_normal!, compute_weighted_norm! =
        setup_solver_functions(params.use_gpu)

    graph_exec = nothing
    iter_params_host = Vector{Float64}(undef, 4)
    halpern_inner_host = Vector{Int64}(undef, 1)
    halpern_factors_host = Vector{Float64}(undef, 2)
    uploaded_sigma = NaN
    uploaded_lambda_max = NaN
    elapsed_time = 0.0

    # Main iteration loop
    for iter = 0:params.max_iter
        periodic_check = rem(iter, params.check_iter) == 0

        # Determine if log should be printed
        if (iter & 31) == 0
            elapsed_time = time() - t_start_alg
        end
        print_yes = should_print_log(iter, params.max_iter, params.print_frequency, elapsed_time, params.time_limit)
        residuals_refreshed = false
        # residuals_to_print = residuals

        # Compute residuals
        if periodic_check
            compute_gap_now = params.use_gpu && iter > 0 && periodic_check
            compute_residuals!(ws, lp, scaling_info, residuals, iter, params, restart_info, compute_gap_now)
            residuals_refreshed = true
        elseif print_yes
            compute_residuals!(ws, lp, scaling_info, residuals, iter, params, restart_info, false)
            # residuals_to_print = residuals
        end
        if params.use_gpu && ws.pending_last_gap && periodic_check
            consume_pending_last_gap_gpu!(ws, restart_info)
        end

        # Check termination conditions (cache wall-clock checks to reduce host overhead)
        if (iter & 31) == 0 || print_yes
            elapsed_time = time() - t_start_alg
        end
        status = check_break(residuals, iter, elapsed_time, params)

        # Check restart conditions
        restart_info.restart_flag = 0
        if periodic_check
            check_restart(restart_info, iter, params.check_iter, ws.sigma)
        end

        # Print iteration log
        if print_yes || (status != "CONTINUE")
            if params.verbose
                print_iteration_log(iter, residuals, ws.sigma, t_start_alg)
            end

            # Save to HDF5 if auto_save is enabled
            if params.auto_save
                try
                    save_state_to_hdf5(params.save_filename, ws, scaling_info, residuals, params, iter, t_start_alg)
                catch e
                    if params.verbose
                        println("Warning: Failed to save to HDF5 file: ", e)
                    end
                end
            end
        end

        # Check and record tolerance thresholds
        if residuals_refreshed
            check_and_record_tolerance!(residuals, iter, t_start_alg, tolerance_levels,
                tolerance_times, tolerance_iters, tolerance_reached, params.verbose)
        end

        # Collect results and return if terminated
        if status != "CONTINUE"
            if params.verbose
                println("\n", "="^80)
                println("SOLUTION SUMMARY")
                println("="^80)
                println("Status: ", status)
                println("Iterations: ", iter)
                println("Time: ", @sprintf("%.4f", time() - t_start_alg), " seconds")
                println("Primal Objective: ", @sprintf("%+.12e", residuals.primal_obj_bar))
                println("Dual Objective: ", @sprintf("%+.12e", residuals.dual_obj_bar))
                println("Primal Residual: ", @sprintf("%.6e", residuals.err_Rp_org_bar))
                println("Dual Residual: ", @sprintf("%.6e", residuals.err_Rd_org_bar))
                println("Relative Gap: ", @sprintf("%.6e", residuals.rel_gap_bar))
                println("KKT Error: ", @sprintf("%.6e", residuals.KKTx_and_gap_org_bar))
                println("="^80)
            end
            results = collect_results!(ws, residuals, scaling_info, iter, t_start_alg, power_time, status, tolerance_times, tolerance_iters)
            return results
        end

        # Update sigma
        update_sigma!(restart_info, ws, residuals)

        # Restart if needed
        do_restart!(restart_info, ws)

        # Rebuild Graph if restarted
        if params.use_gpu && (iter == 0 || restart_info.restart_flag > 0)
            graph = CUDA.capture() do
                update_x_z_normal_gpu!(ws)
                update_y_normal_gpu!(ws)
            end
            graph_exec = CUDA.instantiate(graph)
        end

        # Determine whether to compute bar points for residuals
        ws.to_check = (rem(iter + 1, params.check_iter) == 0) || (restart_info.restart_flag > 0)
        # if params.print_frequency == -1
        #     ws.to_check = ws.to_check || (rem(iter + 1, print_step(iter + 1)) == 0)
        # elseif params.print_frequency > 0
        #     ws.to_check = ws.to_check || (rem(iter + 1, params.print_frequency) == 0)
        # end
        if params.use_gpu
            uploaded_sigma, uploaded_lambda_max = upload_halpern_iter_params_if_needed!(
                ws, iter_params_host, uploaded_sigma, uploaded_lambda_max)
            upload_halpern_restart_params!(ws, restart_info, halpern_inner_host, halpern_factors_host)

            if ws.to_check
                update_x_z_check_gpu!(ws)
                update_y_check_gpu!(ws)
            else
                CUDA.launch(graph_exec)
            end
        else
            current_fact1 = 1.0 / (restart_info.inner + 2.0)
            current_fact2 = 1.0 - current_fact1
            if ws.to_check
                update_x_z_check!(ws, current_fact1, current_fact2)
                update_y_check!(ws, current_fact1, current_fact2)
            else
                update_x_z_normal!(ws, current_fact1, current_fact2)
                update_y_normal!(ws, current_fact1, current_fact2)
            end
        end
        restart_info.inner += 1

        # Compute weighted norm
        if !params.use_gpu && rem(iter + 1, params.check_iter) == 0
            restart_info.current_gap = compute_weighted_norm!(ws)
        end
        if restart_info.restart_flag > 0
            if params.use_gpu
                queue_pending_last_gap_gpu!(ws)
                # restart_info.last_gap = compute_weighted_norm!(ws)
            else
                restart_info.last_gap = compute_weighted_norm!(ws)
            end
        end
    end
end
