function batched_spmm_A!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu; source::Symbol=:X_bar)
    Xsrc = source === :X ? ws.X : source === :X_hat ? ws.X_hat : source === :DX ? ws.DX : ws.X_bar
    mul!(ws.AX, shared.A, Xsrc)
    return ws.AX
end

function batched_spmm_AT!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu; source::Symbol=:Y_bar)
    Ysrc = source === :Y ? ws.Y : source === :Y_hat ? ws.Y_hat : ws.Y_bar
    mul!(ws.ATY, shared.AT, Ysrc)
    return ws.ATY
end

function batched_spmm_pair!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu)
    batched_spmm_A!(ws, shared)
    batched_spmm_AT!(ws, shared)
    CUDA.synchronize()
    return ws.AX, ws.ATY
end

function update_x_z_check_batched_kernel!(
    DX::CuDeviceMatrix{Float64},
    X::CuDeviceMatrix{Float64},
    Z_bar::CuDeviceMatrix{Float64},
    X_bar::CuDeviceMatrix{Float64},
    X_hat::CuDeviceMatrix{Float64},
    L::CuDeviceMatrix{Float64},
    U::CuDeviceMatrix{Float64},
    ATY::CuDeviceMatrix{Float64},
    C::CuDeviceMatrix{Float64},
    last_X::CuDeviceMatrix{Float64},
    sigma::CuDeviceVector{Float64},
    halpern_fact1::CuDeviceVector{Float64},
    halpern_fact2::CuDeviceVector{Float64},
    active::CuDeviceVector{Bool},
    n::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        k = cld(t, n)
        if active[k]
            sig = sigma[k]
            inv_sig = 1.0 / sig
            @inbounds begin
                xi = X[t]
                z_trial = muladd(sig, ATY[t] - C[t], xi)
                xbar = min(max(z_trial, L[t]), U[t])
                zbar = (xbar - z_trial) * inv_sig
                xhat = 2.0 * xbar - xi
                DX[t] = xbar - xhat
                Z_bar[t] = zbar
                X_bar[t] = xbar
                X_hat[t] = xhat
                X[t] = muladd(halpern_fact2[k], xhat, halpern_fact1[k] * last_X[t])
            end
        end
    end
    return
end

function update_x_z_normal_batched_kernel!(
    X::CuDeviceMatrix{Float64},
    X_hat::CuDeviceMatrix{Float64},
    L::CuDeviceMatrix{Float64},
    U::CuDeviceMatrix{Float64},
    ATY::CuDeviceMatrix{Float64},
    C::CuDeviceMatrix{Float64},
    last_X::CuDeviceMatrix{Float64},
    sigma::CuDeviceVector{Float64},
    halpern_fact1::CuDeviceVector{Float64},
    halpern_fact2::CuDeviceVector{Float64},
    active::CuDeviceVector{Bool},
    n::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        k = cld(t, n)
        if active[k]
            sig = sigma[k]
            @inbounds begin
                xi = X[t]
                z_trial = muladd(sig, ATY[t] - C[t], xi)
                xbar = min(max(z_trial, L[t]), U[t])
                xhat = 2.0 * xbar - xi
                X_hat[t] = xhat
                X[t] = muladd(halpern_fact2[k], xhat, halpern_fact1[k] * last_X[t])
            end
        end
    end
    return
end

function update_x_z_check_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu)
    batched_spmm_AT!(ws, shared; source=:Y)
    total = ws.n * ws.B
    @cuda threads = 256 blocks = cld(total, 256) update_x_z_check_batched_kernel!(
        ws.DX, ws.X, ws.Z_bar, ws.X_bar, ws.X_hat, batch.L, batch.U, ws.ATY,
        batch.C, ws.last_X, ws.sigma, ws.halpern_fact1, ws.halpern_fact2,
        ws.active, ws.n, total)
    return nothing
end

function update_x_z_normal_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu)
    batched_spmm_AT!(ws, shared; source=:Y)
    total = ws.n * ws.B
    @cuda threads = 256 blocks = cld(total, 256) update_x_z_normal_batched_kernel!(
        ws.X, ws.X_hat, batch.L, batch.U, ws.ATY, batch.C, ws.last_X,
        ws.sigma, ws.halpern_fact1, ws.halpern_fact2, ws.active, ws.n, total)
    return nothing
end

function update_x_z_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu, halpern_fact1::Float64, halpern_fact2::Float64)
    fill!(ws.halpern_fact1, halpern_fact1)
    fill!(ws.halpern_fact2, halpern_fact2)
    return update_x_z_check_batched_gpu!(ws, shared, batch)
end

function update_y_check_batched_kernel!(
    DY::CuDeviceMatrix{Float64},
    Y_bar::CuDeviceMatrix{Float64},
    Y_hat::CuDeviceMatrix{Float64},
    Y::CuDeviceMatrix{Float64},
    Y_obj::CuDeviceMatrix{Float64},
    AL::CuDeviceMatrix{Float64},
    AU::CuDeviceMatrix{Float64},
    AX::CuDeviceMatrix{Float64},
    last_Y::CuDeviceMatrix{Float64},
    sigma::CuDeviceVector{Float64},
    halpern_fact1::CuDeviceVector{Float64},
    halpern_fact2::CuDeviceVector{Float64},
    active::CuDeviceVector{Bool},
    lambda_max::Float64,
    m::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        k = cld(t, m)
        if active[k]
            fact1 = lambda_max * sigma[k]
            fact2 = 1.0 / fact1
            @inbounds begin
                yi = Y[t]
                v = AX[t] - fact1 * yi
                d = max(AL[t] - v, min(AU[t] - v, 0.0))
                yb = fact2 * d
                yh = 2.0 * yb - yi
                DY[t] = yb - yh
                Y_bar[t] = yb
                Y_hat[t] = yh
                Y_obj[t] = v + d
                Y[t] = muladd(halpern_fact2[k], yh, halpern_fact1[k] * last_Y[t])
            end
        end
    end
    return
end

function update_y_normal_batched_kernel!(
    Y::CuDeviceMatrix{Float64},
    AL::CuDeviceMatrix{Float64},
    AU::CuDeviceMatrix{Float64},
    AX::CuDeviceMatrix{Float64},
    last_Y::CuDeviceMatrix{Float64},
    sigma::CuDeviceVector{Float64},
    halpern_fact1::CuDeviceVector{Float64},
    halpern_fact2::CuDeviceVector{Float64},
    active::CuDeviceVector{Bool},
    lambda_max::Float64,
    m::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        k = cld(t, m)
        if active[k]
            fact1 = lambda_max * sigma[k]
            fact2 = 1.0 / fact1
            @inbounds begin
                yi = Y[t]
                v = AX[t] - fact1 * yi
                d = max(AL[t] - v, min(AU[t] - v, 0.0))
                yb = fact2 * d
                yh = 2.0 * yb - yi
                Y[t] = muladd(halpern_fact2[k], yh, halpern_fact1[k] * last_Y[t])
            end
        end
    end
    return
end

function update_y_check_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu)
    batched_spmm_A!(ws, shared; source=:X_hat)
    total = ws.m * ws.B
    @cuda threads = 256 blocks = cld(total, 256) update_y_check_batched_kernel!(
        ws.DY, ws.Y_bar, ws.Y_hat, ws.Y, ws.Y_obj, batch.AL, batch.AU, ws.AX,
        ws.last_Y, ws.sigma, ws.halpern_fact1, ws.halpern_fact2, ws.active,
        ws.lambda_max, ws.m, total)
    return nothing
end

function update_y_normal_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu)
    batched_spmm_A!(ws, shared; source=:X_hat)
    total = ws.m * ws.B
    @cuda threads = 256 blocks = cld(total, 256) update_y_normal_batched_kernel!(
        ws.Y, batch.AL, batch.AU, ws.AX, ws.last_Y, ws.sigma,
        ws.halpern_fact1, ws.halpern_fact2, ws.active, ws.lambda_max, ws.m, total)
    return nothing
end

function update_y_batched_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu, batch::BatchedLPData_gpu, halpern_fact1::Float64, halpern_fact2::Float64)
    fill!(ws.halpern_fact1, halpern_fact1)
    fill!(ws.halpern_fact2, halpern_fact2)
    return update_y_check_batched_gpu!(ws, shared, batch)
end

function compute_batched_Rd_kernel!(
    col_norm::CuDeviceVector{Float64},
    ATY::CuDeviceMatrix{Float64},
    Z_bar::CuDeviceMatrix{Float64},
    C::CuDeviceMatrix{Float64},
    RD::CuDeviceMatrix{Float64},
    n::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        i = mod1(t, n)
        @inbounds RD[t] = (C[t] - ATY[t] - Z_bar[t]) * col_norm[i]
    end
    return
end

function compute_batched_Rp_kernel!(
    row_norm::CuDeviceVector{Float64},
    RP::CuDeviceMatrix{Float64},
    AL::CuDeviceMatrix{Float64},
    AU::CuDeviceMatrix{Float64},
    AX::CuDeviceMatrix{Float64},
    m::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        i = mod1(t, m)
        @inbounds begin
            v = AX[t]
            RP[t] = row_norm[i] * max(min(AU[t] - v, 0.0), AL[t] - v)
        end
    end
    return
end

function compute_batched_lu_violation_kernel!(
    col_norm::CuDeviceVector{Float64},
    DX::CuDeviceMatrix{Float64},
    X_bar::CuDeviceMatrix{Float64},
    L::CuDeviceMatrix{Float64},
    U::CuDeviceMatrix{Float64},
    n::Int,
    total::Int,
)
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total
        i = mod1(t, n)
        @inbounds begin
            x = X_bar[t]
            violation = x < L[t] ? L[t] - x : (x > U[t] ? x - U[t] : 0.0)
            DX[t] = violation / col_norm[i]
        end
    end
    return
end

function batched_restart_movement_kernel!(
    DX::CuDeviceMatrix{Float64},
    DY::CuDeviceMatrix{Float64},
    X_bar::CuDeviceMatrix{Float64},
    Y_bar::CuDeviceMatrix{Float64},
    last_X::CuDeviceMatrix{Float64},
    last_Y::CuDeviceMatrix{Float64},
    n::Int,
    m::Int,
    B::Int,
)
    total_x = n * B
    total_y = m * B
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total_x
        @inbounds DX[t] = X_bar[t] - last_X[t]
    end
    if t <= total_y
        @inbounds DY[t] = Y_bar[t] - last_Y[t]
    end
    return
end

function do_batched_restart_kernel!(
    X::CuDeviceMatrix{Float64},
    Y::CuDeviceMatrix{Float64},
    last_X::CuDeviceMatrix{Float64},
    last_Y::CuDeviceMatrix{Float64},
    X_bar::CuDeviceMatrix{Float64},
    Y_bar::CuDeviceMatrix{Float64},
    restart_flags::CuDeviceVector{Bool},
    n::Int,
    m::Int,
    B::Int,
)
    total_x = n * B
    total_y = m * B
    t = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if t <= total_x
        kx = cld(t, n)
        if restart_flags[kx]
            @inbounds begin
                X[t] = X_bar[t]
                last_X[t] = X_bar[t]
            end
        end
    end
    if t <= total_y
        ky = cld(t, m)
        if restart_flags[ky]
            @inbounds begin
                Y[t] = Y_bar[t]
                last_Y[t] = Y_bar[t]
            end
        end
    end
    return
end

function allocate_batched_residuals_gpu(B::Int)
    return BatchedResiduals_gpu(
        CUDA.zeros(Float64, B),
        CUDA.zeros(Float64, B),
        CUDA.zeros(Float64, B),
        CUDA.zeros(Float64, B),
        CUDA.zeros(Float64, B),
        CUDA.zeros(Float64, B),
    )
end

function compute_batched_residuals_gpu!(
    residuals::BatchedResiduals_gpu,
    ws::BatchedWorkspace_gpu,
    shared::BatchedSharedMatrix_gpu,
    batch::BatchedLPData_gpu,
    scaling::BatchedScalingInfo_gpu,
    iter::Int=0,
)
    batched_spmm_AT!(ws, shared; source=:Y_bar)
    total_n = ws.n * ws.B
    @cuda threads = 256 blocks = cld(total_n, 256) compute_batched_Rd_kernel!(
        scaling.col_norm, ws.ATY, ws.Z_bar, batch.C, ws.RD, ws.n, total_n)

    batched_spmm_A!(ws, shared; source=:X_bar)
    total_m = ws.m * ws.B
    @cuda threads = 256 blocks = cld(total_m, 256) compute_batched_Rp_kernel!(
        scaling.row_norm, ws.RP, batch.AL, batch.AU, ws.AX, ws.m, total_m)

    scbc = scaling.b_scale .* scaling.c_scale
    residuals.primal_obj .= scbc .* vec(sum(batch.C .* ws.X_bar; dims=1)) .+ batch.obj_constants
    residuals.dual_obj .= scbc .* vec(sum(ws.Y_obj .* ws.Y_bar; dims=1) .+ sum(ws.X_bar .* ws.Z_bar; dims=1)) .+ batch.obj_constants
    residuals.err_Rd .= scaling.c_scale .* vec(sqrt.(sum(abs2.(ws.RD); dims=1))) ./ scaling.norm_c_org
    residuals.err_Rp .= scaling.b_scale .* vec(sqrt.(sum(abs2.(ws.RP); dims=1))) ./ scaling.norm_b_org
    if iter == 0
        @cuda threads = 256 blocks = cld(total_n, 256) compute_batched_lu_violation_kernel!(
            scaling.col_norm, ws.DX, ws.X_bar, batch.L, batch.U, ws.n, total_n)
        residuals.err_Rp .= max.(residuals.err_Rp, scaling.b_scale .* vec(sqrt.(sum(abs2.(ws.DX); dims=1))))
    end
    residuals.rel_gap .= abs.(residuals.primal_obj .- residuals.dual_obj) ./ (1.0 .+ abs.(residuals.primal_obj) .+ abs.(residuals.dual_obj))
    residuals.kkt_error .= max.(residuals.err_Rp, max.(residuals.err_Rd, residuals.rel_gap))
    return residuals
end

function compute_batched_weighted_norm_gpu!(ws::BatchedWorkspace_gpu, shared::BatchedSharedMatrix_gpu)
    batched_spmm_A!(ws, shared; source=:DX)
    dot_prod = 2.0 .* Array(vec(sum(ws.AX .* ws.DY; dims=1)))
    dy_squarenorm = Array(vec(sum(abs2.(ws.DY); dims=1)))
    dx_squarenorm = Array(vec(sum(abs2.(ws.DX); dims=1)))
    weighted_norm = ws.sigma |> Array
    for k in eachindex(weighted_norm)
        sigma = weighted_norm[k]
        value = sigma * (ws.lambda_max * dy_squarenorm[k]) + dx_squarenorm[k] / sigma + dot_prod[k]
        if value < 0.0 && dy_squarenorm[k] > 0.0
            candidate_lambda = -(dot_prod[k] + dx_squarenorm[k] / sigma) / (sigma * dy_squarenorm[k]) * 1.05
            ws.lambda_max = max(ws.lambda_max, candidate_lambda)
            value = sigma * (ws.lambda_max * dy_squarenorm[k]) + dx_squarenorm[k] / sigma + dot_prod[k]
        end
        weighted_norm[k] = sqrt(max(value, 0.0))
    end
    return weighted_norm
end

function compute_batched_restart_movement_norms_gpu!(ws::BatchedWorkspace_gpu)
    total = max(ws.n * ws.B, ws.m * ws.B)
    @cuda threads = 256 blocks = cld(total, 256) batched_restart_movement_kernel!(
        ws.DX, ws.DY, ws.X_bar, ws.Y_bar, ws.last_X, ws.last_Y, ws.n, ws.m, ws.B)
    primal_move = Array(vec(sqrt.(sum(abs2.(ws.DX); dims=1))))
    dual_move = Array(vec(sqrt.(sum(abs2.(ws.DY); dims=1))))
    return primal_move, dual_move
end

function do_batched_restart_gpu!(ws::BatchedWorkspace_gpu)
    total = max(ws.n * ws.B, ws.m * ws.B)
    @cuda threads = 256 blocks = cld(total, 256) do_batched_restart_kernel!(
        ws.X, ws.Y, ws.last_X, ws.last_Y, ws.X_bar, ws.Y_bar,
        ws.restart_flags, ws.n, ws.m, ws.B)
    return nothing
end

function collect_batched_results_gpu(
    ws::BatchedWorkspace_gpu,
    residuals::BatchedResiduals_gpu,
    scaling::BatchedScalingInfo_gpu,
    status::Vector{String},
    iter::Vector{Int},
    t_start::Float64,
    setup_time::Float64,
    power_time::Float64,
)
    X_gpu = (ws.X_bar ./ reshape(scaling.col_norm, :, 1)) .* reshape(scaling.b_scale, 1, :)
    Y_gpu = (ws.Y_bar ./ reshape(scaling.row_norm, :, 1)) .* reshape(scaling.c_scale, 1, :)
    Z_gpu = (ws.Z_bar .* reshape(scaling.col_norm, :, 1)) .* reshape(scaling.c_scale, 1, :)
    solve_time = time() - t_start
    return BatchedHPRLPResults(
        Array(X_gpu),
        Array(Y_gpu),
        Array(Z_gpu),
        Array(residuals.primal_obj),
        Array(residuals.kkt_error),
        Array(residuals.rel_gap),
        status,
        iter,
        setup_time + solve_time,
        setup_time,
        solve_time,
        power_time,
    )
end