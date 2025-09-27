# the function to compute z = a * x + b * y
function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64}, b::Float64, y::CuDeviceVector{Float64}, z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float64, x::CuVector{Float64}, b::Float64, y::CuVector{Float64}, z::CuVector{Float64}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
end

function combined_kernel_x_z_1!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    x0::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    n::Int)

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            xi = x[i]
            ATy_ci = ATy[i] - c[i]
            z_temp = xi + sigma * ATy_ci
            li = l[i]
            ui = u[i]
            # branchless clamp: clamp(z_temp, li, ui)
            xbar = min(max(z_temp, li), ui)
            zbar = (xbar - z_temp) / sigma
            xhat = 2 * xbar - xi
            xnew = muladd(fact2, xhat, fact1 * x0[i])  # fused multiply-add
            dx[i] = xbar - xhat
            z_bar[i] = zbar
            x_bar[i] = xbar
            x_hat[i] = xhat
            x[i] = xnew
        end
    end
    return
end

# the kernel function to update x_bar, Steps 7, 9, and 10 in Algorithm 2
function combined_kernel_x_z_2!(
    x::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    x0::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            xi = x[i]
            li = l[i]
            ui = u[i]
            z_temp = xi + sigma * (ATy[i] - c[i])
            xbar = min(max(z_temp, li), ui)           # branchless clamp
            xhat = 2 * xbar - xi
            xnew = muladd(fact2, xhat, fact1 * x0[i])  # fused multiply-add
            x_hat[i] = xhat
            x[i] = xnew
        end
    end
    return
end

function update_x_z_gpu!(ws::HPRLP_workspace_gpu, fact1::Float64, fact2::Float64)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha, ws.spmv_AT.desc_AT, ws.spmv_AT.desc_y, ws.spmv_AT.beta, ws.spmv_AT.desc_ATy,
        ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
    if ws.to_check
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z_1!(ws.dx, ws.x, ws.z_bar, ws.x_bar, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.last_x, fact1, fact2, ws.n)
    else
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) combined_kernel_x_z_2!(ws.x, ws.x_hat, ws.l, ws.u, ws.sigma, ws.ATy, ws.c, ws.last_x, fact1, fact2, ws.n)
    end
end

function update_x_z_cpu!(ws::HPRLP_workspace_cpu, fact1::Float64, fact2::Float64)
    mul!(ws.ATy, ws.AT, ws.y)
    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    x0 = ws.last_x
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    ATy = ws.ATy
    c = ws.c
    dx = ws.dx
    if ws.to_check
        @simd for i in eachindex(x)
            @inbounds begin
               z_bar[i] = x[i] + sigma * (ATy[i] - c[i])
               x_bar[i] = z_bar[i] < l[i] ? l[i] : (z_bar[i] > u[i] ? u[i] : z_bar[i])
               x_hat[i] = 2 * x_bar[i] - x[i]
               dx[i] = x_bar[i] - x_hat[i]
               x[i] = fact1 * x0[i] + fact2 * x_hat[i]
               z_bar[i] = (x_bar[i] - z_bar[i]) / sigma 
            end
        end
    else
        @simd for i in eachindex(x)
            @inbounds begin
               x_bar[i] = x[i] + sigma * (ATy[i] - c[i])
               x_bar[i] = x_bar[i] < l[i] ? l[i] : (x_bar[i] > u[i] ? u[i] : x_bar[i])
               x_hat[i] = 2 * x_bar[i] - x[i]
               x[i] = fact1 * x0[i] + fact2 * x_hat[i]
            end
        end
    end
end

function update_y_kernel_1!(dy::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    y_obj::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    y0::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int)

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            yi = y[i]
            ai = Ax[i]
            li = AL[i]
            ui = AU[i]
            y0i = y0[i]
            v = ai - fact1 * yi
            # branchless projection difference
            d = max(li - v, min(ui - v, 0.0))
            yb = fact2 * d
            yh = 2 * yb - yi
            ynew = muladd(Halpern_fact2, yh, Halpern_fact1 * y0i)  # fused multiply-add
            dy[i] = yb - yh
            y_bar[i] = yb
            y_obj[i] = v + d
            y[i] = ynew
        end
    end
    return
end

function update_y_kernel_2!(
    y::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    y0::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int)

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            yi = y[i]
            ai = Ax[i]
            li = AL[i]
            ui = AU[i]
            y0i = y0[i]
            v = ai - fact1 * yi
            # branchless projection difference
            d = max(li - v, min(ui - v, 0.0))
            yb = fact2 * d
            yh = 2 * yb - yi
            ynew = muladd(Halpern_fact2, yh, Halpern_fact1 * y0i)  # fused multiply-add
            y[i] = ynew
        end
    end
    return
end

function update_y_gpu!(ws::HPRLP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_x_hat, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
        ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    fact1 = ws.lambda_max * ws.sigma
    fact2 = 1.0 / fact1
    if ws.to_check
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel_1!(ws.dy, ws.y_bar, ws.y, ws.y_obj, ws.AL, ws.AU, ws.Ax, fact1, fact2, ws.last_y, Halpern_fact1, Halpern_fact2, ws.m)
    else
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel_2!(ws.y,ws.AL, ws.AU, ws.Ax, fact1, fact2, ws.last_y, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

function update_y_cpu!(ws::HPRLP_workspace_cpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    mul!(ws.Ax, ws.A, ws.x_hat)
    fact1 = ws.lambda_max * ws.sigma
    fact2 = 1.0 / fact1
    y = ws.y
    y_obj = ws.y_obj
    AL = ws.AL
    AU = ws.AU
    y0 = ws.last_y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    Ax = ws.Ax
    dy = ws.dy
    @simd for i in eachindex(y)
        @inbounds begin
            yi = y[i]
            # scaled residual
            v = Ax[i] - fact1 * yi
            d = max(AL[i] - v, min(AU[i] - v, 0.0))
            # for computing the dual obj function value
            y_obj[i] = v + d
            # scaled update
            yb = fact2 * d
            y_bar[i] = yb
            # branchless y_hat
            y_hat[i] = 2 * yb - yi
            dy[i] = yb - y_hat[i]
            # Halpern update
            y[i] = Halpern_fact1 * y0[i] + Halpern_fact2 * y_hat[i]
        end
    end
    return
end

# the kernel function to compute the dual residuals, ||c - A^T y - z||
function compute_Rd_kernel!(col_norm::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds Rdi = c[i] - ATy[i] - z[i]
        @inbounds Rd[i] = Rdi * col_norm[i]
    end
    return
end

function compute_Rd_gpu!(ws::HPRLP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha, ws.spmv_AT.desc_AT, ws.spmv_AT.desc_y_bar, ws.spmv_AT.beta, ws.spmv_AT.desc_ATy,
        ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(sc.col_norm, ws.ATy, ws.z_bar, ws.c, ws.Rd, ws.n)
end

function compute_err_Rd_cpu!(ws::HPRLP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Rd, ws.AT, ws.y_bar)
    c = ws.c
    Rd = ws.Rd
    z_bar = ws.z_bar
    col_norm = sc.col_norm
    @simd for i in eachindex(Rd)
        @inbounds Rd[i] = Rd[i] + z_bar[i] - c[i]
        @inbounds Rd[i] *= col_norm[i]
    end
end



function compute_err_lu_kernel!(col_norm::CuDeviceVector{Float64},
    dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0)
        @inbounds dx[i] /= col_norm[i]
    end
    return
end


# the kernel function to compute the primal residuals, ||\Pi_D(b - Ax)||
@inline function compute_err_Rp_kernel!(row_norm::CuDeviceVector{Float64},
    Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64}, AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64}, m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            # load into registers
            v = Ax[i]
            low = AL[i]
            high = AU[i]
            row_normi = row_norm[i]
            Rpi = max(min(high - v, 0), low - v)
            Rp[i] = row_normi * Rpi
        end
    end
    return
end


function compute_err_Rp_gpu!(ws::HPRLP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha, ws.spmv_A.desc_A, ws.spmv_A.desc_x_bar, ws.spmv_A.beta, ws.spmv_A.desc_Ax,
        ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_err_Rp_kernel!(sc.row_norm, ws.Rp, ws.AL, ws.AU, ws.Ax, ws.m)
end

function compute_err_Rp_cpu!(ws::HPRLP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Ax, ws.A, ws.x_bar)
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    Rp = ws.Rp
    row_norm = sc.row_norm

    # Parallelize and eliminate bounds checks & branching
    @simd for i in eachindex(Rp)
        @inbounds begin
            v = Ax[i]
            low = AL[i]
            high = AU[i]
            #   • If v < AL: AL−v > 0 and v−AU ≤ 0 → diff = AL−v
            #   • If v > AU: v−AU > 0 and AL−v ≤ 0 → diff = v−AU
            #   • Else both are ≤ 0 → diff = 0
            Rp[i] = max(min(high - v, 0), low - v)
            Rp[i] *= row_norm[i]
        end
    end
end