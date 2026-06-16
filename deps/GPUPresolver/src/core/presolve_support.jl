"""
Shared helpers needed by presolve/postsolve without loading the full solver pipeline.
"""

function setup_gpu_model(model::LP_info_cpu; device_number::Integer=0, verbose::Bool=true)
    CUDA.device!(device_number)
    t_start = time()
    if verbose
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
        Int32(0),
        CUDA.zeros(Int32, size(model.AT, 1)),
    )
    CUDA.synchronize()

    if verbose
        println(@sprintf("COPY TO GPU time: %.2f seconds", time() - t_start))
    end

    return model_gpu
end

function setup_gpu_model(model::LP_info_cpu, params::GPUPresolverParameters)
    return setup_gpu_model(model; device_number=params.device_number, verbose=params.verbose)
end

function setup_gpu_qp_model(
    Q::SparseMatrixCSC{Float64,<:Integer},
    A::SparseMatrixCSC{Float64,<:Integer},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64;
    device_number::Integer=0,
    verbose::Bool=true,
)
    CUDA.device!(device_number)
    t_start = time()
    if verbose
        println("COPY QP TO GPU ...")
    end

    q_diag = Vector{Float64}(undef, size(Q, 1))
    for j in eachindex(q_diag)
        q_diag[j] = Q[j, j]
    end

    qp_gpu = QP_info_gpu(
        CuSparseMatrixCSR(A),
        CuSparseMatrixCSR(transpose(A)),
        CuSparseMatrixCSR(Q),
        CuSparseMatrixCSR(transpose(Q)),
        CuVector(c),
        CuVector(q_diag),
        CuVector(AL),
        CuVector(AU),
        CuVector(l),
        CuVector(u),
        obj_constant,
        Int32(0),
        CUDA.zeros(Int32, size(A, 2)),
    )
    CUDA.synchronize()

    if verbose
        println(@sprintf("COPY QP TO GPU time: %.2f seconds", time() - t_start))
    end

    return qp_gpu
end

function copy_qp_model_to_cpu(qp::QP_info_gpu)
    Q_cpu = SparseMatrixCSC(qp.Q)
    A_cpu = SparseMatrixCSC(qp.A)
    return (
        Q_cpu,
        A_cpu,
        Array(qp.c),
        Array(qp.AL),
        Array(qp.AU),
        Array(qp.l),
        Array(qp.u),
        qp.obj_constant,
    )
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
    AL_nInf = copy(model.AL)
    AU_nInf = copy(model.AU)

    ALh[ALh.==-Inf] .= -1.0e100
    AUh[AUh.==Inf] .= 1.0e100
    lh[lh.==-Inf] .= -1.0e100
    uh[uh.==Inf] .= 1.0e100
    AL_nInf[AL_nInf.==-Inf] .= 0.0
    AU_nInf[AU_nInf.==Inf] .= 0.0

    @. yh = ifelse((AUh .== 1e100) & (ALh .== -1e100), 0.0,
        ifelse(AUh .== 1e100, max(yh, 0.0),
            ifelse(ALh .== -1e100, min(yh, 0.0), yh)))

    @. zh = ifelse((uh .== 1e100) & (lh .== -1e100), 0.0,
        ifelse(uh .== 1e100, max(zh, 0.0),
            ifelse(lh .== -1e100, min(zh, 0.0), zh)))

    Ax = model.A * xh
    ATy = model.AT * yh

    norm_b = 1.0 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    norm_c = 1.0 + norm(model.c)

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
