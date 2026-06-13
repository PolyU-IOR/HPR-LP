mutable struct BatchedSharedMatrix_gpu
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    m::Int
    n::Int
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    lambda_max::Float64
end

mutable struct BatchedLPData_gpu
    C::CuMatrix{Float64}
    AL::CuMatrix{Float64}
    AU::CuMatrix{Float64}
    L::CuMatrix{Float64}
    U::CuMatrix{Float64}
    obj_constants::CuVector{Float64}
    batch_size::Int
end

mutable struct BatchedScalingInfo_gpu
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    b_scale::CuVector{Float64}
    c_scale::CuVector{Float64}
    norm_b::CuVector{Float64}
    norm_c::CuVector{Float64}
    norm_b_org::CuVector{Float64}
    norm_c_org::CuVector{Float64}
end

mutable struct BatchedWorkspace_gpu
    m::Int
    n::Int
    B::Int
    X::CuMatrix{Float64}
    X_hat::CuMatrix{Float64}
    X_bar::CuMatrix{Float64}
    DX::CuMatrix{Float64}
    Y::CuMatrix{Float64}
    Y_hat::CuMatrix{Float64}
    Y_bar::CuMatrix{Float64}
    DY::CuMatrix{Float64}
    Y_obj::CuMatrix{Float64}
    Z_bar::CuMatrix{Float64}
    RP::CuMatrix{Float64}
    RD::CuMatrix{Float64}
    ATY::CuMatrix{Float64}
    AX::CuMatrix{Float64}
    last_X::CuMatrix{Float64}
    last_Y::CuMatrix{Float64}
    sigma::CuVector{Float64}
    lambda_max::Float64
    active::CuVector{Bool}
    restart_flags::CuVector{Bool}
    halpern_fact1::CuVector{Float64}
    halpern_fact2::CuVector{Float64}
end

mutable struct BatchedRestartInfo_gpu
    restart_flag::Vector{Int}
    first_restart::Vector{Bool}
    last_gap::Vector{Float64}
    current_gap::Vector{Float64}
    save_gap::Vector{Float64}
    best_gap::Vector{Float64}
    best_sigma::Vector{Float64}
    inner::Vector{Int}
    sufficient::Vector{Int}
    necessary::Vector{Int}
    long::Vector{Int}
    times::Vector{Int}
    halpern_fact1_host::Vector{Float64}
    halpern_fact2_host::Vector{Float64}
    restart_flags_host::Vector{Bool}
    sigma_host::Vector{Float64}
end

mutable struct BatchedResiduals_gpu
    primal_obj::CuVector{Float64}
    dual_obj::CuVector{Float64}
    err_Rp::CuVector{Float64}
    err_Rd::CuVector{Float64}
    rel_gap::CuVector{Float64}
    kkt_error::CuVector{Float64}
end

mutable struct BatchedHPRLPResults
    X::Matrix{Float64}
    Y::Matrix{Float64}
    Z::Matrix{Float64}
    primal_obj::Vector{Float64}
    residuals::Vector{Float64}
    gap::Vector{Float64}
    status::Vector{String}
    iter::Vector{Int}
    time::Float64
    setup_time::Float64
    solve_time::Float64
    power_time::Float64
end