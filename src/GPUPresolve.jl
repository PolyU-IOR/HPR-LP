module GPUPresolve

using CUDA
using LinearAlgebra
using SparseArrays
using CUDA: CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR
using ..HPRLP: LP_info_cpu, LP_info_gpu, HPRLP_parameters

include("presolve/postsolve_tape.jl")
include("presolve/presolve_structs.jl")
include("presolve/rule_helpers.jl")
include("presolve/gpu_presolve_kernels.jl")
include("presolve/checks.jl")
include("presolve/rules/rule_close_bounds.jl")
include("presolve/rules/rule_empty_rows.jl")
include("presolve/rules/rule_singleton_rows.jl")
include("presolve/rules/rule_activity_checks.jl")
include("presolve/rules/rule_primal_propagation.jl")
include("presolve/rules/rule_parallel_rows.jl")
include("presolve/rules/rule_empty_cols.jl")
include("presolve/rules/rule_singleton_cols.jl")
include("presolve/rules/rule_doubleton_eq.jl")
include("presolve/rules/rule_dual_fix.jl")
include("presolve/rules/rule_parallel_cols.jl")
include("presolve/rules/rule_redundant_bounds.jl")
include("presolve/gpu_presolve.jl")
include("presolve/gpu_postsolve.jl")

Base.@kwdef struct Settings
    verbose::Bool = true
    device_number::Int = 0
    presolve_params::Union{Nothing,PresolveParams} = nothing
end

mutable struct PresolveState
    record::Union{Nothing,PresolveRecord_gpu}
    original_model_gpu::Union{Nothing,LP_info_gpu}
    function PresolveState(record::PresolveRecord_gpu, original_model_gpu::LP_info_gpu)
        obj = new(record, original_model_gpu)
        finalizer(free_presolve_state!, obj)
        return obj
    end
end

function free_presolve_state!(state::PresolveState)
    state.record = nothing
    state.original_model_gpu = nothing
    return nothing
end

function _backend_params(settings::Settings)
    params = HPRLP_parameters()
    params.use_gpu = true
    params.verbose = settings.verbose
    params.device_number = settings.device_number
    return params
end

function _backend_presolve_params(settings::Settings)
    pparams = isnothing(settings.presolve_params) ? PresolveParams() : deepcopy(settings.presolve_params)
    pparams.verbose = pparams.verbose || settings.verbose
    return pparams
end

function run_presolve(
    model::LP_info_gpu;
    settings::Union{Nothing,Settings}=nothing,
)
    stgs = something(settings, Settings())
    if !CUDA.functional()
        stgs.verbose && @warn "GPU presolve requested but CUDA is not functional; skipping presolve."
        return nothing, nothing
    end

    params = _backend_params(stgs)
    try
        CUDA.device!(stgs.device_number)
    catch err
        stgs.verbose && @warn "GPU presolve requested but CUDA device $(stgs.device_number) is unavailable; skipping presolve." exception=(err, catch_backtrace())
        return nothing, nothing
    end

    presolve_params = _backend_presolve_params(stgs)
    presolve_params.record_postsolve_tape = true
    reduced_model_gpu, record = presolve_gpu(model, params; presolve_params=presolve_params)
    return PresolveState(record, model), reduced_model_gpu
end

function run_postsolve(
    state::PresolveState,
    x_red::Vector{Float64},
    y_red::Vector{Float64},
    z_red::Vector{Float64},
    ;
    presolve_params=nothing,
)
    if isnothing(state.record) || isnothing(state.original_model_gpu)
        error("Presolve state has been freed or is invalid.")
    end

    x_org, y_org, z_org = postsolve_gpu(
        x_red,
        y_red,
        z_red,
        state.record;
        presolve_params=presolve_params,
        original_model_gpu=state.original_model_gpu,
    )

    return Array(x_org), Array(y_org), Array(z_org)
end

end # module