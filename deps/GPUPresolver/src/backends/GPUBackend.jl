module GPUBackend

using CUDA
using LinearAlgebra
using SparseArrays
using CUDA: CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR
using ..GPUPresolver: LP_info_cpu, LP_info_gpu, QP_info_gpu, GPUPresolverParameters, setup_gpu_model

const _PRESOLVE_DIR = joinpath(@__DIR__, "..", "presolve")
const _PRESOLVE_RULES_DIR = joinpath(_PRESOLVE_DIR, "rules")
const _PRESOLVE_LP_RULES_DIR = joinpath(_PRESOLVE_RULES_DIR, "lp_rules")

include(joinpath(_PRESOLVE_DIR, "postsolve_tape.jl"))
include(joinpath(_PRESOLVE_DIR, "presolve_structs.jl"))
include(joinpath(_PRESOLVE_DIR, "rule_helpers.jl"))
include(joinpath(_PRESOLVE_DIR, "gpu_presolve_kernels.jl"))
include(joinpath(_PRESOLVE_DIR, "checks.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_close_bounds.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_empty_rows.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_singleton_rows.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_activity_checks.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_primal_propagation.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_parallel_rows.jl"))
include(joinpath(_PRESOLVE_RULES_DIR, "rule_redundant_bounds.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_empty_cols.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_singleton_cols.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_doubleton_eq.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_dual_fix.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_parallel_cols.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_fme_projection.jl"))
include(joinpath(_PRESOLVE_LP_RULES_DIR, "lp_rule_structural_l1_substitution.jl"))
include(joinpath(_PRESOLVE_DIR, "gpu_presolve.jl"))
include(joinpath(_PRESOLVE_DIR, "gpu_presolve_qp.jl"))
include(joinpath(_PRESOLVE_DIR, "gpu_postsolve.jl"))

Base.@kwdef struct Settings
    verbose::Bool = true
    device_number::Int = 0
    presolve_params::Union{Nothing,PresolveParams} = nothing
end

mutable struct PresolveState
    record::Union{Nothing,PresolveRecord_gpu}
    original_model_gpu::Union{Nothing,LP_info_gpu}
    presolve_core_time::Float64
    move_to_cpu_time::Float64
    function PresolveState(record::PresolveRecord_gpu, original_model_gpu::LP_info_gpu)
        obj = new(record, original_model_gpu, 0.0, 0.0)
        finalizer(free_presolve_state!, obj)
        return obj
    end
    function PresolveState(
        record::PresolveRecord_gpu,
        original_model_gpu::LP_info_gpu,
        presolve_core_time::Float64,
        move_to_cpu_time::Float64,
    )
        obj = new(record, original_model_gpu, presolve_core_time, move_to_cpu_time)
        finalizer(free_presolve_state!, obj)
        return obj
    end
end

function free_presolve_state!(state::PresolveState)
    state.record = nothing
    state.original_model_gpu = nothing
    state.presolve_core_time = 0.0
    state.move_to_cpu_time = 0.0
    return nothing
end

function _backend_presolve_params(settings::Settings)
    pparams = isnothing(settings.presolve_params) ? PresolveParams() : deepcopy(settings.presolve_params)
    pparams.verbose = pparams.verbose || settings.verbose
    return pparams
end

function _reduced_cpu_model(lp::LP_info_gpu)
    A_cpu = SparseMatrixCSC(lp.A)
    return LP_info_cpu(
        A_cpu,
        SparseMatrixCSC(lp.AT),
        Array(lp.c),
        Array(lp.AL),
        Array(lp.AU),
        Array(lp.l),
        Array(lp.u),
        lp.obj_constant,
    )
end

function run_presolve(
    model::LP_info_cpu;
    settings::Union{Nothing,Settings}=nothing,
)
    stgs = something(settings, Settings())
    if !CUDA.functional()
        stgs.verbose && @warn "GPU presolve requested but CUDA is not functional; skipping presolve."
        return nothing, nothing
    end

    try
        CUDA.device!(stgs.device_number)
    catch err
        stgs.verbose && @warn "GPU presolve requested but CUDA device $(stgs.device_number) is unavailable; skipping presolve." exception=(err, catch_backtrace())
        return nothing, nothing
    end

    original_model_gpu = setup_gpu_model(model; device_number=stgs.device_number, verbose=stgs.verbose)
    presolve_params = _backend_presolve_params(stgs)
    if presolve_params.enable_structural_l1_substitution &&
       presolve_params.structural_l1_allow_main_flow_without_tape
        stgs.verbose && @debug "Running structural_l1_substitution in the main presolve flow with postsolve tape disabled."
        presolve_params.record_postsolve_tape = false
        presolve_params.record_postsolve_tape_cpu = false
    elseif presolve_params.enable_fme_projection && presolve_params.fme_allow_main_flow_without_tape
        stgs.verbose && @warn "Running experimental FME projection in the main presolve flow with postsolve tape disabled."
        presolve_params.record_postsolve_tape = false
        presolve_params.record_postsolve_tape_cpu = false
    else
        presolve_params.record_postsolve_tape = true
    end
    t_core_start = time()
    reduced_model_gpu, record = presolve_gpu(original_model_gpu; presolve_params=presolve_params, verbose=stgs.verbose)
    CUDA.synchronize()
    presolve_core_time = time() - t_core_start
    t_copy_back_start = time()
    reduced_model = _reduced_cpu_model(reduced_model_gpu)
    copy_back_time = time() - t_copy_back_start
    return PresolveState(record, original_model_gpu, presolve_core_time, copy_back_time), reduced_model
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
