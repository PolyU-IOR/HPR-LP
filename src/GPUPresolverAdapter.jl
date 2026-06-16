using GPUPresolver

struct HPRLPGPUBackendState <: GPUPresolver.AbstractPresolveState
    inner::GPUPresolver.GPUBackendStateHandle
    presolve_params::Any
end

function load_gpu_presolve_setup(path::AbstractString)
    return GPUPresolver.load_presolve_setup(path)
end

function load_gpu_presolve_params(path::AbstractString)
    return load_gpu_presolve_setup(path).presolve_params
end

function _default_gpu_presolve_config_path()
    project_config_path = normpath(joinpath(@__DIR__, "..", "deps", "GPUPresolver", "config", "default.toml"))
    if isfile(project_config_path)
        return project_config_path
    end

    package_entry = pathof(GPUPresolver)
    isnothing(package_entry) && return nothing
    package_config_path = normpath(joinpath(dirname(package_entry), "..", "config", "default.toml"))
    return isfile(package_config_path) ? package_config_path : nothing
end

function _resolve_gpu_presolve_params(presolve_params)
    !isnothing(presolve_params) && return presolve_params
    config_path = _default_gpu_presolve_config_path()
    isnothing(config_path) && return nothing
    return load_gpu_presolve_setup(config_path).presolve_params
end

function _gpu_presolve_coverage_mps_paths()
    raw = strip(get(ENV, "GPUPRESOLVER_WARMUP_COVERAGE_MPS", ""))
    isempty(raw) && return String[]
    return [strip(path) for path in split(raw, ",") if !isempty(strip(path))]
end

function _run_gpu_presolve_coverage_warmup!(
    params::HPRLP_parameters,
    resolved_presolve_params,
    coverage_config::GPUPresolver.PresolveConfig,
)
    coverage_paths = _gpu_presolve_coverage_mps_paths()
    coverage_params = deepcopy(params)
    coverage_params.verbose = false

    if !isempty(coverage_paths)
        for path in coverage_paths
            isfile(path) || continue
            coverage_model = build_from_mps(path, false)
            coverage_state = nothing
            try
                _, coverage_state, _ = run_external_gpu_presolve(
                    coverage_model,
                    coverage_params;
                    presolve_params=resolved_presolve_params,
                )
            finally
                if !isnothing(coverage_state)
                    free_external_gpu_presolve_state!(coverage_state)
                end
            end
        end
        if !GPUPresolver._gpu_presolve_env_bool("GPUPRESOLVER_WARMUP_SYNTHETIC_COVERAGE", true)
            return nothing
        end
    end

    for coverage_model in GPUPresolver._gpu_presolve_warmup_models(:coverage)
        coverage_result = GPUPresolver.run_presolve(
            coverage_model;
            config=coverage_config,
        )
        if !isnothing(coverage_result.state)
            GPUPresolver.free_presolve_state!(coverage_result.state)
        end
    end
    return nothing
end

function warmup_external_gpu_presolve!(
    params::HPRLP_parameters;
    presolve_params=nothing,
    model=nothing,
)
    mode = GPUPresolver._gpu_presolve_warmup_mode()
    if mode == :actual && !isnothing(model)
        GPUPresolver._GPU_PRESOLVE_WARMED[] && return nothing
        state = nothing
        warmup_params = deepcopy(params)
        warmup_params.verbose = false
        resolved_presolve_params = _resolve_gpu_presolve_params(presolve_params)
        coverage_config = GPUPresolver.PresolveConfig(
            backend="GPU",
            verbose=warmup_params.verbose,
            device_number=warmup_params.device_number,
            presolve_params=resolved_presolve_params,
        )
        try
            _, state, _ = run_external_gpu_presolve(
                model,
                warmup_params;
                presolve_params=presolve_params,
            )
            if GPUPresolver._gpu_presolve_env_bool("GPUPRESOLVER_WARMUP_COVERAGE", true)
                _run_gpu_presolve_coverage_warmup!(
                    warmup_params,
                    resolved_presolve_params,
                    coverage_config,
                )
            end
            GPUPresolver._GPU_PRESOLVE_WARMED[] = true
        catch err
            params.verbose && @warn "Actual-model GPU presolve warmup failed; falling back to configured warmup." exception=(err, catch_backtrace())
            GPUPresolver.warmup_gpu_presolve!(
                coverage_config;
                presolve_params=resolved_presolve_params,
            )
        finally
            if !isnothing(state)
                free_external_gpu_presolve_state!(state)
            end
        end
        return nothing
    end

    resolved_presolve_params = _resolve_gpu_presolve_params(presolve_params)
    config = GPUPresolver.PresolveConfig(
        backend="GPU",
        verbose=false,
        device_number=params.device_number,
        presolve_params=resolved_presolve_params,
    )
    return GPUPresolver.warmup_gpu_presolve!(
        config;
        presolve_params=resolved_presolve_params,
    )
end

function _cpu_model_from_hprlp(model::LP_info_cpu)
    return GPUPresolver.build_from_Abc(
        SparseMatrixCSC(model.A),
        copy(model.c),
        copy(model.AL),
        copy(model.AU),
        copy(model.l),
        copy(model.u),
        model.obj_constant,
    )
end

function _cpu_model_from_hprlp(model::LP_info_gpu)
    return GPUPresolver.build_from_Abc(
        SparseMatrixCSC(model.A),
        Array(model.c),
        Array(model.AL),
        Array(model.AU),
        Array(model.l),
        Array(model.u),
        model.obj_constant,
    )
end

function _gpu_model_from_hprlp(model::LP_info_gpu)
    return GPUPresolver.LP_info_gpu(
        model.A,
        model.AT,
        model.c,
        model.AL,
        model.AU,
        model.l,
        model.u,
        model.obj_constant,
        model.AT_leading_slack,
        model.AT_slack_after,
    )
end

function _cpu_model_to_hprlp(model::GPUPresolver.LP_info_cpu)
    A = SparseMatrixCSC(model.A)
    return LP_info_cpu(
        A,
        SparseMatrixCSC(model.AT),
        copy(model.c),
        copy(model.AL),
        copy(model.AU),
        copy(model.l),
        copy(model.u),
        model.obj_constant,
    )
end

function _gpu_model_to_hprlp(model::GPUPresolver.LP_info_gpu)
    return LP_info_gpu(
        model.A,
        model.AT,
        model.c,
        model.AL,
        model.AU,
        model.l,
        model.u,
        model.obj_constant,
        model.AT_leading_slack,
        model.AT_slack_after,
    )
end

function _gpu_presolve_params_for_hprlp(
    config::GPUPresolver.PresolveConfig;
    require_postsolve_tape::Bool=false,
)
    settings = GPUPresolver.GPUBackend.Settings(
        verbose=config.verbose,
        device_number=config.device_number,
        presolve_params=config.presolve_params,
    )
    presolve_params = GPUPresolver.GPUBackend._backend_presolve_params(settings)
    if require_postsolve_tape &&
       presolve_params.enable_structural_l1_substitution &&
       presolve_params.structural_l1_allow_main_flow_without_tape
        presolve_params.record_postsolve_tape = true
        presolve_params.record_postsolve_tape_cpu = false
    elseif require_postsolve_tape &&
           presolve_params.enable_fme_projection &&
           presolve_params.fme_allow_main_flow_without_tape
        config.verbose && @warn "Disabling FME projection because postsolve is enabled and the configured main-flow FME rule does not record tape."
        presolve_params.enable_fme_projection = false
        presolve_params.record_postsolve_tape = true
        presolve_params.record_postsolve_tape_cpu = false
    elseif presolve_params.enable_structural_l1_substitution &&
       presolve_params.structural_l1_allow_main_flow_without_tape
        config.verbose && @debug "Running experimental structural_l1_substitution in the main presolve flow with postsolve tape disabled."
        presolve_params.record_postsolve_tape = false
        presolve_params.record_postsolve_tape_cpu = false
    elseif presolve_params.enable_fme_projection && presolve_params.fme_allow_main_flow_without_tape
        config.verbose && @warn "Running experimental FME projection in the main presolve flow with postsolve tape disabled."
        presolve_params.record_postsolve_tape = false
        presolve_params.record_postsolve_tape_cpu = false
    else
        presolve_params.record_postsolve_tape = true
    end
    return presolve_params
end

function _run_external_gpu_presolve_direct(
    model::LP_info_cpu,
    config::GPUPresolver.PresolveConfig,
    require_postsolve_tape::Bool=false,
)
    if !CUDA.functional()
        config.verbose && @warn "GPU presolve requested but CUDA is not functional; skipping presolve."
        return model, nothing, 0.0
    end

    try
        CUDA.device!(config.device_number)
    catch err
        config.verbose && @warn "GPU presolve requested but CUDA device $(config.device_number) is unavailable; skipping presolve." exception=(err, catch_backtrace())
        return model, nothing, 0.0
    end

    original_model_gpu = GPUPresolver.setup_gpu_model(
        _cpu_model_from_hprlp(model);
        device_number=config.device_number,
        verbose=config.verbose,
    )
    presolve_params = _gpu_presolve_params_for_hprlp(config; require_postsolve_tape=require_postsolve_tape)
    t_core_start = time()
    reduced_model_gpu, record = GPUPresolver.GPUBackend.presolve_gpu(
        original_model_gpu;
        presolve_params=presolve_params,
        verbose=config.verbose,
    )
    CUDA.synchronize()
    presolve_core_time = time() - t_core_start
    if isnothing(record)
        return model, nothing, 0.0
    end

    raw_state = GPUPresolver.GPUBackend.PresolveState(
        record,
        original_model_gpu,
        presolve_core_time,
        0.0,
    )
    state = HPRLPGPUBackendState(
        GPUPresolver.GPUBackendStateHandle(raw_state),
        presolve_params,
    )
    return _gpu_model_to_hprlp(reduced_model_gpu), state, presolve_core_time
end

function _run_external_gpu_presolve_direct(
    model::LP_info_gpu,
    config::GPUPresolver.PresolveConfig,
    require_postsolve_tape::Bool=false,
)
    if !CUDA.functional()
        config.verbose && @warn "GPU presolve requested but CUDA is not functional; skipping presolve."
        return model, nothing, 0.0
    end

    try
        CUDA.device!(config.device_number)
    catch err
        config.verbose && @warn "GPU presolve requested but CUDA device $(config.device_number) is unavailable; skipping presolve." exception=(err, catch_backtrace())
        return model, nothing, 0.0
    end

    original_model_gpu = _gpu_model_from_hprlp(model)
    presolve_params = _gpu_presolve_params_for_hprlp(config; require_postsolve_tape=require_postsolve_tape)
    t_core_start = time()
    reduced_model_gpu, record = GPUPresolver.GPUBackend.presolve_gpu(
        original_model_gpu;
        presolve_params=presolve_params,
        verbose=config.verbose,
    )
    CUDA.synchronize()
    presolve_core_time = time() - t_core_start
    if isnothing(record)
        return model, nothing, 0.0
    end

    raw_state = GPUPresolver.GPUBackend.PresolveState(
        record,
        original_model_gpu,
        presolve_core_time,
        0.0,
    )
    state = HPRLPGPUBackendState(
        GPUPresolver.GPUBackendStateHandle(raw_state),
        presolve_params,
    )
    return _gpu_model_to_hprlp(reduced_model_gpu), state, presolve_core_time
end

function run_external_gpu_presolve(
    model::Union{LP_info_cpu,LP_info_gpu},
    params::HPRLP_parameters;
    presolve_params=nothing,
)
    resolved_presolve_params = _resolve_gpu_presolve_params(presolve_params)
    config = GPUPresolver.PresolveConfig(
        backend="GPU",
        verbose=params.verbose,
        device_number=params.device_number,
        presolve_params=resolved_presolve_params,
    )
    if params.use_gpu
        return _run_external_gpu_presolve_direct(
            model,
            config,
            params.use_postsolve,
        )
    end

    result = GPUPresolver.run_presolve(_cpu_model_from_hprlp(model); config=config)

    if result.status != "OK" || isnothing(result.state)
        return model, nothing, 0.0
    end

    reduced_problem = result.reduced_problem
    reduced_problem isa GPUPresolver.LPProblem ||
        error("GPUPresolver returned unsupported reduced problem type $(typeof(reduced_problem)).")

    return _cpu_model_to_hprlp(reduced_problem.model), result.state, result.presolve_time
end

function run_external_gpu_postsolve(
    presolve_state::HPRLPGPUBackendState,
    x_red::Vector{Float64},
    y_red::Vector{Float64},
    z_red::Vector{Float64};
    presolve_params=nothing,
)
    return GPUPresolver.run_postsolve(
        presolve_state.inner,
        x_red,
        y_red,
        z_red;
        presolve_params=presolve_state.presolve_params,
    )
end

function run_external_gpu_postsolve(
    presolve_state::GPUPresolver.AbstractPresolveState,
    x_red::Vector{Float64},
    y_red::Vector{Float64},
    z_red::Vector{Float64};
    presolve_params=nothing,
)
    resolved_presolve_params = _resolve_gpu_presolve_params(presolve_params)
    return GPUPresolver.run_postsolve(
        presolve_state,
        x_red,
        y_red,
        z_red;
        presolve_params=resolved_presolve_params,
    )
end

function free_external_gpu_presolve_state!(
    presolve_state::HPRLPGPUBackendState,
)
    return free_external_gpu_presolve_state!(presolve_state.inner)
end

function free_external_gpu_presolve_state!(
    presolve_state::GPUPresolver.AbstractPresolveState,
)
    return GPUPresolver.free_presolve_state!(presolve_state)
end
