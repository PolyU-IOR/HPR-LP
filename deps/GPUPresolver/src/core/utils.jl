# The function to read the LP problem from the file and formulate the LP problem
function formulation(A, c, AL, AU, l, u, obj_constant)
    # Validate dimensions: A is m x n, c/l/u have length n, AL/AU have length m
    m, n = size(A)
    @assert length(c) == n "Dimension mismatch: size(A, 2) = $(n), but length(c) = $(length(c))."
    @assert length(l) == n "Dimension mismatch: size(A, 2) = $(n), but length(l) = $(length(l))."
    @assert length(u) == n "Dimension mismatch: size(A, 2) = $(n), but length(u) = $(length(u))."
    @assert length(AL) == m "Dimension mismatch: size(A, 1) = $(m), but length(AL) = $(length(AL))."
    @assert length(AU) == m "Dimension mismatch: size(A, 1) = $(m), but length(AU) = $(length(AU))."

    standard_lp = LP_info_cpu(A, transpose(A), c, AL, AU, l, u, obj_constant)
    return standard_lp
end

const VALID_PRESOLVE_BACKENDS = ("GPU", "NONE")

function normalize_presolve_backend(backend)
    backend_name = if backend isa Bool
        backend ? "GPU" : "NONE"
    else
        uppercase(String(backend))
    end
    backend_name in VALID_PRESOLVE_BACKENDS || throw(ArgumentError(
        "Unsupported presolve backend $(backend). Expected one of GPU or NONE."))
    return backend_name
end

function set_presolve_backend!(params::GPUPresolverParameters, backend)
    params.presolve = normalize_presolve_backend(backend)
    return params
end

presolve_enabled(params::GPUPresolverParameters) = normalize_presolve_backend(params.presolve) != "NONE"

function apply_gpu_presolve(model::LP_info_cpu, params::GPUPresolverParameters; presolve_params=nothing)
    settings = GPUBackend.Settings(
        verbose=params.verbose,
        device_number=params.device_number,
        presolve_params=presolve_params,
    )
    presolve_state, reduced_model = GPUBackend.run_presolve(model; settings=settings)

    if reduced_model === nothing || presolve_state === nothing
        println("GPU presolve failed or returned nothing.")
        if presolve_state !== nothing
            GPUBackend.free_presolve_state!(presolve_state)
        end
        return model, nothing
    end

    if params.verbose
        println("GPU presolve reduced size: $(size(model.A)) -> $(size(reduced_model.A))")
        println("GPU presolve objective offset: $(reduced_model.obj_constant - model.obj_constant)")
        println(@sprintf("MOVE TO CPU TIME: %.2f seconds", presolve_state.move_to_cpu_time))
    end

    return reduced_model, presolve_state
end

function apply_presolve(model::LP_info_cpu, params::GPUPresolverParameters; presolve_params=nothing)
    backend = normalize_presolve_backend(params.presolve)
    if backend == "GPU"
        return apply_gpu_presolve(model, params; presolve_params=presolve_params)
    end
    return model, nothing
end

"""
    build_from_mps(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)

Build an LP model from an MPS file.

# Arguments
- `filename::AbstractString`: Path to the `.mps` or `.mps.gz` file
- `verbose::Bool`: Enable verbose output (default: true)
- `mpsformat::Symbol`: MPS format hint passed to `MPSReader` (`:auto`, `:fixed`, `:free`)

# Returns
- `LP_info_cpu`: LP model ready to be solved

# Example
```julia
using GPUPresolver

model = build_from_mps("problem.mps")
result = run_presolve(model; config=PresolveConfig(backend="GPU"))
```

See also: [`build_from_Abc`](@ref), [`run_presolve`](@ref)
"""
function build_from_mps(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)
    t_start = time()
    if verbose
        println("READING FILE ... ", filename)
    end
    lp = Logging.with_logger(Logging.NullLogger()) do
        MPSReader.read_mps(filename; keep_names=false, mpsformat=mpsformat)
    end
    read_time = time() - t_start
    if verbose
        println(@sprintf("READING FILE time: %.2f seconds", read_time))
    end

    t_start = time()
    if verbose
        println("FORMULATING LP ...")
    end
    A = sparse(lp.arows, lp.acols, lp.avals, lp.nrow, lp.ncol)
    standard_lp = formulation(A, lp.c, lp.lcon, lp.ucon, lp.lvar, lp.uvar, lp.obj_constant)
    if verbose
        println(@sprintf("FORMULATING LP time: %.2f seconds", time() - t_start))
    end

    return standard_lp
end

function _resolve_qps_mpsformat(mpsformat::Symbol)
    if mpsformat == :auto || mpsformat == :free
        return :free
    elseif mpsformat == :fixed
        return :fixed
    end
    throw(ArgumentError("Unsupported mpsformat=$(mpsformat) for QP reader. Expected :auto, :free, or :fixed."))
end

function _read_qp_mps(filename::AbstractString; mpsformat::Symbol=:auto)
    path = String(filename)
    suffix = lowercase(path)
    if !(endswith(suffix, ".mps") || endswith(suffix, ".mps.gz"))
        throw(ArgumentError("Unsupported file format for QP. Expected .mps or .mps.gz, got: $(filename)"))
    end

    qps = if endswith(suffix, ".gz")
        open(path) do io
            gz = CodecZlib.GzipDecompressorStream(io)
            try
                Logging.with_logger(Logging.NullLogger()) do
                    QPSReader.readqps(gz, mpsformat=_resolve_qps_mpsformat(mpsformat))
                end
            finally
                close(gz)
            end
        end
    else
        open(path) do io
            Logging.with_logger(Logging.NullLogger()) do
                QPSReader.readqps(io, mpsformat=_resolve_qps_mpsformat(mpsformat))
            end
        end
    end

    A = sparse(qps.arows, qps.acols, qps.avals, qps.ncon, qps.nvar)
    Q = sparse(qps.qrows, qps.qcols, qps.qvals, qps.nvar, qps.nvar)
    c = Vector{Float64}(qps.c)
    AL = Vector{Float64}(qps.lcon)
    AU = Vector{Float64}(qps.ucon)
    l = Vector{Float64}(qps.lvar)
    u = Vector{Float64}(qps.uvar)
    c0 = Float64(qps.c0)

    # QPSReader can return only one triangle of Q; mirror it to a symmetric matrix.
    diag_Q = diag(Q)
    Q = Q + transpose(Q) - Diagonal(diag_Q)

    return Q, A, c, AL, AU, l, u, c0
end

"""
    build_from_mps_qp(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)

Build a QP data tuple from an MPS/QPS file.

The returned tuple matches the `GPUPresolver.QPProblem` constructor order:
`(Q, A, c, AL, AU, l, u, obj_constant)`.
"""
function build_from_mps_qp(filename::AbstractString, verbose::Bool=true; mpsformat::Symbol=:auto)
    t_start = time()
    if verbose
        println("READING QP FILE ... ", filename)
    end
    Q, A, c, AL, AU, l, u, c0 = _read_qp_mps(filename; mpsformat=mpsformat)
    read_time = time() - t_start
    if verbose
        println(@sprintf("READING QP FILE time: %.2f seconds", read_time))
    end

    return (
        SparseMatrixCSC{Float64,Int32}(Q),
        SparseMatrixCSC{Float64,Int32}(A),
        Vector{Float64}(c),
        Vector{Float64}(AL),
        Vector{Float64}(AU),
        Vector{Float64}(l),
        Vector{Float64}(u),
        Float64(c0),
    )
end

"""
    build_from_Abc(A, c, AL, AU, l, u, obj_constant=0.0)

Build an LP model from matrix form.

# Arguments
- `A::Union{SparseMatrixCSC, Matrix}`: Constraint matrix (m × n). Dense matrices will be automatically converted to sparse format with a warning.
- `c::Vector{Float64}`: Objective coefficients (length n)
- `AL::Vector{Float64}`: Lower bounds for constraints Ax (length m)
- `AU::Vector{Float64}`: Upper bounds for constraints Ax (length m)
- `l::Vector{Float64}`: Lower bounds for variables x (length n)
- `u::Vector{Float64}`: Upper bounds for variables x (length n)
- `obj_constant::Float64`: Constant term in objective function (default: 0.0)

# Returns
- `LP_info_cpu`: LP model ready to be solved

# Example
```julia
using SparseArrays, GPUPresolver

A = sparse([1.0 2.0; 3.0 1.0])
c = [-3.0, -5.0]
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]

model = build_from_Abc(A, c, AL, AU, l, u)
result = run_presolve(model; config=PresolveConfig(backend="GPU"))
```

See also: [`run_presolve`](@ref)
"""
function build_from_Abc(A::Union{SparseMatrixCSC, Matrix},
    c::Vector{Float64},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64=0.0)

    # Convert dense matrix to sparse if needed
    if A isa Matrix
        @warn "Dense matrix detected. Converting to sparse format. For better performance, please provide a SparseMatrixCSC."
        A_sparse = sparse(A)
    else
        A_sparse = A
    end

    # Create copies to avoid modifying the input
    A_copy = copy(A_sparse)
    c_copy = copy(c)
    AL_copy = copy(AL)
    AU_copy = copy(AU)
    l_copy = copy(l)
    u_copy = copy(u)

    # Build the LP model
    standard_lp = formulation(A_copy, c_copy, AL_copy, AU_copy, l_copy, u_copy, obj_constant)

    return standard_lp
end

function _dataset_tolerance_suffixes(stoptol::Real)
    suffixes = Int[]
    if stoptol <= 1.0e-4
        push!(suffixes, 4)
    end
    if stoptol <= 1.0e-6
        push!(suffixes, 6)
    end
    if stoptol <= 1.0e-8
        push!(suffixes, 8)
    end
    return suffixes
end

function _dataset_safe_sgm(list)
    data = Float64[x for x in list if x isa Number]
    isempty(data) && return 0.0
    return exp(mean(log.(data .+ 10.0))) - 10.0
end

function _append_result_summary_rows!(table::DataFrame, params::GPUPresolverParameters)
    numeric = table[table.name .!= "SGM10", :]
    numeric = numeric[numeric.name .!= "solved", :]
    cols = Symbol.(names(table))

    summary = Dict{Symbol,Any}(col => "" for col in cols)
    summary[:name] = "SGM10"
    if :iter in cols
        summary[:iter] = _dataset_safe_sgm(numeric.iter)
    end
    if :alg_time in cols
        summary[:alg_time] = _dataset_safe_sgm(numeric.alg_time)
    end
    if :presolve_time in cols
        summary[:presolve_time] = _dataset_safe_sgm(numeric.presolve_time)
    end
    if :postsolve_time in cols
        summary[:postsolve_time] = _dataset_safe_sgm(numeric.postsolve_time)
    end
    if :presolve_nnz0 in cols
        summary[:presolve_nnz0] = _dataset_safe_sgm(numeric.presolve_nnz0)
    end
    if :presolve_nnz1 in cols
        summary[:presolve_nnz1] = _dataset_safe_sgm(numeric.presolve_nnz1)
    end
    for suffix in _dataset_tolerance_suffixes(params.stoptol)
        iter_col = Symbol("iter_$(suffix)")
        time_col = Symbol("time_$(suffix)")
        if iter_col in cols
            summary[iter_col] = _dataset_safe_sgm(numeric[!, iter_col])
        end
        if time_col in cols
            summary[time_col] = _dataset_safe_sgm(numeric[!, time_col])
        end
    end
    push!(table, summary)

    solved = Dict{Symbol,Any}(col => "" for col in cols)
    solved[:name] = "solved"
    solved[:alg_time] = count(x -> x isa Number && x < params.time_limit, numeric.alg_time)
    if :original_kkt_ok in cols
        solved[:original_kkt_ok] = count(isequal(true), numeric[!, :original_kkt_ok])
    end
    for suffix in _dataset_tolerance_suffixes(params.stoptol)
        time_col = Symbol("time_$(suffix)")
        if time_col in cols
            solved[time_col] = count(x -> x isa Number && x < params.time_limit, numeric[!, time_col])
        end
    end
    push!(table, solved)
    return table
end

# ------------------------------------------------------------------------------
# Presolve-only script utilities
# ------------------------------------------------------------------------------

function get_script_string_env(name::String, default::AbstractString)
    value = get(ENV, name, "")
    stripped = strip(value)
    return isempty(stripped) ? String(default) : String(stripped)
end

function get_script_int_env(name::String, default::Int)
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return parse(Int, value)
end

function get_script_float_env(name::String, default::Float64)
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return parse(Float64, value)
end

function get_script_bool_env(name::String, default::Bool)
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return default
    if value in ("1", "true", "yes", "on")
        return true
    elseif value in ("0", "false", "no", "off")
        return false
    end
    error("Invalid boolean env $name=$value. Expected one of 1/0, true/false, yes/no, on/off.")
end

function load_presolve_file_config(path::AbstractString)
    trimmed = strip(String(path))
    isempty(trimmed) && return Dict{String,Any}()
    isfile(trimmed) || error("PRESOLVE_CONFIG_FILE does not exist: $trimmed")
    parsed = TOML.parsefile(trimmed)
    return parsed isa Dict ? parsed : Dict{String,Any}()
end

function _config_string_setting(config::AbstractDict, key_path, default::AbstractString)
    config_value = _config_get(config, key_path, default)
    return String(config_value)
end

function _config_int_setting(config::AbstractDict, key_path, default::Int)
    config_value = _config_get(config, key_path, default)
    config_value isa Integer && return Int(config_value)
    config_value isa AbstractString && return parse(Int, String(config_value))
    error("Invalid integer value for $(join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], ".")): $(repr(config_value))")
end

function _config_float_setting(config::AbstractDict, key_path, default::Float64)
    config_value = _config_get(config, key_path, default)
    config_value isa Real && return Float64(config_value)
    config_value isa AbstractString && return parse(Float64, String(config_value))
    error("Invalid float value for $(join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], ".")): $(repr(config_value))")
end

function _config_bool_setting(config::AbstractDict, key_path, default::Bool)
    config_value = _config_get(config, key_path, default)
    return _as_bool(config_value, join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], "."))
end

function _config_get(config::AbstractDict, key_path::AbstractVector{<:AbstractString}, default)
    current = config
    for (idx, key) in enumerate(key_path)
        current isa AbstractDict || return default
        haskey(current, key) || return default
        value = current[key]
        idx == length(key_path) && return value
        current = value
    end
    return default
end

_config_get(config::AbstractDict, key_path::Tuple, default) =
    _config_get(config, String[string(k) for k in key_path], default)

_config_get(config::AbstractDict, key_path::AbstractString, default) =
    _config_get(config, [String(key_path)], default)

function _as_bool(value, name::AbstractString)
    value isa Bool && return value
    if value isa Integer
        value == 1 && return true
        value == 0 && return false
    elseif value isa AbstractString
        lowered = lowercase(strip(String(value)))
        lowered in ("1", "true", "yes", "on") && return true
        lowered in ("0", "false", "no", "off") && return false
    end
    error("Invalid boolean value for $name: $(repr(value))")
end

function get_script_string_setting(config::AbstractDict, env_name::String, key_path, default::AbstractString)
    env_value = get(ENV, env_name, "")
    stripped = strip(env_value)
    !isempty(stripped) && return String(stripped)
    config_value = _config_get(config, key_path, default)
    return String(config_value)
end

function get_script_int_setting(config::AbstractDict, env_name::String, key_path, default::Int)
    env_value = get(ENV, env_name, "")
    !isempty(strip(env_value)) && return parse(Int, env_value)
    config_value = _config_get(config, key_path, default)
    config_value isa Integer && return Int(config_value)
    config_value isa AbstractString && return parse(Int, String(config_value))
    error("Invalid integer value for $(join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], ".")): $(repr(config_value))")
end

function get_script_float_setting(config::AbstractDict, env_name::String, key_path, default::Float64)
    env_value = get(ENV, env_name, "")
    !isempty(strip(env_value)) && return parse(Float64, env_value)
    config_value = _config_get(config, key_path, default)
    config_value isa Real && return Float64(config_value)
    config_value isa AbstractString && return parse(Float64, String(config_value))
    error("Invalid float value for $(join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], ".")): $(repr(config_value))")
end

function get_script_bool_setting(config::AbstractDict, env_name::String, key_path, default::Bool)
    env_value = get(ENV, env_name, "")
    !isempty(strip(env_value)) && return get_script_bool_env(env_name, default)
    config_value = _config_get(config, key_path, default)
    return _as_bool(config_value, join(String[string(k) for k in (key_path isa Tuple ? key_path : key_path isa AbstractVector ? key_path : [key_path])], "."))
end

function build_presolve_setup(config_dict::AbstractDict)
    problem_type = normalize_problem_type(_config_string_setting(config_dict, ("problem", "type"), "LP"))
    device_number = _config_int_setting(config_dict, ("runtime", "device_number"), 0)
    verbose = _config_bool_setting(config_dict, ("runtime", "verbose"), true)
    scheduler_mode = _config_string_setting(config_dict, ("runtime", "scheduler_mode"), "fixed")
    presolve_backend = presolve_backend_for_mode(
        _config_string_setting(config_dict, ("runtime", "backend"), "GPU"),
        scheduler_mode,
    )

    switches = default_rule_switches(
        enable_close_bounds=_config_bool_setting(config_dict, ("rules", "close_bounds"), true),
        enable_empty_rows=_config_bool_setting(config_dict, ("rules", "empty_rows"), true),
        enable_singleton_rows=_config_bool_setting(config_dict, ("rules", "singleton_rows"), true),
        enable_activity_checks=_config_bool_setting(config_dict, ("rules", "activity_checks"), true),
        enable_primal_propagation=_config_bool_setting(config_dict, ("rules", "primal_propagation"), true),
        enable_parallel_rows=_config_bool_setting(config_dict, ("rules", "parallel_rows"), true),
        enable_empty_cols=_config_bool_setting(config_dict, ("rules", "empty_cols"), true),
        enable_singleton_cols_eq=_config_bool_setting(config_dict, ("rules", "singleton_cols_eq"), true),
        enable_singleton_cols_dual_infer=_config_bool_setting(config_dict, ("rules", "singleton_cols_dual_infer"), true),
        enable_doubleton_eq=_config_bool_setting(config_dict, ("rules", "doubleton_eq"), true),
        enable_linear_eq_agg=_config_bool_setting(config_dict, ("rules", "linear_eq_agg"), false),
        enable_dual_fix=_config_bool_setting(config_dict, ("rules", "dual_fix"), true),
        enable_parallel_cols=_config_bool_setting(config_dict, ("rules", "parallel_cols"), true),
        enable_redundant_bounds=_config_bool_setting(config_dict, ("rules", "redundant_bounds"), true),
        enable_fme_projection=_config_bool_setting(config_dict, ("rules", "fme_projection"), false),
        enable_structural_l1_substitution=_config_bool_setting(config_dict, ("rules", "structural_l1_substitution"), false),
        qp_linear_eq_agg_max_support=_config_int_setting(config_dict, ("qp", "linear_eq_agg_max_support"), 8),
        qp_doubleton_max_q_fill_abs=_config_int_setting(config_dict, ("qp", "doubleton_max_q_fill_abs"), 64),
        qp_doubleton_max_q_fill_ratio=_config_float_setting(config_dict, ("qp", "doubleton_max_q_fill_ratio"), 2.0),
        qp_singleton_max_support=_config_int_setting(config_dict, ("qp", "singleton_max_support"), -1),
        qp_singleton_max_q_fill_abs=_config_int_setting(config_dict, ("qp", "singleton_max_q_fill_abs"), -1),
        qp_singleton_max_q_fill_ratio=_config_float_setting(config_dict, ("qp", "singleton_max_q_fill_ratio"), Inf),
        qp_singleton_cols_eq_require_qdiag_zero=_config_bool_setting(config_dict, ("qp", "singleton_cols_eq_require_qdiag_zero"), false),
        qp_doubleton_eq_require_qdiag_zero=_config_bool_setting(config_dict, ("qp", "doubleton_eq_require_qdiag_zero"), false),
        fme_zero_objective_only=_config_bool_setting(config_dict, ("fme", "zero_objective_only"), true),
        fme_pair_limit=_config_int_setting(config_dict, ("fme", "pair_limit"), 8),
        fme_nnz_ratio_limit=_config_float_setting(config_dict, ("fme", "nnz_ratio_limit"), 1.0),
        fme_nnz_abs_slack=_config_int_setting(config_dict, ("fme", "nnz_abs_slack"), 0),
        fme_max_elims_per_call=_config_int_setting(config_dict, ("fme", "max_elims_per_call"), 0),
        fme_allow_main_flow_without_tape=_config_bool_setting(config_dict, ("fme", "allow_main_flow_without_tape"), false),
        fme_include_variable_bounds=_config_bool_setting(config_dict, ("fme", "include_variable_bounds"), false),
        fme_use_simple_screen=_config_bool_setting(config_dict, ("fme", "use_simple_screen"), false),
        fme_simple_side_limit=_config_int_setting(config_dict, ("fme", "simple_side_limit"), 2),
        verbose_fme=_config_bool_setting(config_dict, ("fme", "verbose"), false),
        structural_l1_pattern=Symbol(_config_string_setting(config_dict, ("structural_l1", "pattern"), "auto")),
        structural_l1_allow_main_flow_without_tape=_config_bool_setting(config_dict, ("structural_l1", "allow_main_flow_without_tape"), true),
        structural_l1_gpu_only=_config_bool_setting(config_dict, ("structural_l1", "gpu_only"), false),
        structural_l1_residual_bound_as_free_min=_config_float_setting(config_dict, ("structural_l1", "residual_bound_as_free_min"), Inf),
        doubleton_eq_max_fill_in_proxy=_config_int_setting(config_dict, ("doubleton", "max_fill_in_proxy"), 10),
        doubleton_eq_scan=_config_bool_setting(config_dict, ("doubleton", "scan"), false),
        doubleton_eq_min_selected_per_batch=_config_int_setting(config_dict, ("doubleton", "min_selected_per_batch"), 256),
        doubleton_eq_min_selected_ratio=_config_float_setting(config_dict, ("doubleton", "min_selected_ratio"), 0.005),
        doubleton_eq_max_batch_rounds=_config_int_setting(config_dict, ("doubleton", "max_batch_rounds"), 0),
        doubleton_eq_max_time=_config_float_setting(config_dict, ("doubleton", "max_time"), 0.0),
        presolve_scheduler_mode=scheduler_mode,
        enable_tiered_bootstrap=_config_bool_setting(
            config_dict,
            ("runtime", "tiered_bootstrap"),
            _config_bool_setting(config_dict, ("scheduling", "tiered_bootstrap"), true),
        ),
    )

    presolve_params = build_custom_presolve_params(
        switches;
        max_iters=_config_int_setting(config_dict, ("limits", "max_presolve_iters"), 10),
        max_time=_config_float_setting(config_dict, ("limits", "max_presolve_time"), 1000.0),
        feasibility_tol=_config_float_setting(config_dict, ("tolerances", "feasibility"), 1e-6),
        bound_tol=_config_float_setting(config_dict, ("tolerances", "bound"), 1e-6),
        zero_tol=_config_float_setting(config_dict, ("tolerances", "zero"), 1e-10),
    )

    config = build_presolve_config(
        device_number=device_number,
        verbose=verbose,
        presolve_backend=presolve_backend,
        presolve_params=presolve_params,
    )

    return (
        problem_type=problem_type,
        config=config,
        presolve_params=presolve_params,
        rule_switches=switches,
    )
end

function load_presolve_setup(path::AbstractString)
    config_dict = load_presolve_file_config(path)
    return build_presolve_setup(config_dict)
end

function default_config_path()
    return normpath(joinpath(@__DIR__, "..", "..", "config", "default.toml"))
end

function load_default_presolve_setup()
    return load_presolve_setup(default_config_path())
end

function normalize_problem_type(problem_type)
    t = uppercase(strip(String(problem_type)))
    t in ("LP", "QP") || error("Unsupported PROBLEM_TYPE=$(repr(problem_type)). Expected \"LP\" or \"QP\".")
    return t
end

function normalize_presolve_scheduler_mode(mode)
    mode === true && return :tiered
    mode === false && return :fixed

    mode_name = replace(lowercase(String(mode)), "-" => "_")
    if mode_name in ("tiered", "true")
        return :tiered
    elseif mode_name in ("fixed", "false")
        return :fixed
    end

    error("Unsupported PRESOLVE_SCHEDULER_MODE=$(repr(mode)). Expected :fixed or :tiered.")
end

function presolve_backend_for_mode(presolve_backend::AbstractString, mode)
    backend = uppercase(strip(String(presolve_backend)))
    if backend == "GPU"
        normalize_presolve_scheduler_mode(mode)
        return "GPU"
    elseif backend == "NONE"
        return "NONE"
    end

    error("Unsupported PRESOLVE_BACKEND=$(repr(presolve_backend)). Expected \"GPU\" or \"NONE\".")
end

function build_presolve_config(;
    device_number::Int=0,
    verbose::Bool=true,
    presolve_backend::String="GPU",
    presolve_params=nothing,
)
    return PresolveConfig(
        backend=presolve_backend,
        verbose=verbose,
        device_number=device_number,
        presolve_params=presolve_params,
    )
end

function default_rule_switches(;
    enable_close_bounds::Bool=true,
    enable_empty_rows::Bool=true,
    enable_singleton_rows::Bool=true,
    enable_activity_checks::Bool=true,
    enable_primal_propagation::Bool=true,
    enable_parallel_rows::Bool=true,
    enable_empty_cols::Bool=true,
    enable_singleton_cols_eq::Bool=true,
    enable_singleton_cols_dual_infer::Bool=true,
    enable_doubleton_eq::Bool=true,
    enable_linear_eq_agg::Bool=false,
    enable_dual_fix::Bool=true,
    enable_parallel_cols::Bool=true,
    enable_redundant_bounds::Bool=true,
    enable_fme_projection::Bool=false,
    enable_structural_l1_substitution::Bool=false,
    qp_linear_eq_agg_max_support::Int=8,
    qp_doubleton_max_q_fill_abs::Int=64,
    qp_doubleton_max_q_fill_ratio::Float64=2.0,
    qp_singleton_max_support::Int=-1,
    qp_singleton_max_q_fill_abs::Int=-1,
    qp_singleton_max_q_fill_ratio::Float64=Inf,
    qp_singleton_cols_eq_require_qdiag_zero::Bool=false,
    qp_doubleton_eq_require_qdiag_zero::Bool=false,
    fme_zero_objective_only::Bool=true,
    fme_pair_limit::Int=8,
    fme_nnz_ratio_limit::Float64=1.0,
    fme_nnz_abs_slack::Int=0,
    fme_max_elims_per_call::Int=0,
    fme_allow_main_flow_without_tape::Bool=false,
    fme_include_variable_bounds::Bool=false,
    fme_use_simple_screen::Bool=false,
    fme_simple_side_limit::Int=2,
    verbose_fme::Bool=false,
    structural_l1_pattern::Symbol=:auto,
    structural_l1_allow_main_flow_without_tape::Bool=true,
    structural_l1_gpu_only::Bool=false,
    structural_l1_residual_bound_as_free_min::Float64=Inf,
    doubleton_eq_max_fill_in_proxy::Int=10,
    doubleton_eq_scan::Bool=false,
    doubleton_eq_min_selected_per_batch::Int=256,
    doubleton_eq_min_selected_ratio::Float64=0.005,
    doubleton_eq_max_batch_rounds::Int=0,
    doubleton_eq_max_time::Float64=0.0,
    presolve_scheduler_mode=:fixed,
    enable_tiered_bootstrap::Bool=true,
    tiered_cleanup_max_rounds::Int=2,
    tiered_light_continue_ratio::Float64=0.995,
    tiered_cycle_stop_ratio::Float64=0.999,
    tiered_max_light_streak::Int=3,
    tiered_global_period::Int=2,
)
    return (
        enable_close_bounds=enable_close_bounds,
        enable_empty_rows=enable_empty_rows,
        enable_singleton_rows=enable_singleton_rows,
        enable_activity_checks=enable_activity_checks,
        enable_primal_propagation=enable_primal_propagation,
        enable_parallel_rows=enable_parallel_rows,
        enable_empty_cols=enable_empty_cols,
        enable_singleton_cols_eq=enable_singleton_cols_eq,
        enable_singleton_cols_dual_infer=enable_singleton_cols_dual_infer,
        enable_doubleton_eq=enable_doubleton_eq,
        enable_linear_eq_agg=enable_linear_eq_agg,
        enable_dual_fix=enable_dual_fix,
        enable_parallel_cols=enable_parallel_cols,
        enable_redundant_bounds=enable_redundant_bounds,
        enable_fme_projection=enable_fme_projection,
        enable_structural_l1_substitution=enable_structural_l1_substitution,
        qp_linear_eq_agg_max_support=qp_linear_eq_agg_max_support,
        qp_doubleton_max_q_fill_abs=qp_doubleton_max_q_fill_abs,
        qp_doubleton_max_q_fill_ratio=qp_doubleton_max_q_fill_ratio,
        qp_singleton_max_support=qp_singleton_max_support,
        qp_singleton_max_q_fill_abs=qp_singleton_max_q_fill_abs,
        qp_singleton_max_q_fill_ratio=qp_singleton_max_q_fill_ratio,
        qp_singleton_cols_eq_require_qdiag_zero=qp_singleton_cols_eq_require_qdiag_zero,
        qp_doubleton_eq_require_qdiag_zero=qp_doubleton_eq_require_qdiag_zero,
        fme_zero_objective_only=fme_zero_objective_only,
        fme_pair_limit=fme_pair_limit,
        fme_nnz_ratio_limit=fme_nnz_ratio_limit,
        fme_nnz_abs_slack=fme_nnz_abs_slack,
        fme_max_elims_per_call=fme_max_elims_per_call,
        fme_allow_main_flow_without_tape=fme_allow_main_flow_without_tape,
        fme_include_variable_bounds=fme_include_variable_bounds,
        fme_use_simple_screen=fme_use_simple_screen,
        fme_simple_side_limit=fme_simple_side_limit,
        verbose_fme=verbose_fme,
        structural_l1_pattern=structural_l1_pattern,
        structural_l1_allow_main_flow_without_tape=structural_l1_allow_main_flow_without_tape,
        structural_l1_gpu_only=structural_l1_gpu_only,
        structural_l1_residual_bound_as_free_min=structural_l1_residual_bound_as_free_min,
        doubleton_eq_max_fill_in_proxy=doubleton_eq_max_fill_in_proxy,
        doubleton_eq_scan=doubleton_eq_scan,
        doubleton_eq_min_selected_per_batch=doubleton_eq_min_selected_per_batch,
        doubleton_eq_min_selected_ratio=doubleton_eq_min_selected_ratio,
        doubleton_eq_max_batch_rounds=doubleton_eq_max_batch_rounds,
        doubleton_eq_max_time=doubleton_eq_max_time,
        presolve_scheduler_mode=presolve_scheduler_mode,
        enable_tiered_bootstrap=enable_tiered_bootstrap,
        tiered_cleanup_max_rounds=tiered_cleanup_max_rounds,
        tiered_light_continue_ratio=tiered_light_continue_ratio,
        tiered_cycle_stop_ratio=tiered_cycle_stop_ratio,
        tiered_max_light_streak=tiered_max_light_streak,
        tiered_global_period=tiered_global_period,
    )
end

function build_custom_presolve_params(
    switches::NamedTuple;
    max_iters::Int=10,
    max_time::Float64=1000.0,
    feasibility_tol::Float64=1e-6,
    bound_tol::Float64=1e-6,
    zero_tol::Float64=1e-10,
)
    p = GPUBackend.PresolveParams()
    p.max_iters = max_iters
    p.max_time = max_time

    p.enable_close_bounds = get(switches, :enable_close_bounds, true)
    p.enable_empty_rows = get(switches, :enable_empty_rows, true)
    p.enable_singleton_rows = get(switches, :enable_singleton_rows, true)
    p.enable_activity_checks = get(switches, :enable_activity_checks, true)
    p.enable_primal_propagation = get(switches, :enable_primal_propagation, true)
    p.enable_parallel_rows = get(switches, :enable_parallel_rows, true)
    p.enable_empty_cols = get(switches, :enable_empty_cols, true)
    p.enable_singleton_cols_eq = get(switches, :enable_singleton_cols_eq, true)
    p.enable_singleton_cols_dual_infer = get(switches, :enable_singleton_cols_dual_infer, true)
    p.enable_doubleton_eq = get(switches, :enable_doubleton_eq, true)
    p.enable_linear_eq_agg = get(switches, :enable_linear_eq_agg, false)
    p.enable_dual_fix = get(switches, :enable_dual_fix, true)
    p.enable_parallel_cols = get(switches, :enable_parallel_cols, true)
    p.enable_redundant_bounds = get(switches, :enable_redundant_bounds, true)
    p.enable_fme_projection = get(switches, :enable_fme_projection, false)
    p.enable_structural_l1_substitution = get(switches, :enable_structural_l1_substitution, false)
    p.qp_linear_eq_agg_max_support = get(switches, :qp_linear_eq_agg_max_support, 8)
    p.qp_doubleton_max_q_fill_abs = get(switches, :qp_doubleton_max_q_fill_abs, 64)
    p.qp_doubleton_max_q_fill_ratio = get(switches, :qp_doubleton_max_q_fill_ratio, 2.0)
    p.qp_singleton_max_support = get(switches, :qp_singleton_max_support, -1)
    p.qp_singleton_max_q_fill_abs = get(switches, :qp_singleton_max_q_fill_abs, -1)
    p.qp_singleton_max_q_fill_ratio = get(switches, :qp_singleton_max_q_fill_ratio, Inf)
    p.qp_singleton_cols_eq_require_qdiag_zero = get(switches, :qp_singleton_cols_eq_require_qdiag_zero, false)
    p.qp_doubleton_eq_require_qdiag_zero = get(switches, :qp_doubleton_eq_require_qdiag_zero, false)
    p.fme_zero_objective_only = get(switches, :fme_zero_objective_only, true)
    p.fme_pair_limit = get(switches, :fme_pair_limit, 8)
    p.fme_nnz_ratio_limit = get(switches, :fme_nnz_ratio_limit, 1.0)
    p.fme_nnz_abs_slack = get(switches, :fme_nnz_abs_slack, 0)
    p.fme_max_elims_per_call = get(switches, :fme_max_elims_per_call, 0)
    p.fme_allow_main_flow_without_tape = get(switches, :fme_allow_main_flow_without_tape, false)
    p.fme_include_variable_bounds = get(switches, :fme_include_variable_bounds, false)
    p.fme_use_simple_screen = get(switches, :fme_use_simple_screen, false)
    p.fme_simple_side_limit = get(switches, :fme_simple_side_limit, 2)
    p.verbose_fme = get(switches, :verbose_fme, false)
    p.structural_l1_pattern = get(switches, :structural_l1_pattern, :auto)
    p.structural_l1_allow_main_flow_without_tape = get(switches, :structural_l1_allow_main_flow_without_tape, true)
    p.structural_l1_gpu_only = get(switches, :structural_l1_gpu_only, false)
    p.structural_l1_residual_bound_as_free_min = get(switches, :structural_l1_residual_bound_as_free_min, Inf)
    p.doubleton_eq_max_fill_in_proxy = max(0, get(switches, :doubleton_eq_max_fill_in_proxy, 10))
    p.doubleton_eq_scan = get(switches, :doubleton_eq_scan, false)
    p.doubleton_eq_min_selected_per_batch = max(0, get(switches, :doubleton_eq_min_selected_per_batch, 256))
    p.doubleton_eq_min_selected_ratio = max(0.0, get(switches, :doubleton_eq_min_selected_ratio, 0.005))
    p.doubleton_eq_max_batch_rounds = max(0, get(switches, :doubleton_eq_max_batch_rounds, 0))
    p.doubleton_eq_max_time = max(0.0, get(switches, :doubleton_eq_max_time, 0.0))

    scheduler_mode = get(switches, :presolve_scheduler_mode, :fixed)
    p.gpu_presolve_scheduler = normalize_presolve_scheduler_mode(scheduler_mode)
    p.enable_tiered_bootstrap = get(switches, :enable_tiered_bootstrap, true)
    p.tiered_cleanup_max_rounds = get(switches, :tiered_cleanup_max_rounds, 2)
    p.tiered_light_continue_ratio = get(switches, :tiered_light_continue_ratio, 0.995)
    p.tiered_cycle_stop_ratio = get(switches, :tiered_cycle_stop_ratio, 0.999)
    p.tiered_max_light_streak = get(switches, :tiered_max_light_streak, 3)
    p.tiered_global_period = get(switches, :tiered_global_period, 2)

    row_order = [:empty_rows, :singleton_rows, :activity_checks, :primal_propagation, :parallel_rows]
    col_order = [:close_bounds, :structural_l1_substitution, :empty_cols, :singleton_cols_dual_infer, :singleton_cols_eq, :doubleton_eq, :linear_eq_agg, :dual_fix, :parallel_cols, :fme_projection]

    p.row_rule_order = Symbol[
        r for r in row_order if (
            (r == :empty_rows && p.enable_empty_rows) ||
            (r == :singleton_rows && p.enable_singleton_rows) ||
            (r == :activity_checks && p.enable_activity_checks) ||
            (r == :primal_propagation && p.enable_primal_propagation) ||
            (r == :parallel_rows && p.enable_parallel_rows)
        )
    ]
    p.col_rule_order = Symbol[
        r for r in col_order if (
            (r == :close_bounds && p.enable_close_bounds) ||
            (r == :structural_l1_substitution && p.enable_structural_l1_substitution) ||
            (r == :empty_cols && p.enable_empty_cols) ||
            (r == :singleton_cols_dual_infer && p.enable_singleton_cols_dual_infer) ||
            (r == :singleton_cols_eq && p.enable_singleton_cols_eq) ||
            (r == :doubleton_eq && p.enable_doubleton_eq) ||
            (r == :linear_eq_agg && p.enable_linear_eq_agg) ||
            (r == :dual_fix && p.enable_dual_fix) ||
            (r == :parallel_cols && p.enable_parallel_cols) ||
            (r == :fme_projection && p.enable_fme_projection)
        )
    ]

    p.feasibility_tol = feasibility_tol
    p.bound_tol = bound_tol
    p.zero_tol = zero_tol
    return p
end

function _build_problem_from_file(file_name::AbstractString, problem_type::String, verbose::Bool)
    if problem_type == "QP"
        Q, A, c, AL, AU, l, u, c0 = build_from_mps_qp(file_name, verbose)
        problem = QPProblem(Q, A, c, AL, AU, l, u, c0)
        return problem, problem.A
    end

    model = build_from_mps(file_name, verbose)
    problem = LPProblem(model)
    return problem, model.A
end

const _GPU_PRESOLVE_WARMED = Ref(false)

function _gpu_presolve_warmup_mode()
    mode = lowercase(strip(get(ENV, "GPUPRESOLVER_WARMUP_MODE", "actual")))
    if mode in ("off", "none", "false", "0")
        return :off
    elseif mode in ("tiny", "small")
        return :tiny
    elseif mode in ("actual", "model")
        return :actual
    elseif mode in ("representative", "rep", "default", "true", "1")
        return :representative
    end
    return :actual
end

function _gpu_presolve_env_int(name::AbstractString, default::Int; min_value::Int, max_value::Int)
    raw = get(ENV, name, string(default))
    parsed = tryparse(Int, raw)
    value = isnothing(parsed) ? default : parsed
    return clamp(value, min_value, max_value)
end

function _gpu_presolve_env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    return default
end

function _tiny_gpu_presolve_warmup_model()
    A = sparse(
        Int32[2, 3, 3, 4, 4, 5, 5, 6],
        Int32[1, 2, 3, 3, 4, 4, 5, 8],
        Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 2.0],
        6,
        8,
    )
    c = Float64[0.0, 1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    AL = Float64[-Inf, 1.0, 2.0, 0.0, -Inf, -1.0]
    AU = Float64[Inf, 1.0, 2.0, 3.0, 4.0, 2.0]
    l = Float64[0.0, 0.0, 0.0, 0.0, -1.0e-10, 0.0, 0.0, 0.0]
    u = Float64[10.0, 10.0, 10.0, 10.0, 1.0e-10, Inf, Inf, 5.0]
    return build_from_Abc(A, c, AL, AU, l, u, 0.0)
end

function _block_gpu_presolve_warmup_model(blocks::Int)
    rows_per_block = 4
    cols_per_block = 4
    m = rows_per_block * blocks
    n = cols_per_block * blocks
    nnz_per_block = 7
    nnz_total = nnz_per_block * blocks

    row_idx = Vector{Int32}(undef, nnz_total)
    col_idx = Vector{Int32}(undef, nnz_total)
    values = Vector{Float64}(undef, nnz_total)
    AL = Vector{Float64}(undef, m)
    AU = Vector{Float64}(undef, m)
    c = Vector{Float64}(undef, n)
    l = fill(0.0, n)
    u = fill(10.0, n)

    p = 1
    for block in 0:(blocks - 1)
        r0 = rows_per_block * block
        c0 = cols_per_block * block

        c[c0 + 1] = 0.0
        c[c0 + 2] = 1.0
        c[c0 + 3] = -1.0
        c[c0 + 4] = 0.25

        AL[r0 + 1] = 0.0
        AU[r0 + 1] = 0.0
        row_idx[p] = Int32(r0 + 1)
        col_idx[p] = Int32(c0 + 1)
        values[p] = 1.0
        p += 1

        AL[r0 + 2] = -Inf
        AU[r0 + 2] = 30.0
        row_idx[p] = Int32(r0 + 2)
        col_idx[p] = Int32(c0 + 2)
        values[p] = 1.0
        p += 1
        row_idx[p] = Int32(r0 + 2)
        col_idx[p] = Int32(c0 + 3)
        values[p] = 1.0
        p += 1

        AL[r0 + 3] = 0.0
        AU[r0 + 3] = 15.0
        row_idx[p] = Int32(r0 + 3)
        col_idx[p] = Int32(c0 + 2)
        values[p] = 1.0
        p += 1
        row_idx[p] = Int32(r0 + 3)
        col_idx[p] = Int32(c0 + 4)
        values[p] = 1.0
        p += 1

        AL[r0 + 4] = 0.0
        AU[r0 + 4] = 15.0
        row_idx[p] = Int32(r0 + 4)
        col_idx[p] = Int32(c0 + 2)
        values[p] = 1.0
        p += 1
        row_idx[p] = Int32(r0 + 4)
        col_idx[p] = Int32(c0 + 4)
        values[p] = 1.0
        p += 1
    end

    return build_from_Abc(sparse(row_idx, col_idx, values, m, n), c, AL, AU, l, u, 0.0)
end

function _rule_coverage_gpu_presolve_warmup_model()
    row_idx = Int32[]
    col_idx = Int32[]
    values = Float64[]
    AL = Float64[]
    AU = Float64[]
    c = Float64[]
    l = Float64[]
    u = Float64[]

    function add_col(; obj=0.0, lower=0.0, upper=10.0)
        push!(c, obj)
        push!(l, lower)
        push!(u, upper)
        return length(c)
    end

    function add_row(lower, upper, entries::Vector{Tuple{Int,Float64}})
        row = length(AL) + 1
        push!(AL, lower)
        push!(AU, upper)
        for (col, val) in entries
            push!(row_idx, Int32(row))
            push!(col_idx, Int32(col))
            push!(values, val)
        end
        return row
    end

    add_row(0.0, 0.0, Tuple{Int,Float64}[])

    sr = add_col(obj=0.0, lower=-10.0, upper=10.0)
    add_row(2.0, 2.0, [(sr, 1.0)])

    ar1 = add_col(obj=0.0, lower=0.0, upper=1.0)
    ar2 = add_col(obj=0.0, lower=0.0, upper=1.0)
    add_row(-Inf, 2.0, [(ar1, 1.0), (ar2, 1.0)])
    add_row(0.0, Inf, [(ar1, 1.0), (ar2, 1.0)])
    add_row(0.0, 2.0, [(ar1, 1.0), (ar2, 1.0)])

    pp1 = add_col(obj=0.0, lower=0.0, upper=10.0)
    pp2 = add_col(obj=0.0, lower=3.0, upper=10.0)
    add_row(-Inf, 5.0, [(pp1, 1.0), (pp2, 1.0)])

    pr1 = add_col(obj=0.0, lower=0.0, upper=10.0)
    pr2 = add_col(obj=0.0, lower=0.0, upper=10.0)
    add_row(-Inf, 8.0, [(pr1, 1.0), (pr2, 1.0)])
    add_row(-Inf, 16.0, [(pr1, 2.0), (pr2, 2.0)])

    pc1 = add_col(obj=1.0, lower=0.0, upper=10.0)
    pc2 = add_col(obj=1.0, lower=0.0, upper=10.0)
    add_row(-Inf, 10.0, [(pc1, 1.0), (pc2, 1.0)])
    add_row(-Inf, 20.0, [(pc1, 2.0), (pc2, 2.0)])

    eq_single = add_col(obj=1.0, lower=0.0, upper=10.0)
    eq_partner = add_col(obj=0.0, lower=0.0, upper=4.0)
    add_row(3.0, 3.0, [(eq_single, 1.0), (eq_partner, 1.0)])

    ineq_single = add_col(obj=1.0, lower=0.0, upper=10.0)
    ineq_partner = add_col(obj=0.0, lower=0.0, upper=2.0)
    add_row(-Inf, 3.0, [(ineq_single, 1.0), (ineq_partner, 1.0)])

    dual_fix_col = add_col(obj=1.0, lower=0.0, upper=10.0)
    dual_partner = add_col(obj=0.0, lower=0.0, upper=10.0)
    add_row(-Inf, 12.0, [(dual_fix_col, 1.0), (dual_partner, 1.0)])

    empty_col = add_col(obj=-1.0, lower=-5.0, upper=5.0)
    _ = empty_col

    long_entries = Tuple{Int,Float64}[]
    for _ in 1:80
        col = add_col(obj=0.0, lower=0.0, upper=100.0)
        push!(long_entries, (col, 1.0))
    end
    add_row(-Inf, 10.0, long_entries)

    return build_from_Abc(
        sparse(row_idx, col_idx, values, length(AL), length(c)),
        c,
        AL,
        AU,
        l,
        u,
        0.0,
    )
end

function _parallel_col_coverage_gpu_presolve_warmup_model()
    A = sparse(
        Int32[1, 1, 2, 2, 1, 2],
        Int32[1, 2, 1, 2, 3, 3],
        Float64[1.0, 1.0, 2.0, 2.0, -1.0, 1.0],
        2,
        3,
    )
    c = Float64[1.0, 1.0, 0.0]
    AL = Float64[0.0, 0.0]
    AU = Float64[10.0, 20.0]
    l = Float64[0.0, 0.0, 0.0]
    u = Float64[10.0, 10.0, 10.0]
    return build_from_Abc(A, c, AL, AU, l, u, 0.0)
end

function _representative_gpu_presolve_warmup_model()
    blocks = _gpu_presolve_env_int(
        "GPUPRESOLVER_REPRESENTATIVE_WARMUP_BLOCKS",
        262144;
        min_value=1,
        max_value=1048576,
    )
    return _block_gpu_presolve_warmup_model(blocks)
end

function _coverage_gpu_presolve_warmup_model()
    blocks = _gpu_presolve_env_int(
        "GPUPRESOLVER_COVERAGE_WARMUP_BLOCKS",
        16384;
        min_value=1,
        max_value=262144,
    )
    return _block_gpu_presolve_warmup_model(blocks)
end

function _coverage_gpu_presolve_warmup_models()
    return LP_info_cpu[
        _coverage_gpu_presolve_warmup_model(),
        _rule_coverage_gpu_presolve_warmup_model(),
        _parallel_col_coverage_gpu_presolve_warmup_model(),
    ]
end

function _gpu_presolve_warmup_models(mode::Symbol=_gpu_presolve_warmup_mode())
    mode == :off && return LP_info_cpu[]
    mode == :tiny && return LP_info_cpu[_tiny_gpu_presolve_warmup_model()]
    mode == :coverage && return _coverage_gpu_presolve_warmup_models()
    return LP_info_cpu[_representative_gpu_presolve_warmup_model()]
end

function _gpu_presolve_warmup_model(mode::Symbol=_gpu_presolve_warmup_mode())
    models = _gpu_presolve_warmup_models(mode)
    return isempty(models) ? nothing : first(models)
end

function warmup_gpu_presolve!(config; presolve_params=nothing)
    _GPU_PRESOLVE_WARMED[] && return nothing
    normalize_presolve_backend(config.backend) == "GPU" || return nothing
    mode = _gpu_presolve_warmup_mode()
    mode == :off && return nothing
    models = _gpu_presolve_warmup_models(mode == :actual ? :coverage : mode)

    warmup_config = isnothing(presolve_params) ? config : PresolveConfig(
        backend=String(config.backend),
        verbose=false,
        device_number=config.device_number,
        presolve_params=presolve_params,
    )

    state = nothing
    try
        for model in models
            result = run_presolve(model; config=warmup_config)
            state = result.state
            if !isnothing(state)
                free_presolve_state!(state)
                state = nothing
            end
        end
        _GPU_PRESOLVE_WARMED[] = true
    catch err
        if mode == :representative
            config.verbose && @warn "Representative GPU presolve warmup failed; trying tiny warmup." exception=(err, catch_backtrace())
            try
                result = run_presolve(_tiny_gpu_presolve_warmup_model(); config=warmup_config)
                state = result.state
                _GPU_PRESOLVE_WARMED[] = true
            catch tiny_err
                config.verbose && @warn "GPU presolve warmup failed; continuing without warmup." exception=(tiny_err, catch_backtrace())
            end
        else
            config.verbose && @warn "GPU presolve warmup failed; continuing without warmup." exception=(err, catch_backtrace())
        end
    finally
        if !isnothing(state)
            free_presolve_state!(state)
        end
    end
    return nothing
end

function run_presolve_only_record(
    file_name::AbstractString,
    problem_type::String,
    config;
    presolve_params=nothing,
    throw_errors::Bool=true,
)
    normalized_type = normalize_problem_type(problem_type)
    normalized_backend = normalize_presolve_backend(config.backend)

    presolve_state = nothing
    m0 = -1
    n0 = -1
    nnz0 = -1
    qnnz0 = missing
    m1 = -1
    n1 = -1
    nnz1 = -1
    qnnz1 = missing
    presolve_time = NaN
    error_message = ""

    try
        problem, A0 = _build_problem_from_file(file_name, normalized_type, config.verbose)
        m0, n0 = size(A0)
        nnz0 = nnz(A0)
        m1, n1, nnz1 = m0, n0, nnz0
        if normalized_type == "QP"
            qnnz0 = nnz(problem.Q)
            qnnz1 = qnnz0
        end

        t_start = time()
        run_config = if isnothing(presolve_params)
            config
        else
            PresolveConfig(
                backend=String(config.backend),
                verbose=config.verbose,
                device_number=config.device_number,
                presolve_params=presolve_params,
            )
        end

        presolve_result = run_presolve(problem; config=run_config)
        presolve_time = presolve_result.presolve_time > 0 ? presolve_result.presolve_time : (time() - t_start)
        presolve_state = presolve_result.state

        if presolve_result.status == "OK" && !isnothing(presolve_state) && !isnothing(presolve_result.reduced_problem)
            reduced_A = normalized_type == "QP" ? presolve_result.reduced_problem.A : presolve_result.reduced_problem.model.A
            m1, n1 = size(reduced_A)
            nnz1 = nnz(reduced_A)
            if normalized_type == "QP"
                qnnz1 = nnz(presolve_result.reduced_problem.Q)
            end
            return (
                status="OK",
                presolve_time=presolve_time,
                m0=m0,
                n0=n0,
                nnz0=nnz0,
                qnnz0=qnnz0,
                m1=m1,
                n1=n1,
                nnz1=nnz1,
                qnnz1=qnnz1,
                error_message=error_message,
            )
        end

        return (
            status="PRESOLVE_FAILED",
            presolve_time=presolve_time,
            m0=m0,
            n0=n0,
            nnz0=nnz0,
            qnnz0=qnnz0,
            m1=m1,
            n1=n1,
            nnz1=nnz1,
            qnnz1=qnnz1,
            error_message=error_message,
        )
    catch err
        if throw_errors
            rethrow(err)
        end
        error_message = sprint(showerror, err, catch_backtrace())
        return (
            status="ERROR",
            presolve_time=presolve_time,
            m0=m0,
            n0=n0,
            nnz0=nnz0,
            qnnz0=qnnz0,
            m1=m1,
            n1=n1,
            nnz1=nnz1,
            qnnz1=qnnz1,
            error_message=error_message,
        )
    finally
        if !isnothing(presolve_state)
            free_presolve_state!(presolve_state)
        end
    end
end

function print_single_presolve_summary(io::IO, file_name::AbstractString, result)
    println(io, "File: ", file_name)
    println(io, "Status: ", result.status)
    if result.m0 >= 0 && result.n0 >= 0 && result.nnz0 >= 0
        println(io, "dims: m=$(result.m0)->$(result.m1), n=$(result.n0)->$(result.n1)")
        println(io, "nnz: $(result.nnz0)->$(result.nnz1)")
        if !ismissing(result.qnnz0) && !ismissing(result.qnnz1)
            println(io, "qnnz: $(result.qnnz0)->$(result.qnnz1)")
        end
    end
end

function run_single_presolve_only(
    file_name::AbstractString,
    problem_type::String,
    config;
    presolve_params=nothing,
)
    result = run_presolve_only_record(
        file_name,
        problem_type,
        config;
        presolve_params=presolve_params,
        throw_errors=true,
    )
    print_single_presolve_summary(stdout, file_name, result)
    return result
end

function shifted_geometric_mean(values::AbstractVector, shift::Real)
    valid = Float64[]
    for v in values
        ismissing(v) && continue
        x = Float64(v)
        isfinite(x) || continue
        push!(valid, x)
    end
    isempty(valid) && return NaN

    log_sum = 0.0
    for v in valid
        log_sum += log(v + shift)
    end
    return exp(log_sum / length(valid)) - shift
end

function append_sgm10_row!(result_table::DataFrame)
    ok_rows = result_table[result_table.status .== "OK", :]
    isempty(ok_rows) && return result_table

    sgm10_time = shifted_geometric_mean(ok_rows.presolve_time, 10.0)
    push!(
        result_table,
        (
            name="SGM10",
            status="OK",
            presolve_time=sgm10_time,
            m0=missing,
            n0=missing,
            nnz0=missing,
            qnnz0=missing,
            m1=missing,
            n1=missing,
            nnz1=missing,
            qnnz1=missing,
        ),
    )
    return result_table
end

function run_dataset_presolve_only(
    data_path::AbstractString,
    result_path::AbstractString,
    problem_type::String,
    config;
    presolve_params=nothing,
    max_files::Int=0,
    result_file::String="GPUPresolver_presolve_only_result.csv",
    log_file::String="GPUPresolver_presolve_only_log.txt",
)
    normalized_type = normalize_problem_type(problem_type)
    files = sort(filter(file -> begin
        lower = lowercase(file)
        endswith(lower, ".mps") || endswith(lower, ".mps.gz")
    end, readdir(data_path)))

    if max_files > 0
        files = files[1:min(max_files, length(files))]
    end

    isdir(result_path) || mkpath(result_path)
    result_csv = joinpath(result_path, result_file)
    log_path = joinpath(result_path, log_file)
    result_table = DataFrame(
        name=String[],
        status=String[],
        presolve_time=Float64[],
        m0=Union{Missing,Int}[],
        n0=Union{Missing,Int}[],
        nnz0=Union{Missing,Int}[],
        qnnz0=Union{Missing,Int}[],
        m1=Union{Missing,Int}[],
        n1=Union{Missing,Int}[],
        nnz1=Union{Missing,Int}[],
        qnnz1=Union{Missing,Int}[],
    )

    function _run_with_output_capture(f::Function)
        capture_path = tempname()
        result = nothing
        captured = ""
        try
            open(capture_path, "w+") do capture_io
                result = redirect_stdout(capture_io) do
                    redirect_stderr(capture_io) do
                        f()
                    end
                end
                flush(capture_io)
            end
            captured = read(capture_path, String)
        finally
            rm(capture_path; force=true)
        end
        return result, captured
    end

    open(log_path, "w") do io
        if normalized_type == "LP" && normalize_presolve_backend(config.backend) == "GPU"
            warmup_gpu_presolve!(config; presolve_params=presolve_params)
        end

        for (i, file) in enumerate(files)
            file_name = joinpath(data_path, file)
            println(@sprintf("presolving problem %d/%d: %s", i, length(files), file))
            println(io, @sprintf("presolving problem %d/%d: %s", i, length(files), file))
            println(io, "-"^80)

            result, captured_output = _run_with_output_capture() do
                run_presolve_only_record(
                    file_name,
                    normalized_type,
                    config;
                    presolve_params=presolve_params,
                    throw_errors=false,
                )
            end

            if !isempty(captured_output)
                print(stdout, captured_output)
                print(io, captured_output)
            end

            if result.status == "OK"
                println(io, @sprintf("status: OK, presolve_time: %.6fs", result.presolve_time))
                println(io, "dims: m=$(result.m0)->$(result.m1), n=$(result.n0)->$(result.n1)")
                println(io, "nnz: $(result.nnz0)->$(result.nnz1)")
                if !ismissing(result.qnnz0) && !ismissing(result.qnnz1)
                    println(io, "qnnz: $(result.qnnz0)->$(result.qnnz1)")
                end
            elseif result.status == "PRESOLVE_FAILED"
                println(io, "status: PRESOLVE_FAILED")
            else
                println(io, "status: ERROR")
                println(io, result.error_message)
            end

            push!(
                result_table,
                (
                    name=file,
                    status=result.status,
                    presolve_time=isfinite(result.presolve_time) ? result.presolve_time : NaN,
                    m0=result.m0,
                    n0=result.n0,
                    nnz0=result.nnz0,
                    qnnz0=result.qnnz0,
                    m1=result.m1,
                    n1=result.n1,
                    nnz1=result.nnz1,
                    qnnz1=result.qnnz1,
                ),
            )
            println(io)
        end

        println(io, "Presolve-only dataset run completed: ", length(files), " problems")
        println(io, "Results written to: ", result_csv)
        println(io, "Log written to: ", log_path)
    end

    append_sgm10_row!(result_table)
    CSV.write(result_csv, result_table)
    println("Presolve-only dataset run completed: ", length(files), " problems")
    println("Results written to: ", result_csv)
    println("Log written to: ", log_path)
    return result_table
end
