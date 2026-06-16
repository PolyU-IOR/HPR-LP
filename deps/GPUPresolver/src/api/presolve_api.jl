"""
Problem-agnostic presolve API.

LP is fully wired to the GPU-native backend.
QP is routed through the QP-native GPU presolve backend, so QP reductions live
in the QP scheduler and QP rule files rather than API-layer replay code.
"""

abstract type AbstractPresolveProblem end
abstract type AbstractPresolveState end

struct LPProblem <: AbstractPresolveProblem
    model::LP_info_cpu
end

struct QPProblem <: AbstractPresolveProblem
    Q::SparseMatrixCSC{Float64,Int32}
    A::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    obj_constant::Float64
end

Base.@kwdef struct PresolveConfig
    backend::String = "GPU"
    verbose::Bool = true
    device_number::Int = 0
    presolve_params::Any = nothing
end

struct GPUBackendStateHandle <: AbstractPresolveState
    inner::GPUBackend.PresolveState
end

struct PresolveResult{P<:AbstractPresolveProblem,S}
    reduced_problem::P
    state::S
    backend::String
    status::String
    presolve_time::Float64
end

abstract type PresolveProblemKind end

struct LPPresolveKind <: PresolveProblemKind end
struct QPPresolveKind <: PresolveProblemKind end

struct RuleSpec
    public_name::Symbol
    impl_name::Symbol
    execution_stage::Symbol
    supported::Bool
end

const RULE_STAGE_GPU_LOOP = :gpu_loop
const RULE_STAGE_UNSUPPORTED = :unsupported

_rule_spec(::LPPresolveKind, rule::Symbol) = RuleSpec(rule, Symbol(:lp_, rule), RULE_STAGE_GPU_LOOP, true)
_rule_spec(kind::PresolveProblemKind, rule::Symbol) = _rule_spec(kind, Val(rule))

_qp_gpu_rule(rule::Symbol) = RuleSpec(rule, Symbol(:qp_, rule, :_gpu), RULE_STAGE_GPU_LOOP, true)
_qp_unsupported_rule(rule::Symbol) = RuleSpec(rule, :unsupported, RULE_STAGE_UNSUPPORTED, false)

_rule_spec(::QPPresolveKind, ::Val{:empty_rows}) = _qp_gpu_rule(:empty_rows)
_rule_spec(::QPPresolveKind, ::Val{:singleton_rows}) = _qp_gpu_rule(:singleton_rows)
_rule_spec(::QPPresolveKind, ::Val{:activity_checks}) = _qp_gpu_rule(:activity_checks)
_rule_spec(::QPPresolveKind, ::Val{:primal_propagation}) = _qp_gpu_rule(:primal_propagation)
_rule_spec(::QPPresolveKind, ::Val{:parallel_rows}) = _qp_gpu_rule(:parallel_rows)
_rule_spec(::QPPresolveKind, ::Val{:close_bounds}) = _qp_gpu_rule(:close_bounds)
_rule_spec(::QPPresolveKind, ::Val{:singleton_cols_eq}) = _qp_gpu_rule(:singleton_cols_eq)
_rule_spec(::QPPresolveKind, ::Val{:doubleton_eq}) = _qp_gpu_rule(:doubleton_eq)
_rule_spec(::QPPresolveKind, ::Val{:redundant_bounds}) = _qp_unsupported_rule(:redundant_bounds)
_rule_spec(::QPPresolveKind, ::Val{:empty_cols}) = _qp_gpu_rule(:empty_cols)
_rule_spec(::QPPresolveKind, ::Val{:singleton_cols_dual_infer}) =
    _qp_gpu_rule(:singleton_cols_dual_infer)
_rule_spec(::QPPresolveKind, ::Val{:parallel_cols}) = _qp_gpu_rule(:parallel_cols)
_rule_spec(::QPPresolveKind, ::Val{:dual_fix}) = _qp_gpu_rule(:dual_fix)
_rule_spec(::QPPresolveKind, ::Val{:linear_eq_agg}) = _qp_gpu_rule(:linear_eq_agg)
_rule_spec(::QPPresolveKind, ::Val{rule}) where {rule} = _qp_unsupported_rule(rule)

function _resolve_rule_specs(kind::PresolveProblemKind, rules::Vector{Symbol})
    return RuleSpec[_rule_spec(kind, rule) for rule in rules]
end

_rule_public_names_for_stage(specs::Vector{RuleSpec}, stage::Symbol) =
    Symbol[spec.public_name for spec in specs if spec.execution_stage == stage]

_rule_impl_names_for_stage(specs::Vector{RuleSpec}, stage::Symbol) =
    Symbol[spec.impl_name for spec in specs if spec.execution_stage == stage]

function _rule_switch_enabled(pparams::GPUBackend.PresolveParams, rule::Symbol)
    field_name = Symbol(:enable_, rule)
    hasfield(typeof(pparams), field_name) || return true
    return getfield(pparams, field_name)::Bool
end

function _validate_qp_problem(problem::QPProblem)
    m, n = size(problem.A)
    q_m, q_n = size(problem.Q)
    q_m == q_n || throw(ArgumentError("QPProblem.Q must be square, got size ($(q_m), $(q_n))."))
    q_m == n || throw(ArgumentError("Dimension mismatch: size(Q,1) = $(q_m), size(A,2) = $(n)."))
    length(problem.c) == n || throw(ArgumentError("Dimension mismatch: length(c) = $(length(problem.c)), size(A,2) = $(n)."))
    length(problem.l) == n || throw(ArgumentError("Dimension mismatch: length(l) = $(length(problem.l)), size(A,2) = $(n)."))
    length(problem.u) == n || throw(ArgumentError("Dimension mismatch: length(u) = $(length(problem.u)), size(A,2) = $(n)."))
    length(problem.AL) == m || throw(ArgumentError("Dimension mismatch: length(AL) = $(length(problem.AL)), size(A,1) = $(m)."))
    length(problem.AU) == m || throw(ArgumentError("Dimension mismatch: length(AU) = $(length(problem.AU)), size(A,1) = $(m)."))
    return nothing
end

function _qp_symmetrize(Q::SparseMatrixCSC{Float64,Int32}; tol::Float64=1.0e-10)
    if nnz(Q) == 0
        return Q
    end

    Q_diff = Q - transpose(Q)
    max_dev = norm(Q_diff, Inf)
    if max_dev > tol
        @warn "QPProblem.Q is not symmetric (max deviation: $(max_dev)). Using symmetric part Q <- 0.5*(Q + Q')."
    end

    return SparseMatrixCSC{Float64,Int32}(0.5 .* (Q + transpose(Q)))
end

function _qp_default_gpu_presolve_params()
    return build_custom_presolve_params(default_rule_switches(presolve_scheduler_mode=:fixed))
end

const QP_NATIVE_GPU_COL_RULES = Set([:close_bounds, :empty_cols, :singleton_cols_dual_infer, :singleton_cols_eq, :dual_fix, :parallel_cols, :doubleton_eq, :linear_eq_agg])
const QP_NATIVE_GPU_ROW_RULES = Set([:empty_rows, :singleton_rows, :activity_checks, :primal_propagation, :parallel_rows])

function _resolve_qp_native_rule_plan!(pparams::GPUBackend.PresolveParams)
    ignored = Symbol[]
    for rule in pparams.row_rule_order
        if !(rule in QP_NATIVE_GPU_ROW_RULES) && _rule_switch_enabled(pparams, rule)
            push!(ignored, rule)
        end
    end
    for rule in pparams.col_rule_order
        if !(rule in QP_NATIVE_GPU_COL_RULES) && _rule_switch_enabled(pparams, rule)
            push!(ignored, rule)
        end
    end
    ignored = unique(ignored)
    if !isempty(ignored) && pparams.verbose
        @warn "Ignoring QP rules that are not in the QP-native scheduler yet: $(join(string.(ignored), ", "))."
    end

    pparams.row_rule_order = Symbol[
        rule for rule in pparams.row_rule_order
        if rule in QP_NATIVE_GPU_ROW_RULES && _rule_switch_enabled(pparams, rule)
    ]
    pparams.col_rule_order = Symbol[
        rule for rule in pparams.col_rule_order
        if rule in QP_NATIVE_GPU_COL_RULES && _rule_switch_enabled(pparams, rule)
    ]
    return nothing
end

function _run_qp_gpu_presolve(
    problem::QPProblem,
    config::PresolveConfig,
)
    Q_sym = _qp_symmetrize(problem.Q)
    pparams = if isnothing(config.presolve_params)
        _qp_default_gpu_presolve_params()
    else
        deepcopy(config.presolve_params)
    end
    pparams.verbose = pparams.verbose || config.verbose
    pparams.record_postsolve_tape = true
    pparams.host_iteration_callback = nothing
    _resolve_qp_native_rule_plan!(pparams)

    original_qp_gpu = setup_gpu_qp_model(
        Q_sym,
        problem.A,
        problem.c,
        problem.AL,
        problem.AU,
        problem.l,
        problem.u,
        problem.obj_constant;
        device_number=config.device_number,
        verbose=config.verbose,
    )

    t_core_start = time()
    reduced_qp_gpu, record = GPUBackend.presolve_gpu(
        original_qp_gpu;
        presolve_params=pparams,
        verbose=config.verbose,
    )
    CUDA.synchronize()
    presolve_core_time = time() - t_core_start

    t_copy_back_start = time()
    Q_red, A_red, c_red, AL_red, AU_red, l_red, u_red, obj_constant_red =
        copy_qp_model_to_cpu(reduced_qp_gpu)
    copy_back_time = time() - t_copy_back_start
    if isnothing(record)
        return problem, nothing
    end

    original_model_gpu = LP_info_gpu(
        original_qp_gpu.A,
        original_qp_gpu.AT,
        original_qp_gpu.c,
        original_qp_gpu.AL,
        original_qp_gpu.AU,
        original_qp_gpu.l,
        original_qp_gpu.u,
        original_qp_gpu.obj_constant,
        original_qp_gpu.AT_leading_slack,
        original_qp_gpu.AT_slack_after,
    )
    raw_state = GPUBackend.PresolveState(
        record,
        original_model_gpu,
        presolve_core_time,
        copy_back_time,
    )

    raw_state.record.obj_constant_new = obj_constant_red
    reduced_problem = QPProblem(
        Q_red,
        A_red,
        c_red,
        AL_red,
        AU_red,
        l_red,
        u_red,
        obj_constant_red,
    )
    return reduced_problem, raw_state
end

_wrap_presolve_state(state::Nothing) = nothing
_wrap_presolve_state(state::GPUBackend.PresolveState) = GPUBackendStateHandle(state)

function _run_gpu_presolve(
    model::LP_info_cpu,
    config::PresolveConfig,
)
    settings = GPUBackend.Settings(
        verbose=config.verbose,
        device_number=config.device_number,
        presolve_params=config.presolve_params,
    )
    raw_state, reduced_model = GPUBackend.run_presolve(model; settings=settings)
    if isnothing(raw_state) || isnothing(reduced_model)
        return model, nothing
    end
    return reduced_model, raw_state
end

function run_presolve(
    problem::LPProblem;
    config::PresolveConfig=PresolveConfig(),
)
    backend = normalize_presolve_backend(config.backend)
    reduced_model = problem.model
    raw_state = nothing
    if backend == "GPU"
        reduced_model, raw_state = _run_gpu_presolve(problem.model, config)
    end

    state = _wrap_presolve_state(raw_state)
    presolve_time = if isnothing(raw_state) || !hasproperty(raw_state, :presolve_core_time)
        0.0
    else
        getproperty(raw_state, :presolve_core_time)
    end

    status = if backend == "NONE"
        "SKIPPED"
    elseif isnothing(raw_state)
        "PRESOLVE_FAILED"
    else
        "OK"
    end

    return PresolveResult(
        LPProblem(reduced_model),
        state,
        backend,
        status,
        presolve_time,
    )
end

function run_presolve(
    model::LP_info_cpu;
    config::PresolveConfig=PresolveConfig(),
)
    return run_presolve(LPProblem(model); config=config)
end

@inline function _normalize_problem_type(problem_type::Union{AbstractString,Symbol})
    normalized = uppercase(strip(String(problem_type)))
    if normalized in ("LP", "QP")
        return normalized
    end
    throw(ArgumentError(
        "Unsupported problem_type=$(repr(problem_type)). Expected \"LP\" or \"QP\".",
    ))
end

function _build_problem_from_mps(
    file_name::AbstractString,
    problem_type::String;
    verbose::Bool=true,
    mpsformat::Symbol=:auto,
)
    if problem_type == "QP"
        Q, A, c, AL, AU, l, u, c0 = build_from_mps_qp(file_name, verbose; mpsformat=mpsformat)
        return QPProblem(Q, A, c, AL, AU, l, u, c0)
    end

    return LPProblem(build_from_mps(file_name, verbose; mpsformat=mpsformat))
end

"""
    run_presolve(file_name::AbstractString; problem_type="LP", config=PresolveConfig(), mpsformat=:auto)

Build an LP/QP problem from an MPS file and run presolve in one call.

`problem_type` accepts `"LP"` or `"QP"` (also `:LP` / `:QP`).
"""
function run_presolve(
    file_name::AbstractString;
    problem_type::Union{AbstractString,Symbol}="LP",
    config::PresolveConfig=PresolveConfig(),
    mpsformat::Symbol=:auto,
)
    normalized_type = _normalize_problem_type(problem_type)
    backend = normalize_presolve_backend(config.backend)
    problem = _build_problem_from_mps(
        file_name,
        normalized_type;
        verbose=config.verbose,
        mpsformat=mpsformat,
    )
    return run_presolve(problem; config=config)
end

function run_presolve(
    problem::QPProblem;
    config::PresolveConfig=PresolveConfig(),
)
    _validate_qp_problem(problem)
    backend = normalize_presolve_backend(config.backend)

    reduced_problem = problem
    raw_state = nothing
    if backend == "GPU"
        reduced_problem, raw_state = _run_qp_gpu_presolve(problem, config)
    end

    state = _wrap_presolve_state(raw_state)
    presolve_time = if isnothing(raw_state) || !hasproperty(raw_state, :presolve_core_time)
        0.0
    else
        getproperty(raw_state, :presolve_core_time)
    end
    status = if backend == "NONE"
        "SKIPPED"
    elseif isnothing(raw_state)
        "PRESOLVE_FAILED"
    else
        "OK"
    end

    return PresolveResult(
        reduced_problem,
        state,
        backend,
        status,
        presolve_time,
    )
end

function run_postsolve(
    state::GPUBackendStateHandle,
    x_red::Vector{Float64},
    y_red::Vector{Float64},
    z_red::Vector{Float64};
    presolve_params=nothing,
)
    return GPUBackend.run_postsolve(
        state.inner,
        x_red,
        y_red,
        z_red;
        presolve_params=presolve_params,
    )
end

function free_presolve_state!(state::Nothing)
    return nothing
end

function free_presolve_state!(state::GPUBackendStateHandle)
    return GPUBackend.free_presolve_state!(state.inner)
end
