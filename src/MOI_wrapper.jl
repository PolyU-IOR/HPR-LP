# MOI Wrapper for HPRLP
# Based on the structure of Clp.jl's MOI wrapper

import MathOptInterface as MOI

# Supported scalar sets
const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

"""
    Optimizer()

Create a new HPRLP Optimizer object.

Set optimizer attributes using `MOI.RawOptimizerAttribute` or
`JuMP.set_optimizer_attribute`.

## Example

```julia
using JuMP, HPRLP
model = JuMP.Model(HPRLP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-4)
set_optimizer_attribute(model, "use_gpu", true)
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    params::HPRLP_parameters
    results::Union{Nothing, HPRLP_results}
    silent::Bool
    cache::Union{Nothing, OptimizerCache}
    index_map::Union{Nothing, MOI.IndexMap}
    obj_sense::MOI.OptimizationSense  # Track objective sense for result conversion
    
    function Optimizer()
        return new(HPRLP_parameters(), nothing, false, nothing, nothing, MOI.MIN_SENSE)
    end
end

# ====================
#   Utility functions
# ====================

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

# ====================
#   Empty functions
# ====================

function MOI.is_empty(model::Optimizer)
    return model.results === nothing
end

function MOI.empty!(model::Optimizer)
    model.results = nothing
    model.cache = nothing
    model.index_map = nothing
    model.obj_sense = MOI.MIN_SENSE  # Reset to default
    return
end

# ====================
#   Solver attributes
# ====================

MOI.get(::Optimizer, ::MOI.SolverName) = "HPRLP"

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    return "0.1.0"  # Update this to match your package version
end

# HPRLP does not support incremental interface - requires copy_to
MOI.supports_incremental_interface(::Optimizer) = false

# Silent mode
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

# Time limit
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    model.params.time_limit = Float64(value)
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    model.params.time_limit = 3600.0  # Default value
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return model.params.time_limit
end

# Number of threads (not supported)
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false

# Raw optimizer attributes for HPRLP parameters
const SUPPORTED_PARAMETERS = (
    "stoptol",
    "max_iter",
    "time_limit",
    "check_iter",
    "use_Ruiz_scaling",
    "use_Pock_Chambolle_scaling",
    "use_bc_scaling",
    "use_gpu",
    "device_number",
    "warm_up",
    "print_frequency",
    "verbose",
)

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return param.name in SUPPORTED_PARAMETERS
end

function MOI.set(model::Optimizer, param::MOI.RawOptimizerAttribute, value)
    name = String(param.name)
    if name == "stoptol"
        model.params.stoptol = Float64(value)
    elseif name == "max_iter"
        model.params.max_iter = Int(value)
    elseif name == "time_limit"
        model.params.time_limit = Float64(value)
    elseif name == "check_iter"
        model.params.check_iter = Int(value)
    elseif name == "use_Ruiz_scaling"
        model.params.use_Ruiz_scaling = Bool(value)
    elseif name == "use_Pock_Chambolle_scaling"
        model.params.use_Pock_Chambolle_scaling = Bool(value)
    elseif name == "use_bc_scaling"
        model.params.use_bc_scaling = Bool(value)
    elseif name == "use_gpu"
        model.params.use_gpu = Bool(value)
    elseif name == "device_number"
        model.params.device_number = Int(value)
    elseif name == "warm_up"
        model.params.warm_up = Bool(value)
    elseif name == "print_frequency"
        model.params.print_frequency = Int(value)
    elseif name == "verbose"
        model.params.verbose = Bool(value)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(model::Optimizer, param::MOI.RawOptimizerAttribute)
    name = String(param.name)
    if name == "stoptol"
        return model.params.stoptol
    elseif name == "max_iter"
        return model.params.max_iter
    elseif name == "time_limit"
        return model.params.time_limit
    elseif name == "check_iter"
        return model.params.check_iter
    elseif name == "use_Ruiz_scaling"
        return model.params.use_Ruiz_scaling
    elseif name == "use_Pock_Chambolle_scaling"
        return model.params.use_Pock_Chambolle_scaling
    elseif name == "use_bc_scaling"
        return model.params.use_bc_scaling
    elseif name == "use_gpu"
        return model.params.use_gpu
    elseif name == "device_number"
        return model.params.device_number
    elseif name == "warm_up"
        return model.params.warm_up
    elseif name == "print_frequency"
        return model.params.print_frequency
    elseif name == "verbose"
        return model.params.verbose
    end
    throw(MOI.UnsupportedAttribute(param))
end

# ========================================
#   Supported constraints and objectives
# ========================================

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,MOI.ScalarAffineFunction{Float64}}},
    ::Type{<:SCALAR_SETS},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

# =======================
#   `copy_to` function
# =======================

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    # This is called with the cache directly
    @assert MOI.is_empty(dest)
    dest.cache = src
    dest.index_map = _index_map(src)
    return dest.index_map
end

function MOI.copy_to(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    # Throw error if there are unsupported constraints
    # This validates the model is actually LP (no quadratic, conic, integer, etc.)
    MOI.Utilities.throw_unsupported(src)
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    # For general MOI models, copy to cache first
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    
    # Copy from cache to optimizer
    MOI.copy_to(dest, cache)
    
    return index_map
end

# Helper function to create index map from cache
function _index_map(src::OptimizerCache)
    index_map = MOI.IndexMap()
    # Map variables (1-indexed)
    for (i, x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        index_map[x] = MOI.VariableIndex(i)
    end
    # Map constraints
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        _index_map_constraints(src, index_map, F, S)
    end
    return index_map
end

function _index_map_constraints(
    src::OptimizerCache,
    index_map,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{S},
) where {S<:SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},S}())
        row = MOI.Utilities.rows(src.constraints, ci)
        index_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}(row)
    end
    return
end

function _index_map_constraints(
    src::OptimizerCache,
    index_map,
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {S<:SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
        col = index_map[MOI.VariableIndex(ci.value)].value
        index_map[ci] = MOI.ConstraintIndex{MOI.VariableIndex,S}(col)
    end
    return
end

# ===============================
#   Optimize and post-optimize
# ===============================

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    # Extract LP data
    A = src.constraints.coefficients
    row_bounds = src.constraints.constants
    
    # Extract objective
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    c = zeros(A.n)
    for term in obj.terms
        c[term.variable.value] += term.coefficient
    end
    obj_constant = obj.constant
    
    # Handle objective sense
    sense = MOI.get(src, MOI.ObjectiveSense())
    dest.obj_sense = sense  # Store for later use in result retrieval
    if sense == MOI.MAX_SENSE
        c = -c
        obj_constant = -obj_constant
    end
    
    # Extract variable bounds
    l = src.variables.lower
    u = src.variables.upper
    
    # Extract constraint bounds
    AL = row_bounds.lower
    AU = row_bounds.upper
    
    # Convert to standard Julia SparseMatrixCSC
    A_sparse = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval)
    
    # Set verbose: if silent mode is set, disable verbose regardless of parameter
    # Otherwise, use the parameter setting
    if dest.silent
        dest.params.verbose = false
    end
    
    # Run the HPRLP solver
    dest.results = run_lp(A_sparse, AL, AU, c, l, u, obj_constant, dest.params)
    
    return
end

function MOI.optimize!(model::Optimizer)
    # Extract from stored cache
    if model.cache === nothing
        error("No problem has been loaded. Use JuMP.Model(HPRLP.Optimizer) and build your model first.")
    end
    
    MOI.optimize!(model, model.cache)
    return
end

# Solve time
function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    if model.results === nothing
        return 0.0
    end
    return model.results.time
end

# Objective value
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available. Call optimize! first.")
    end
    value = model.results.primal_obj
    # Convert back if this was a maximization problem
    if model.obj_sense == MOI.MAX_SENSE
        value = -value
    end
    return value
end

# Number of variables
function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if model.results === nothing
        return 0
    end
    return length(model.results.x)
end

# ===============================
#   Termination and Result Status
# ===============================

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.results === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    
    if model.results.output_type == "OPTIMAL"
        return MOI.OPTIMAL
    elseif model.results.output_type == "MAX_ITER"
        return MOI.ITERATION_LIMIT
    elseif model.results.output_type == "TIME_LIMIT"
        return MOI.TIME_LIMIT
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.results === nothing
        return "OPTIMIZE_NOT_CALLED"
    end
    return model.results.output_type
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.results === nothing
        return 0
    end
    # HPRLP always returns a result if it has run
    return model.results.output_type == "OPTIMAL" ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.output_type == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.output_type == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

# ===================
#   Primal solution
# ===================

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return model.results.x[x.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    xs::Vector{MOI.VariableIndex},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return [model.results.x[x.value] for x in xs]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    # This would require storing the constraint matrix and computing Ax
    # For now, return a fallback
    return MOI.Utilities.get_fallback(model, attr, c)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

# =================
#   Dual solution
# =================

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return dual for row constraint
    return model.results.y[c.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return reduced cost (should be non-positive for upper bounds)
    return min(0.0, model.results.z[c.value])
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return reduced cost (should be non-negative for lower bounds)
    return max(0.0, model.results.z[c.value])
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{
        MOI.VariableIndex,
        <:Union{MOI.Interval{Float64},MOI.EqualTo{Float64}},
    },
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # For interval and equality constraints, return the full reduced cost
    return model.results.z[c.value]
end
