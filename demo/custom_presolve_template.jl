using SparseArrays
import HPRLP

# =============================================================================
# Custom Presolve Template for HPR-LP
# =============================================================================
#
# 1) Copy this file into your own Julia project and adapt run_custom_presolve /
#    run_custom_postsolve for your own reduction/reconstruction logic.
# 2) Make sure the method definitions stay in scope before calling optimize.

struct MyPresolveState
    # Save anything needed to lift reduced solution back.
    # For identity demo this just tracks a constant offset.
    obj_constant_offset::Float64
end

"""
    HPRLP.build_custom_presolve_state(original_model, reduced_model, params; presolve_params=nothing)

Build an optional state object for postsolve. The default return is any opaque object.
"""
function HPRLP.build_custom_presolve_state(
    original_model::Union{HPRLP.LP_info_cpu,HPRLP.LP_info_gpu},
    reduced_model::Union{HPRLP.LP_info_cpu,HPRLP.LP_info_gpu},
    params::HPRLP.HPRLP_parameters;
    presolve_params=nothing,
)
    return MyPresolveState(reduced_model.obj_constant - original_model.obj_constant)
end

"""
    HPRLP.run_custom_presolve(model, params; presolve_params=nothing)

Perform your custom reduction here. Return:
  (reduced_model, presolve_state)
where `presolve_state` will be passed to `run_custom_postsolve`.
"""
function HPRLP.run_custom_presolve(
    model::Union{HPRLP.LP_info_cpu,HPRLP.LP_info_gpu},
    params::HPRLP.HPRLP_parameters;
    presolve_params=nothing,
)
    # Example implementation: identity presolve (no matrix changes)
    state = HPRLP.build_custom_presolve_state(model, model, params; presolve_params=presolve_params)
    return model, state
end

"""
    HPRLP.run_custom_postsolve(state, x_red, y_red, z_red; presolve_params=nothing)

Lift reduced solution back to the original model. Return `(x, y, z)` in original order.
"""
function HPRLP.run_custom_postsolve(
    state::MyPresolveState,
    x_red::AbstractVector,
    y_red::AbstractVector,
    z_red::AbstractVector;
    presolve_params=nothing,
)
    # Example implementation: no-op lift
    return x_red, y_red, z_red
end

function HPRLP.free_custom_presolve_state!(state::MyPresolveState)
    # No external resources, nothing to release.
    return nothing
end

# Example solve with a tiny LP:
#   min -3x1 - 5x2
#   s.t. -x1 - 2x2 >= -10
#        -3x1 - x2 >= -12
#        x1, x2 >= 0
A = sparse([-1 -2; -3 -1])
c = Vector{Float64}([-3.0, -5.0])
AL = Vector{Float64}([-10.0, -12.0])
AU = Vector{Float64}([Inf, Inf])
l = Vector{Float64}([0.0, 0.0])
u = Vector{Float64}([Inf, Inf])
obj_constant = 0.0

model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)
params = HPRLP.HPRLP_parameters()
params.presolve = "CUSTOM"
params.use_postsolve = true
params.use_gpu = false
params.warm_up = false
params.verbose = false

result = HPRLP.optimize(model, params)
println("Status: ", result.status)
println("Primal Obj: ", result.primal_obj)
