"""
    build_custom_presolve_state(
        original_model::Union{LP_info_cpu,LP_info_gpu},
        reduced_model::Union{LP_info_cpu,LP_info_gpu},
        params::HPRLP_parameters;
        presolve_params=nothing,
    )

Build and return an opaque state object that can be used by custom presolve postsolve.

Default behavior keeps the original/reduced model pair unchanged and returns `nothing`.
Override this method (in your own file) for stateful custom presolve/postsolve workflows.
"""
function build_custom_presolve_state(
    original_model::Union{LP_info_cpu,LP_info_gpu},
    reduced_model::Union{LP_info_cpu,LP_info_gpu},
    params::HPRLP_parameters;
    presolve_params=nothing,
)
    return nothing
end

"""
    run_custom_presolve(
        model::Union{LP_info_cpu,LP_info_gpu},
        params::HPRLP_parameters;
        presolve_params=nothing,
    )

Default no-op custom presolve hook.

Users implementing a custom presolve should override this method and return
`(reduced_model, presolve_state)` where `presolve_state` can be any object
and will be reused by `run_custom_postsolve` during solution lifting.
"""
function run_custom_presolve(
    model::Union{LP_info_cpu,LP_info_gpu},
    params::HPRLP_parameters;
    presolve_params=nothing,
)
    if params.verbose
        println("CUSTOM PRESOLVE disabled; using original model directly.")
    end
    return model, build_custom_presolve_state(model, model, params; presolve_params=presolve_params)
end

"""
    run_custom_postsolve(
        presolve_state,
        x_red::AbstractVector{<:Real},
        y_red::AbstractVector{<:Real},
        z_red::AbstractVector{<:Real};
        presolve_params=nothing,
    )

Default postsolve handler.

This method should be overridden together with [`run_custom_presolve`] when a custom
backend uses stateful reductions and needs to reconstruct the original solution.
"""
function run_custom_postsolve(
    presolve_state,
    x_red::AbstractVector,
    y_red::AbstractVector,
    z_red::AbstractVector;
    presolve_params=nothing,
)
    error(
        "Custom postsolve is not implemented for presolve state type ",
        typeof(presolve_state),
        ". Please define run_custom_postsolve for your custom backend."
    )
end

"""
    free_custom_presolve_state!(state)

Release or detach custom presolve resources.

Override if your custom presolve state owns external resources.
The default implementation is a no-op.
"""
free_custom_presolve_state!(::Any) = nothing

