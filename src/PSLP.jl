module PSLP

using Libdl
using SparseArrays

# ================= Path Configuration =================
const DEPS_DIR = joinpath(@__DIR__, "..", "deps")
const LIB_NAME = "libPSLP.$(Libdl.dlext)"
const LIB_PATH = joinpath(DEPS_DIR, LIB_NAME)

function __init__()
    return nothing
end

is_available() = isfile(LIB_PATH)

function _ensure_available()
    is_available() || error("PSLP dynamic library not found at $LIB_PATH.\n" * "Please check your build steps.")
    return nothing
end

# ================= C Struct Definitions =================
# Fixed: Cint -> Csize_t to match C 'size_t' (8 bytes on 64-bit)
struct PresolvedProblem
    Ax::Ptr{Float64}; Ai::Ptr{Cint}; Ap::Ptr{Cint}
    m::Csize_t; n::Csize_t; nnz::Csize_t  # Changed from Cint to Csize_t
    lhs::Ptr{Float64}; rhs::Ptr{Float64}; c::Ptr{Float64}
    lbs::Ptr{Float64}; ubs::Ptr{Float64}; obj_offset::Float64
end

Base.@kwdef struct Settings
    ston_cols::Bool = true
    dton_eq::Bool = true
    parallel_rows::Bool = true
    parallel_cols::Bool = true
    primal_propagation::Bool = true
    finite_bound_tightening::Bool = true
    dual_fix::Bool = true
    relax_bounds::Bool = false
    max_shift::Cint = 10
    max_time::Float64 = 60.0
    verbose::Bool = true
end

struct Solution
    x::Ptr{Float64}; y::Ptr{Float64}; z::Ptr{Float64}
    dim_x::Csize_t; dim_y::Csize_t        # Changed from Cint to Csize_t
end

struct PresolverStruct
    stats::Ptr{Cvoid}; stgs::Ptr{Settings}; prob::Ptr{Cvoid}
    reduced_prob::Ptr{PresolvedProblem}; sol::Ptr{Solution}
end

# ================= Wrapper Handle =================
mutable struct PresolverModel
    ptr::Ptr{PresolverStruct}
    settings_ptr::Ptr{Settings}
    
    function PresolverModel(ptr::Ptr{PresolverStruct}, settings_ptr::Ptr{Settings})
        obj = new(ptr)
        obj.settings_ptr = settings_ptr
        finalizer(free_presolver_wrapper, obj)
        return obj
    end
end

function free_presolver_wrapper(model::PresolverModel)
    if model.ptr != C_NULL
        _with_silent_stdio() do
            ccall((:free_presolver, LIB_PATH), Cvoid, (Ptr{Cvoid},), model.ptr)
        end
        model.ptr = C_NULL
    end
    if model.settings_ptr != C_NULL
        ccall((:free_settings, LIB_PATH), Cvoid, (Ptr{Settings},), model.settings_ptr)
        model.settings_ptr = C_NULL
    end
end

function _default_settings_ptr()
    _ensure_available()
    return ccall((:default_settings, LIB_PATH), Ptr{Settings}, ())
end

function default_settings_struct()
    ptr = _default_settings_ptr()
    ptr == C_NULL && error("PSLP failed to allocate default settings.")
    stgs = unsafe_load(ptr)
    ccall((:free_settings, LIB_PATH), Cvoid, (Ptr{Settings},), ptr)
    return stgs
end

function settings_all_false()
    stgs = default_settings_struct()
    return Settings(
        ston_cols=false,
        dton_eq=false,
        parallel_rows=false,
        parallel_cols=false,
        primal_propagation=false,
        finite_bound_tightening=false,
        dual_fix=false,
        relax_bounds=false,
        max_shift=stgs.max_shift,
        max_time=stgs.max_time,
        verbose=false,
    )
end

function _allocate_settings_ptr(settings::Union{Nothing,Settings}=nothing)
    ptr = _default_settings_ptr()
    ptr == C_NULL && error("PSLP failed to allocate settings.")
    if settings !== nothing
        unsafe_store!(ptr, settings)
    end
    return ptr
end

function _with_silent_stdio(f::Function)
    open("/dev/null", "w") do io
        Base.redirect_stdio(stdout=io, stderr=io) do
            ret = f()
            ccall(:fflush, Cint, (Ptr{Cvoid},), C_NULL) 
            return ret
        end
    end
end


# ================= Core Functions =================

function load_and_run_presolve(
    c::Vector{Float64},
    A::SparseMatrixCSC{Float64,<:Integer},
    l::Vector{Float64},
    u::Vector{Float64},
    lhs::Vector{Float64},
    rhs::Vector{Float64};
    settings::Union{Nothing,Settings}=nothing,
)
    
    # 1. Convert CSC to CSR (using transpose trick)
    A_trans = sparse(transpose(A))
    
    Ax_c = Vector{Float64}(A_trans.nzval)
    Ai_c = Vector{Cint}(A_trans.rowval .- 1) 
    Ap_c = Vector{Cint}(A_trans.colptr .- 1)
    
    m, n = size(A)
    nnz_val = nnz(A)

    # 2. Call C Interface
    stgs_ptr = _allocate_settings_ptr(settings)
    
    # Updated ccall signature: Cint -> Csize_t
    ptr = GC.@preserve Ax_c Ai_c Ap_c lhs rhs l u c begin
        ccall((:new_presolver, LIB_PATH), Ptr{PresolverStruct},
              (Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Csize_t, Csize_t, Csize_t,
               Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Settings}),
              Ax_c, Ai_c, Ap_c, Csize_t(m), Csize_t(n), Csize_t(nnz_val), lhs, rhs, l, u, c, stgs_ptr)
    end

    if ptr == C_NULL
        ccall((:free_settings, LIB_PATH), Cvoid, (Ptr{Settings},), stgs_ptr)
        return nothing, nothing
    end

    model = PresolverModel(ptr, stgs_ptr)

    # 3. Run Presolve
    # status = ccall((:run_presolver, LIB_PATH), Cint, (Ptr{PresolverStruct},), model.ptr)
    status = _with_silent_stdio() do
        ccall((:run_presolver, LIB_PATH), Cint, (Ptr{PresolverStruct},), model.ptr)
    end

    
    # 4. Extract Reduced Data
    presolver_data = unsafe_load(model.ptr)
    
    if presolver_data.reduced_prob == C_NULL
        return model, nothing 
    end
    
    red_prob = unsafe_load(presolver_data.reduced_prob)

    m_red, n_red, nnz_red = red_prob.m, red_prob.n, red_prob.nnz
    
    c_red   = unsafe_wrap(Array, red_prob.c, n_red) |> copy
    lbs_red = unsafe_wrap(Array, red_prob.lbs, n_red) |> copy
    ubs_red = unsafe_wrap(Array, red_prob.ubs, n_red) |> copy
    lhs_red = unsafe_wrap(Array, red_prob.lhs, m_red) |> copy
    rhs_red = unsafe_wrap(Array, red_prob.rhs, m_red) |> copy
    
    Ax_ptr = unsafe_wrap(Array, red_prob.Ax, nnz_red)
    Ai_ptr = unsafe_wrap(Array, red_prob.Ai, nnz_red)
    Ap_ptr = unsafe_wrap(Array, red_prob.Ap, m_red + 1)
    
    # Convert CSR back to CSC for Julia
    A_red_trans = SparseMatrixCSC(n_red, m_red, Ap_ptr .+ 1, Ai_ptr .+ 1, Ax_ptr)
    A_red = sparse(transpose(A_red_trans))

    return model, (c_red, A_red, lbs_red, ubs_red, lhs_red, rhs_red, red_prob.obj_offset)
end

"""
    postsolve(model, x_red, y_red, z_red)
    
Recovers full x, y, z from reduced solutions.
"""
function postsolve(model::PresolverModel, 
                   x_red::Vector{Float64}, 
                   y_red::Vector{Float64}, 
                   z_red::Vector{Float64})

    if model.ptr == C_NULL
        error("Presolver model has been freed or is invalid.")
    end

    presolver_data = unsafe_load(model.ptr)
    
    if presolver_data.reduced_prob == C_NULL
        # If no reduced problem, maybe original problem was solved directly or empty?
        # But generally this shouldn't happen if load_and_run succeeded.
        error("PSLP Error: reduced_prob is NULL during postsolve.")
    end

    # Call C Postsolve
    # GC.@preserve x_red y_red z_red begin
    #     ccall((:postsolve, LIB_PATH), Cvoid,
    #           (Ptr{PresolverStruct}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
    #           model.ptr, x_red, y_red, z_red)
    # end
    _with_silent_stdio() do
        GC.@preserve x_red y_red z_red begin
            ccall((:postsolve, LIB_PATH), Cvoid,
                (Ptr{PresolverStruct}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                model.ptr, x_red, y_red, z_red)
        end
    end

    # Retrieve Full Solutions
    presolver_data_updated = unsafe_load(model.ptr)
    sol_ptr = presolver_data_updated.sol
    
    if sol_ptr == C_NULL
        error("PSLP Error: Solution struct is NULL after postsolve.")
    end
    
    sol_data = unsafe_load(sol_ptr)
    
    # Extract arrays
    x_full = copy(unsafe_wrap(Array, sol_data.x, sol_data.dim_x))
    y_full = copy(unsafe_wrap(Array, sol_data.y, sol_data.dim_y))
    z_full = copy(unsafe_wrap(Array, sol_data.z, sol_data.dim_x))

    return x_full, y_full, z_full
end

end # module
