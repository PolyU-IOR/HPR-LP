#=

Data Structures
    •   LP_info_cpu / LP_info_gpu -> Stores LP problem data in CPU/GPU memory.
    •   GPUPresolverParameters -> Public-facing runtime settings for package entrypoints and scripts.

This file is intentionally for shared/public data containers used across API,
backend, and support layers. Presolve-internal execution state belongs in
`src/presolve/presolve_structs.jl`.

=#

"""
    GPUPresolverParameters

Public-facing runtime settings for package entrypoints and scripts.

This is the outer configuration surface for package calls and development
scripts. It is distinct from `PresolveParams` in
`src/presolve/presolve_structs.jl`, which stores internal GPU presolve rule
selection and execution controls.

# Fields
- `stoptol::Float64`: Summary/report tolerance used by helper utilities (default: `1e-4`)
- `time_limit::Float64`: Summary/report time cap used by helper utilities (default: `3600.0`)
- `device_number::Int`: GPU device number (default: `0`)
- `verbose::Bool`: Enable verbose output (default: `true`)
- `presolve::String`: Presolve control selector (`"GPU"` or `"NONE"`) (default: `"GPU"`)

# Example
```julia
params = GPUPresolverParameters()
params.device_number = 1
params.verbose = false
params.presolve = "NONE"
```
"""
mutable struct GPUPresolverParameters
    stoptol::Float64
    time_limit::Float64
    device_number::Int
    verbose::Bool
    presolve::String

    GPUPresolverParameters() = new(1e-4, 3600.0, 0, true, "GPU")
end

mutable struct LP_info_cpu
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    obj_constant::Float64
end

mutable struct LP_info_gpu
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    c::CuVector{Float64}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    obj_constant::Float64
    AT_leading_slack::Int32
    AT_slack_after::CuVector{Int32}
end

mutable struct QP_info_gpu
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    Q::CuSparseMatrixCSR{Float64,Int32}
    QT::CuSparseMatrixCSR{Float64,Int32}
    c::CuVector{Float64}
    q_diag::CuVector{Float64}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    obj_constant::Float64
    AT_leading_slack::Int32
    AT_slack_after::CuVector{Int32}
end
