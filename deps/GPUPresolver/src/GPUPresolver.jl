module GPUPresolver

using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using Printf
using CSV
using DataFrames
using Statistics
using Logging
using TOML
using QPSReader
using CodecZlib

include(joinpath(@__DIR__, "..", "MPSReader", "src", "MPSReader.jl"))

include(joinpath(@__DIR__, "core", "structs.jl"))
include(joinpath(@__DIR__, "core", "utils.jl"))
include(joinpath(@__DIR__, "core", "presolve_support.jl"))
include(joinpath(@__DIR__, "backends", "GPUBackend.jl"))
using .GPUBackend
include(joinpath(@__DIR__, "api", "presolve_api.jl"))

# Export presolve-focused API
export build_from_Abc, build_from_mps, build_from_mps_qp
export AbstractPresolveProblem, AbstractPresolveState
export LPProblem, QPProblem
export PresolveConfig, PresolveResult
export default_config_path, load_default_presolve_setup, load_presolve_setup
export run_presolve, run_postsolve, free_presolve_state!
export compute_original_kkt_metrics, check_org_recovery_failures
export GPUPresolverParameters

end
