module HPRLP

using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using Printf
using CSV
using DataFrames
using Random
using Statistics
using Logging
using JuMP
using HDF5
using Dates
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "MPSReader", "src", "MPSReader.jl"))

include("structs.jl")
include("GPUPresolverAdapter.jl")
include("custom_presolve.jl")
include("utils.jl")
include("folding.jl")
include("kernels.jl")
include("algorithm.jl")
include("MOI_wrapper.jl")
include("PSLP.jl")
using .PSLP

# Export the Optimizer for JuMP usage
export Optimizer

# Export main functions and types for direct API usage
export HPRLP_parameters, HPRLP_results
export build_from_Abc, optimize
export run_folding, unfold_solution, FoldingMap, set_folding_mode!
export load_gpu_presolve_setup
export load_gpu_presolve_params
export build_custom_presolve_state, run_custom_presolve, run_custom_postsolve, free_custom_presolve_state!

end
