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
include("utils.jl")
include("kernels.jl")
include("algorithm.jl")
include("MOI_wrapper.jl")
include("PSLP.jl")
using .PSLP
include("GPUPresolve.jl")
using .GPUPresolve

# Export the Optimizer for JuMP usage
export Optimizer

# Export main functions and types for direct API usage
export HPRLP_parameters, HPRLP_results
export build_from_mps, build_from_Abc, optimize

end