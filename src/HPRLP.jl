module Hprlp

using SparseArrays
using LinearAlgebra
using QPSReader
using CUDA
using CUDA.CUSPARSE
using Printf
using CSV
using DataFrames
using Random
using Statistics
using Logging
using JuMP
import MathOptInterface as MOI

include("structs.jl")
include("utils.jl")
include("kernels.jl")
include("algorithm.jl")
include("MOI_wrapper.jl")

# Export the Optimizer for JuMP usage
export Optimizer

# Export main functions and types for direct API usage
export run_lp, run_single, HPRLP_parameters, HPRLP_results

end