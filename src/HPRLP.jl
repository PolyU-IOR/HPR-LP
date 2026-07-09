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
include(joinpath("batch", "structs_gpu.jl"))
include("utils.jl")
include("io.jl")
include("kernels.jl")
include("algorithm.jl")
include(joinpath("batch", "utils_gpu.jl"))
include(joinpath("batch", "kernels_gpu.jl"))
include(joinpath("batch", "algorithm_gpu.jl"))
include(joinpath("batch", "dataset_gpu.jl"))
include("MOI_wrapper.jl")
include("PSLP.jl")
using .PSLP
include("GPUPresolve.jl")
using .GPUPresolve

# Export the Optimizer for JuMP usage
export Optimizer

# Export main functions and types for direct API usage
export HPRLP_parameters, HPRLP_results
export build_from_Abc, build_from_mps, optimize, optimize_from_autosave
export save_lp_to_hdf5, save_mps_as_hdf5, read_from_hdf5, read_from_h5, run_dataset
export BatchedSharedMatrix_gpu, BatchedLPData_gpu, BatchedScalingInfo_gpu, BatchedWorkspace_gpu, BatchedHPRLPResults
export build_batched_shared_matrix_gpu, build_batched_lp_gpu, prepare_batched_gpu_problem, optimize_batched_gpu
export allocate_batched_workspace_gpu, batched_spmm_A!, batched_spmm_AT!, batched_spmm_pair!
export update_x_z_batched_gpu!, update_y_batched_gpu!, compute_batched_residuals_gpu!
export update_x_z_check_batched_gpu!, update_x_z_normal_batched_gpu!
export update_y_check_batched_gpu!, update_y_normal_batched_gpu!
export run_batched_dataset

end
