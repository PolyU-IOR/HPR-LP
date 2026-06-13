import HPRLP
using Printf
using Random

file_name = "model.mps" # Replace with the LP file you want to test

# Batch construction options
batch_size = 4
instance_mode = :copy # :copy or :perturbed_obj
obj_perturbation_scale = 1.0e-3
random_seed = 1234

# Current true-batched GPU path is presolve-free and GPU-only.
params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4
# params.max_iter = 10_000
params.device_number = 0
params.use_gpu = true
params.warm_up = true
params.verbose = true
params.presolve = "NONE"

function repeat_columns(v::Vector{Float64}, batch_size::Int)
    return repeat(reshape(v, :, 1), 1, batch_size)
end

function make_batched_objectives(c::Vector{Float64}, batch_size::Int, mode::Symbol, scale::Float64, rng::AbstractRNG)
    C = repeat_columns(c, batch_size)
    if mode == :copy
        return C
    elseif mode == :perturbed_obj
        for k in 2:batch_size
            C[:, k] .= c .* (1.0 .+ scale .* randn(rng, length(c)))
        end
        return C
    else
        throw(ArgumentError("Unsupported instance_mode $(mode). Use :copy or :perturbed_obj."))
    end
end

function summarize_batched_results(result::HPRLP.BatchedHPRLPResults)
    println("Batched results:")
    for k in eachindex(result.status)
        println(@sprintf(
            "  [%d] status=%s iter=%d primal_obj=%+.12e residual=%.6e gap=%.6e",
            k,
            result.status[k],
            result.iter[k],
            result.primal_obj[k],
            result.residuals[k],
            result.gap[k],
        ))
    end
end

rng = MersenneTwister(random_seed)

println("READING FILE ... ", file_name)
model = HPRLP.build_from_mps(file_name)

C = make_batched_objectives(model.c, batch_size, instance_mode, obj_perturbation_scale, rng)
AL = repeat_columns(model.AL, batch_size)
AU = repeat_columns(model.AU, batch_size)
L = repeat_columns(model.l, batch_size)
U = repeat_columns(model.u, batch_size)
obj_constants = fill(model.obj_constant, batch_size)

println("BATCH CONFIGURATION")
println("  batch_size = ", batch_size)
println("  instance_mode = ", instance_mode)
println("  obj_perturbation_scale = ", obj_perturbation_scale)
println("  matrix sizes:")
println("    A  = ", size(model.A))
println("    C  = ", size(C))
println("    AL = ", size(AL))
println("    AU = ", size(AU))
println("    L  = ", size(L))
println("    U  = ", size(U))
println("  presolve = ", params.presolve)
println("  use_gpu = ", params.use_gpu)
println("  warm_up = ", params.warm_up)
println()

println("RUNNING TRUE BATCHED GPU MODE ...")
wall_time = @elapsed result = HPRLP.optimize_batched_gpu(
    model.A,
    C,
    AL,
    AU,
    L,
    U,
    params;
    obj_constants=obj_constants,
)

summarize_batched_results(result)
println()
println(@sprintf("Script wall time: %.4fs", wall_time))
println(@sprintf("Reported total time: %.4fs", result.time))
println(@sprintf("  setup time: %.4fs", result.setup_time))
println(@sprintf("  solve time: %.4fs", result.solve_time))
println(@sprintf("  power time: %.4fs", result.power_time))
if params.warm_up
    println("  benchmark note: script wall time includes warm-up; reported times exclude warm-up")
end

println()
println("NOTE")
println("  This script uses the true matrix-form batched GPU path: C/L/U are n x B and AL/AU are m x B.")
println("  Current batched mode uses SpMM for sparse products, batched dense update kernels, and per-column restart/sigma state; presolve is still disabled.")