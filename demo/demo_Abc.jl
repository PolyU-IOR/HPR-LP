using SparseArrays
using LinearAlgebra
import HPRLP

# =============================================================================
# Example: Using the API with build_from_Abc and optimize
# =============================================================================

# min <c,x>
# s.t. AL <= Ax <= AU
#       l <= x <= u

# Example: With warmup (recommended for better performance)
# min -3x1 - 5x2
# s.t. -x1 - 2x2 >= -10
#      -3x1 - x2 >= -12
#      x1 >= 0, x2 >= 0
#      x1 <= Inf, x2 <= Inf

println("="^80)
println("Example: Using build_from_Abc and optimize with warmup (recommended)")
println("="^80)

A = sparse([-1 -2; -3 -1])
c = Vector{Float64}([-3, -5])
AL = Vector{Float64}([-10, -12])
AU = Vector{Float64}([Inf, Inf])
l = Vector{Float64}([0, 0])
u = Vector{Float64}([Inf, Inf])

obj_constant = 0.0

# Build the model (no scaling yet)
model = HPRLP.build_from_Abc(A, c, AL, AU, l, u, obj_constant)

# Set up parameters with warmup enabled
params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.use_gpu = true
params.warm_up = true

# Optimize the model (scaling and warmup happen inside optimize)
result = HPRLP.optimize(model, params)

println()
println("Results:")
println("  Objective value: ", result.primal_obj)
println("  x1 = ", result.x[1])
println("  x2 = ", result.x[2])
println("  Status: ", result.status)

