using SparseArrays
using LinearAlgebra
import HprLP


# min <c,x>
# s.t. AL <= Ax <= AU
#       l <= x <= u


# Example 1
# min -3x1 - 5x2
# s.t. -x1 - 2x2 >= -10
#      -3x1 - x2 >= -12
#      x1 >= 0, x2 >= 0
#      x1 <= Inf, x2 <= Inf

A = sparse([-1 -2; -3 -1])
AL = Vector{Float64}([-10, -12])
AU = Vector{Float64}([Inf, Inf])
c = Vector{Float64}([-3, -5])
l = Vector{Float64}([0, 0])
u = Vector{Float64}([Inf, Inf])

obj_constant = 0.0

params = HprLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.use_gpu = true
params.warm_up = true

result = HprLP.run_lp(A, AL, AU, c, l, u, obj_constant, params)

println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
