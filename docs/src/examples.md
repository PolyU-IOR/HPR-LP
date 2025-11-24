# Examples

Complete, runnable examples demonstrating HPRLP usage. More examples coming soon!

For detailed guides on each input method, see:
- [Direct API Guide](guide/direct_api.md)
- [JuMP Integration Guide](guide/jump_integration.md)  
- [MPS Files Guide](guide/mps_files.md)

## Example 1: Direct API - Basic LP

Solve a simple 2-variable LP problem using matrices.

```julia
using HPRLP
using SparseArrays

# Problem:
# min  -3x₁ - 5x₂
# s.t.  x₁ + 2x₂ ≤ 10
#      3x₁ +  x₂ ≤ 12
#      x₁, x₂ ≥ 0

# Standard form: AL ≤ Ax ≤ AU
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
c = [-3.0, -5.0]
l = [0.0, 0.0]
u = [Inf, Inf]

params = HPRLP_parameters()
params.use_gpu = false

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Status: ", result.output_type)
println("Objective: ", result.primal_obj)
println("Solution: x₁ = ", result.x[1], ", x₂ = ", result.x[2])
```

## Example 2: MPS Files

Read and solve a problem from an MPS file.

```julia
using HPRLP

params = HPRLP_parameters()
params.stoptol = 1e-6
params.use_gpu = true
params.verbose = true

result = run_single("problem.mps", params)

if result.output_type == "OPTIMAL"
    println("✓ Optimal solution found!")
    println("  Objective: ", result.primal_obj)
    println("  Time: ", result.time, " seconds")
end
```

## Example 3: JuMP Integration

Build and solve using JuMP's modeling language.

```julia
using JuMP, HPRLP

model = Model(HPRLP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-4)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, -3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

optimize!(model)

println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
println("x1 = ", value(x1), ", x2 = ", value(x2))
```


## More Examples

*More examples will be added in future releases.*
