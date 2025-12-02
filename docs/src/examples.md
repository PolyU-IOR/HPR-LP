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
c = [-3.0, -5.0]
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

# Build and solve
model = build_from_Abc(A, c, AL, AU, l, u)

params = HPRLP_parameters()
params.use_gpu = false

result = optimize(model, params)

println("Status: ", result.status)
println("Objective: ", result.primal_obj)
println("Solution: x₁ = ", result.x[1], ", x₂ = ", result.x[2])
```

## Example 2: MPS Files

Read and solve a problem from an MPS file.

```julia
using HPRLP

# Build model from file
model = build_from_mps("problem.mps")

# Configure parameters
params = HPRLP_parameters()
params.stoptol = 1e-6
params.use_gpu = true
params.verbose = true

# Solve
result = optimize(model, params)

if result.status == "OPTIMAL"
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

## Example 4: Using Warm-Start

Solve related problems with warm-start.

```julia
using HPRLP
using SparseArrays

A = sparse([1.0 2.0; 3.0 1.0])
c = [-3.0, -5.0]
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]

# First solve
model = build_from_Abc(A, c, AL, AU, l, u)
params = HPRLP_parameters()
result1 = optimize(model, params)

# Solve modified problem with warm-start
AU_new = [11.0, 12.0]
model2 = build_from_Abc(A, c, AL, AU_new, l, u)
params.initial_x = result1.x
params.initial_y = result1.y
result2 = optimize(model2, params)
```

## Example 5: Auto-Save Feature

Enable auto-save for long optimizations.

```julia
using HPRLP

model = build_from_mps("large_problem.mps")

params = HPRLP_parameters()
params.time_limit = 3600
params.auto_save = true
params.save_filename = "best_solution.h5"

result = optimize(model, params)
```

## Example 6: Reading Auto-Saved Results

```julia
using HDF5

h5open("best_solution.h5", "r") do file
    x_best = read(file, "x")
    y_best = read(file, "y")
    println("Best solution found at iteration: ", read(file, "iter"))
end
```


## More Examples

*More examples will be added in future releases.*
