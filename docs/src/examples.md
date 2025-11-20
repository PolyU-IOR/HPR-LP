# Examples

This page provides complete, runnable examples demonstrating various use cases of HPRLP.

## Example 1: Basic Linear Program

Solve a simple 2-variable LP problem.

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
params.verbose = true

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("\n=== Results ===")
println("Status: ", result.output_type)
println("Objective value: ", result.primal_obj)
println("x₁ = ", result.x[1])
println("x₂ = ", result.x[2])
println("Iterations: ", result.iter)
println("Time: ", result.time, " seconds")
```

## Example 2: Solving an MPS File

Read and solve a problem from an MPS file.

```julia
using HPRLP

# Configure solver
params = HPRLP_parameters()
params.stoptol = 1e-6        # High accuracy
params.use_gpu = true        # Enable GPU
params.verbose = true        # Show progress
params.time_limit = 3600     # 1 hour limit

# Solve
result = run_single("path/to/problem.mps", params)

# Display results
if result.output_type == "OPTIMAL"
    println("✓ Optimal solution found!")
    println("  Objective: ", result.primal_obj)
    println("  Iterations: ", result.iter)
    println("  Time: ", result.time, " seconds")
    println("  Residuals: ", result.residuals)
else
    println("⚠ Solver stopped: ", result.output_type)
    println("  Best objective: ", result.primal_obj)
end
```

## Example 3: JuMP Model

Build and solve using JuMP's modeling language.

```julia
using JuMP, HPRLP

# Create model
model = Model(HPRLP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-4)
set_optimizer_attribute(model, "use_gpu", false)

# Define problem
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, -3x1 - 5x2)
@constraint(model, c1, x1 + 2x2 <= 10)
@constraint(model, c2, 3x1 + x2 <= 12)

# Solve
optimize!(model)

# Results
println("Status: ", termination_status(model))
println("Optimal value: ", objective_value(model))
println("x1 = ", value(x1))
println("x2 = ", value(x2))
println("Solve time: ", solve_time(model), " seconds")
```

## Example 4: Production Planning

A realistic production planning problem.

```julia
using JuMP, HPRLP

# Problem data
n_products = 10
n_resources = 5

# Profit per unit
profit = [50, 60, 40, 70, 55, 45, 65, 48, 52, 58]

# Resource usage (resources × products)
resource_usage = [
    2 3 1 4 2 1 3 2 1 3;
    1 2 2 1 3 2 1 2 3 1;
    3 1 2 2 1 3 2 1 2 2;
    1 1 1 2 2 1 3 1 1 2;
    2 2 3 1 1 2 1 3 2 1
]

# Available resources
capacity = [100, 80, 120, 90, 110]

# Build model
model = Model(HPRLP.Optimizer)
set_silent(model)

@variable(model, production[1:n_products] >= 0)
@objective(model, Max, sum(profit[p] * production[p] for p in 1:n_products))

for r in 1:n_resources
    @constraint(model, 
        sum(resource_usage[r,p] * production[p] for p in 1:n_products) 
        <= capacity[r]
    )
end

# Solve
optimize!(model)

# Display solution
println("=== Production Plan ===")
println("Total Profit: \$", round(objective_value(model), digits=2))
println("\nProduction Quantities:")
for p in 1:n_products
    qty = value(production[p])
    if qty > 0.01
        revenue = profit[p] * qty
        println("  Product $p: ", round(qty, digits=2), 
                " units (revenue: \$", round(revenue, digits=2), ")")
    end
end
```

## Example 5: Transportation Problem

Classical transportation problem.

```julia
using JuMP, HPRLP

# Supply at each factory
factories = 1:3
supply = [300, 400, 500]

# Demand at each warehouse
warehouses = 1:4
demand = [250, 350, 200, 400]

# Transportation costs (factories × warehouses)
cost = [
    4 6 8 5;
    3 7 9 4;
    5 5 6 7
]

# Build model
model = Model(HPRLP.Optimizer)
set_silent(model)

@variable(model, ship[f in factories, w in warehouses] >= 0)
@objective(model, Min, 
    sum(cost[f,w] * ship[f,w] for f in factories, w in warehouses)
)

# Supply constraints
for f in factories
    @constraint(model, sum(ship[f,w] for w in warehouses) <= supply[f])
end

# Demand constraints
for w in warehouses
    @constraint(model, sum(ship[f,w] for f in factories) >= demand[w])
end

# Solve
optimize!(model)

# Display solution
println("=== Transportation Plan ===")
println("Total Cost: \$", round(objective_value(model), digits=2))
println("\nShipments:")
for f in factories
    for w in warehouses
        qty = value(ship[f,w])
        if qty > 0.01
            println("  Factory $f → Warehouse $w: ", 
                    round(qty, digits=0), " units")
        end
    end
end
```

## Example 6: Portfolio Optimization

Simple portfolio selection problem.

```julia
using HPRLP
using SparseArrays
using LinearAlgebra

# Number of assets
n_assets = 50

# Expected returns
expected_returns = 0.05 .+ 0.10 * rand(n_assets)

# Constraints
A = sparse(ones(1, n_assets))  # Sum of weights
AL = [1.0]                      # = 1 (fully invested)
AU = [1.0]

# Objective: maximize expected return (min -return)
c = -expected_returns

# Variable bounds: 0 ≤ w ≤ 0.1 (max 10% per asset)
l = zeros(n_assets)
u = 0.1 * ones(n_assets)

# Solve
params = HPRLP_parameters()
params.use_gpu = false
params.verbose = false

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

# Results
weights = result.x
expected_portfolio_return = -result.primal_obj

println("=== Portfolio ===")
println("Expected Return: ", round(expected_portfolio_return * 100, digits=2), "%")
println("Number of Assets: ", count(weights .> 1e-6))
println("\nTop 5 Holdings:")
sorted_idx = sortperm(weights, rev=true)
for i in 1:min(5, n_assets)
    idx = sorted_idx[i]
    if weights[idx] > 1e-6
        println("  Asset $idx: ", round(weights[idx] * 100, digits=2), 
                "% (return: ", round(expected_returns[idx] * 100, digits=2), "%)")
    end
end
```

## Example 7: Batch Processing

Solve multiple problems in batch.

```julia
using HPRLP

function solve_batch_problems(files::Vector{String}, output_csv::String)
    # Setup
    params = HPRLP_parameters()
    params.verbose = false
    params.use_gpu = true
    params.time_limit = 600  # 10 minutes per problem
    
    # Results storage
    results = []
    
    println("Processing $(length(files)) files...")
    
    for (i, file) in enumerate(files)
        println("  [$i/$(length(files))] $file")
        
        try
            result = run_single(file, params)
            
            push!(results, (
                file = basename(file),
                status = result.output_type,
                objective = result.primal_obj,
                iterations = result.iter,
                time = result.time,
                residuals = result.residuals
            ))
            
            println("    ✓ $(result.output_type) in $(round(result.time, digits=2))s")
        catch e
            println("    ✗ ERROR: $e")
            push!(results, (
                file = basename(file),
                status = "ERROR",
                objective = NaN,
                iterations = 0,
                time = 0.0,
                residuals = NaN
            ))
        end
    end
    
    # Save results (simplified - use CSV.jl for real implementation)
    open(output_csv, "w") do io
        println(io, "File,Status,Objective,Iterations,Time,Residuals")
        for r in results
            println(io, "$(r.file),$(r.status),$(r.objective),$(r.iterations),$(r.time),$(r.residuals)")
        end
    end
    
    println("\nResults saved to: $output_csv")
    return results
end

# Usage
files = ["problem1.mps", "problem2.mps", "problem3.mps"]
results = solve_batch_problems(files, "results.csv")
```

## Example 8: Comparing Solvers

Compare HPRLP with other solvers.

```julia
using JuMP, HPRLP, HiGHS
using BenchmarkTools

function build_test_problem()
    model = Model()
    
    # Large random LP
    n = 1000
    m = 500
    
    @variable(model, x[1:n] >= 0)
    @objective(model, Min, sum(rand() * x[i] for i in 1:n))
    
    for j in 1:m
        @constraint(model, 
            sum(rand() * x[i] for i in 1:n) <= 100
        )
    end
    
    return model
end

# Test HPRLP (CPU)
model1 = build_test_problem()
set_optimizer(model1, HPRLP.Optimizer)
set_silent(model1)
set_optimizer_attribute(model1, "use_gpu", false)
@time optimize!(model1)
println("HPRLP (CPU): ", objective_value(model1))

# Test HPRLP (GPU)
model2 = build_test_problem()
set_optimizer(model2, HPRLP.Optimizer)
set_silent(model2)
set_optimizer_attribute(model2, "use_gpu", true)
@time optimize!(model2)
println("HPRLP (GPU): ", objective_value(model2))

# Test HiGHS
model3 = build_test_problem()
set_optimizer(model3, HiGHS.Optimizer)
set_silent(model3)
@time optimize!(model3)
println("HiGHS: ", objective_value(model3))
```

## Example 9: Warm-up for Performance

Demonstrate warm-up benefit.

```julia
using HPRLP
using SparseArrays

# Problem definition
A = sparse(rand(100, 200))
AL = zeros(100)
AU = 100 * ones(100)
c = rand(200)
l = zeros(200)
u = 100 * ones(200)

# Without warm-up (cold start)
params_cold = HPRLP_parameters()
params_cold.warm_up = false
params_cold.verbose = false
params_cold.use_gpu = false

println("=== Cold Start ===")
@time result1 = run_lp(A, AL, AU, c, l, u, 0.0, params_cold)
println("Solve time: ", result1.time)

# With warm-up (subsequent runs faster)
params_warm = HPRLP_parameters()
params_warm.warm_up = true
params_warm.verbose = false
params_warm.use_gpu = false

println("\n=== With Warm-up ===")
@time result2 = run_lp(A, AL, AU, c, l, u, 0.0, params_warm)
println("Solve time: ", result2.time)

# Second run (already compiled)
println("\n=== Second Run (Already Compiled) ===")
@time result3 = run_lp(A, AL, AU, c, l, u, 0.0, params_warm)
println("Solve time: ", result3.time)
```

## Running the Examples

All examples can be run directly after installing HPRLP:

```julia
using Pkg
Pkg.add("HPRLP")  # or add from GitHub

# Copy and paste any example above
# ...
```

For more examples, see the [demo](https://github.com/PolyU-IOR/HPR-LP/tree/main/demo) directory in the repository.
