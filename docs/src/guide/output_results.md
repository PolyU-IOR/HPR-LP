# Output & Results

This guide explains how to interpret the results returned by HPRLP solvers and understand the termination status.

## Result Fields Summary

All HPRLP solving functions return an `HPRLP_results` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `output_type` | String | Termination status: `"OPTIMAL"`, `"MAX_ITER"`, or `"TIME_LIMIT"` |
| `x` | Vector{Float64} | Primal solution (decision variables) |
| `y` | Vector{Float64} | Dual variables for constraints |
| `z` | Vector{Float64} | Dual variables for bounds |
| `primal_obj` | Float64 | Primal objective value (``c^T x + c_0``) |
| `iter` | Int | Total number of iterations |
| `time` | Float64 | Total solve time (seconds) |
| `iter_4`, `time_4` | Int, Float64 | Iterations/time to reach 1e-4 accuracy |
| `iter_6`, `time_6` | Int, Float64 | Iterations/time to reach 1e-6 accuracy |
| `iter_8`, `time_8` | Int, Float64 | Iterations/time to reach 1e-8 accuracy |
| `residuals` | Float64 | Combined measure of constraint violations and objective gap |
| `gap` | Float64 | Objective gap between primal and dual |

## Basic Usage

```julia
result = run_lp(A, AL, AU, c, l, u, c0, params)

# Access results
println("Status: ", result.output_type)
println("Objective: ", result.primal_obj)
println("Solution: ", result.x)
println("Time: ", result.time, " seconds")
```

## Result Fields

### Termination Status

#### `output_type::String`
Indicates why the solver stopped:

- **`"OPTIMAL"`** - Successfully found an optimal solution within tolerance
- **`"MAX_ITER"`** - Reached maximum iteration limit before converging
- **`"TIME_LIMIT"`** - Reached time limit before converging

```julia
if result.output_type == "OPTIMAL"
    println("✓ Optimal solution found!")
    println("Objective value: ", result.primal_obj)
elseif result.output_type == "MAX_ITER"
    println("⚠ Iteration limit reached")
    println("Best objective: ", result.primal_obj)
    println("Residual: ", result.residuals)
elseif result.output_type == "TIME_LIMIT"
    println("⚠ Time limit reached")
    println("Best objective: ", result.primal_obj)
end
```

### Solution Vectors

#### `x::Vector{Float64}`
The primal solution vector (decision variables).

```julia
# Access solution values
println("x₁ = ", result.x[1])
println("x₂ = ", result.x[2])

# Use in calculations
optimal_cost = sum(c .* result.x)
```

#### `y::Vector{Float64}`
The dual variables for constraints.

#### `z::Vector{Float64}`
The dual variables for bound constraints (reduced costs).

### Objective Values

#### `primal_obj::Float64`
The primal objective value: $c^T x + c_0$

```julia
println("Optimal cost: ", result.primal_obj)
```

### Performance Metrics

#### `iter::Int`
Total number of iterations performed.

```julia
println("Solved in ", result.iter, " iterations")
```

#### `time::Float64`
Total solve time in seconds (excluding setup and scaling).

```julia
println("Solve time: ", result.time, " seconds")
```

#### Accuracy Milestones

HPRLP tracks when certain accuracy levels are reached:

- `iter_4`, `time_4` - Iterations/time to reach 1e-4 accuracy
- `iter_6`, `time_6` - Iterations/time to reach 1e-6 accuracy  
- `iter_8`, `time_8` - Iterations/time to reach 1e-8 accuracy

```julia
if result.time_6 < params.time_limit
    println("Reached 1e-6 accuracy in ", result.iter_6, " iterations")
    println("Time to 1e-6: ", result.time_6, " seconds")
end
```

### Quality Metrics

#### `residuals::Float64`
Combined measure of constraint violations and objective gap. Lower is better.

```julia
println("Final residual: ", result.residuals)
```

#### `gap::Float64`
Objective gap between primal and dual solutions.

```julia
println("Objective gap: ", result.gap)
```

## Checking Solution Quality

### Optimal Solutions

```julia
function check_solution_quality(result, params)
    if result.output_type != "OPTIMAL"
        @warn "Solution not optimal: $(result.output_type)"
    end
    
    if result.residuals > params.stoptol
        @warn "Residuals exceed tolerance: $(result.residuals) > $(params.stoptol)"
    end
    
    if result.gap > 1e-3
        @warn "Large objective gap: $(result.gap)"
    end
    
    return result.output_type == "OPTIMAL" && 
           result.residuals <= params.stoptol
end
```

### Verifying Feasibility

```julia
function verify_primal_feasibility(result, A, AL, AU, l, u)
    x = result.x
    
    # Check variable bounds
    if any(x .< l .- 1e-6) || any(x .> u .+ 1e-6)
        @warn "Variable bounds violated"
        return false
    end
    
    # Check constraint bounds
    Ax = A * x
    if any(Ax .< AL .- 1e-6) || any(Ax .> AU .+ 1e-6)
        @warn "Constraint bounds violated"
        return false
    end
    
    println("✓ Solution is feasible")
    return true
end
```

## Examples

### Basic Usage

```julia
using HPRLP

params = HPRLP_parameters()
result = run_single("problem.mps", params)

println("═══════ Solution Summary ═══════")
println("Status:     ", result.output_type)
println("Objective:  ", result.primal_obj)
println("Iterations: ", result.iter)
println("Time:       ", round(result.time, digits=3), " sec")
println("Residual:   ", result.residuals)
println("════════════════════════════════")
```

### Detailed Analysis

```julia
function analyze_results(result, params)
    println("\n" * "="^50)
    println("HPRLP Solution Analysis")
    println("="^50)
    
    # Termination status
    println("\n[Termination]")
    println("  Status: ", result.output_type)
    
    # Objective information
    println("\n[Objective]")
    println("  Primal: ", result.primal_obj)
    println("  Gap:    ", result.gap)
    
    # Performance
    println("\n[Performance]")
    println("  Iterations: ", result.iter)
    println("  Time:       ", round(result.time, digits=3), " seconds")
    
    # Accuracy milestones
    println("\n[Accuracy Milestones]")
    if result.iter_4 > 0
        println("  1e-4: ", result.iter_4, " iterations, ", 
                round(result.time_4, digits=3), " sec")
    end
    if result.iter_6 > 0
        println("  1e-6: ", result.iter_6, " iterations, ", 
                round(result.time_6, digits=3), " sec")
    end
    if result.iter_8 > 0
        println("  1e-8: ", result.iter_8, " iterations, ", 
                round(result.time_8, digits=3), " sec")
    end
    
    # Quality assessment
    println("\n[Solution Quality]")
    println("  Residual:  ", result.residuals)
    println("  Tolerance: ", params.stoptol)
    
    quality = if result.residuals < params.stoptol / 10
        "Excellent"
    elseif result.residuals < params.stoptol
        "Good"
    elseif result.residuals < params.stoptol * 10
        "Acceptable"
    else
        "Poor"
    end
    println("  Quality:   ", quality)
    
    println("="^50 * "\n")
end

# Usage
result = run_single("model.mps", params)
analyze_results(result, params)
```

### Comparing Solutions

```julia
function compare_results(result1, result2, label1="Result 1", label2="Result 2")
    println("\n", "="^60)
    println("Solution Comparison")
    println("="^60)
    
    println("\n", rpad("Metric", 20), rpad(label1, 20), label2)
    println("-"^60)
    
    println(rpad("Status", 20), 
            rpad(result1.output_type, 20), result2.output_type)
    println(rpad("Objective", 20), 
            rpad(string(round(result1.primal_obj, digits=6)), 20),
            round(result2.primal_obj, digits=6))
    println(rpad("Iterations", 20), 
            rpad(string(result1.iter), 20), result2.iter)
    println(rpad("Time (sec)", 20), 
            rpad(string(round(result1.time, digits=3)), 20),
            round(result2.time, digits=3))
    println(rpad("Residual", 20), 
            rpad(string(result1.residuals), 20), result2.residuals)
    
    println("="^60 * "\n")
end

# Usage: Compare GPU vs CPU
params_gpu = HPRLP_parameters()
params_gpu.use_gpu = true

params_cpu = HPRLP_parameters()
params_cpu.use_gpu = false

result_gpu = run_single("model.mps", params_gpu)
result_cpu = run_single("model.mps", params_cpu)

compare_results(result_gpu, result_cpu, "GPU", "CPU")
```

## Common Issues

### Non-Optimal Termination

If solver stops with `MAX_ITER` or `TIME_LIMIT`:

```julia
if result.output_type != "OPTIMAL"
    # Check if close to optimal
    if result.residuals < 1e-3
        println("Near-optimal solution found")
        println("Consider increasing time_limit or max_iter")
    else
        println("Poor solution quality")
        println("Problem may be ill-conditioned")
        println("Try adjusting scaling parameters")
    end
end
```

### Large Residuals

```julia
if result.residuals > params.stoptol * 100
    @warn "Very large residuals - possible issues:"
    println("  - Problem may be infeasible or unbounded")
    println("  - Numerical scaling issues")
    println("  - Try enabling all scaling options")
    println("  - Check problem formulation")
end
```
