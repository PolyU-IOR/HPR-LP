# API Reference

## Main Functions

```@docs
run_lp
run_single
```

## Types

### Parameters

```@docs
HPRLP_parameters
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stoptol` | `Float64` | `1e-4` | Stopping tolerance for convergence |
| `max_iter` | `Int` | `typemax(Int32)` | Maximum number of iterations |
| `time_limit` | `Float64` | `3600.0` | Time limit in seconds |
| `check_iter` | `Int` | `150` | Frequency of residual checks |
| `use_Ruiz_scaling` | `Bool` | `true` | Enable Ruiz equilibration scaling |
| `use_Pock_Chambolle_scaling` | `Bool` | `true` | Enable Pock-Chambolle scaling |
| `use_bc_scaling` | `Bool` | `true` | Enable scaling for b and c vectors |
| `use_gpu` | `Bool` | `true` | Use GPU acceleration if available |
| `device_number` | `Int` | `0` | GPU device number (for multi-GPU systems) |
| `warm_up` | `Bool` | `true` | Perform warm-up run to compile code |
| `print_frequency` | `Int` | `-1` | Print frequency (-1 for automatic) |
| `verbose` | `Bool` | `true` | Print solver output |

#### Example

```julia
params = HPRLP_parameters()
params.stoptol = 1e-6       # Tighter tolerance
params.use_gpu = true       # Enable GPU
params.verbose = false      # Silent mode
params.time_limit = 3600    # 1 hour limit
```

### Results

```@docs
HPRLP_results
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `iter` | `Int` | Total number of iterations |
| `iter_4`, `iter_6`, `iter_8` | `Int` | Iterations to reach 1e-4, 1e-6, 1e-8 accuracy |
| `time` | `Float64` | Total solve time in seconds |
| `time_4`, `time_6`, `time_8` | `Float64` | Time to reach 1e-4, 1e-6, 1e-8 accuracy |
| `power_time` | `Float64` | Time spent in power method for eigenvalue estimation |
| `primal_obj` | `Float64` | Primal objective value |
| `residuals` | `Float64` | Relative residuals (primal, dual, gap) |
| `gap` | `Float64` | Duality gap |
| `output_type` | `String` | Status: "OPTIMAL", "MAX_ITER", or "TIME_LIMIT" |
| `x` | `Vector{Float64}` | Primal solution vector |
| `y` | `Vector{Float64}` | Dual solution vector (constraints) |
| `z` | `Vector{Float64}` | Dual solution vector (bounds) |

#### Example

```julia
result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Status: ", result.output_type)
println("Objective: ", result.primal_obj)
println("Solution: ", result.x)
println("Iterations: ", result.iter)
println("Time: ", result.time, " seconds")
println("Residuals: ", result.residuals)
```

## MOI/JuMP Integration

### Optimizer

```@docs
HprLP.Optimizer
```

The `Optimizer` type implements the MathOptInterface (MOI) for integration with JuMP and other optimization modeling frameworks.

### Supported Attributes

#### Optimizer Attributes

- `MOI.Silent`: Set to `true` for silent mode (equivalent to `verbose = false`)
- `MOI.TimeLimitSec`: Set time limit in seconds
- `MOI.SolveTimeSec`: Get solve time (after optimization)

#### Custom Attributes (via `MOI.RawOptimizerAttribute`)

All fields from `HPRLP_parameters` can be set as raw optimizer attributes:

```julia
using JuMP, HprLP

model = Model(HprLP.Optimizer)

# Standard MOI attributes
set_silent(model)
set_time_limit_sec(model, 3600.0)

# Custom HPRLP attributes
set_optimizer_attribute(model, "stoptol", 1e-6)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "use_Ruiz_scaling", true)
set_optimizer_attribute(model, "device_number", 0)
```

### Supported Problem Types

HPRLP supports linear programming problems with:

- **Objective**: Linear minimization or maximization
- **Variables**: Continuous with optional bounds
- **Constraints**: 
  - `MOI.EqualTo`: Equality constraints (Ax = b)
  - `MOI.LessThan`: Upper bound constraints (Ax ≤ b)
  - `MOI.GreaterThan`: Lower bound constraints (Ax ≥ b)
  - `MOI.Interval`: Two-sided constraints (l ≤ Ax ≤ u)

### Result Queries

After calling `optimize!(model)`:

```julia
# Termination status
status = termination_status(model)  # OPTIMAL, TIME_LIMIT, or ITERATION_LIMIT

# Solution
if has_values(model)
    obj_val = objective_value(model)
    x_val = value.(x)  # where x is your variable(s)
end

# Timing
solve_time = solve_time(model)
```

## Internal Functions

The following functions are exported for advanced users but are primarily used internally:

- `solve(lp, scaling_info, params)`: Main solver routine
- `formulation(lp, verbose)`: Convert QPS data to standard form
- `scaling!(lp, use_Ruiz, use_PC, use_bc)`: Apply problem scaling
- `power_iteration_gpu(...)`: Estimate maximum eigenvalue on GPU
- `power_iteration_cpu(...)`: Estimate maximum eigenvalue on CPU

## Function Reference

### `run_lp`

```julia
run_lp(A::SparseMatrixCSC,
       AL::Vector{Float64},
       AU::Vector{Float64},
       c::Vector{Float64},
       l::Vector{Float64},
       u::Vector{Float64},
       obj_constant::Float64,
       params::HPRLP_parameters) -> HPRLP_results
```

Solve a linear programming problem directly from matrix inputs.

**Arguments:**
- `A`: Sparse constraint matrix (m × n)
- `AL`: Lower bounds on constraints (m-vector)
- `AU`: Upper bounds on constraints (m-vector)
- `c`: Objective coefficients (n-vector)
- `l`: Lower bounds on variables (n-vector)
- `u`: Upper bounds on variables (n-vector)
- `obj_constant`: Constant term in objective
- `params`: Solver parameters

**Returns:** `HPRLP_results` containing solution and statistics

### `run_single`

```julia
run_single(file_name::String, 
           params::HPRLP_parameters) -> HPRLP_results
```

Solve a linear programming problem from an MPS file.

**Arguments:**
- `file_name`: Path to MPS file (must have `.mps` extension)
- `params`: Solver parameters

**Returns:** `HPRLP_results` containing solution and statistics

**Example:**
```julia
params = HPRLP_parameters()
params.use_gpu = false
params.verbose = true

result = run_single("problem.mps", params)
```
