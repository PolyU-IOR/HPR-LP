# Getting Started

## Prerequisites

Before using HPRLP, ensure you have:

- **Julia** (version 1.10 or higher recommended)
- **CUDA** (optional, for GPU acceleration)
  - CUDA-capable GPU with compute capability 5.0 or higher
  - CUDA Toolkit 11.0 or higher

## Installation

### From Julia General Registry (after registration)

```julia
using Pkg
Pkg.add("HPRLP")
```

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/HPR-LP")
```

### Development Installation

To install for development:

```bash
git clone https://github.com/PolyU-IOR/HPR-LP.git
cd HPR-LP
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verifying CUDA Installation

If you plan to use GPU acceleration, verify CUDA is working:

```julia
using CUDA
CUDA.versioninfo()
```

If CUDA is not available, HPRLP will automatically fall back to CPU mode.

## First Example: Solving a Simple LP

Let's solve a basic linear programming problem:

```math
\begin{array}{ll}
\min \quad & -3x_1 - 5x_2 \\
\text{s.t.} \quad & x_1 + 2x_2 \leq 10 \\
& 3x_1 + x_2 \leq 12 \\
& x_1, x_2 \geq 0
\end{array}
```

### Using the Direct API

```julia
using HPRLP
using SparseArrays

# Convert to standard form: AL ≤ Ax ≤ AU
# Inequality ax + by ≤ c becomes -ax - by ≥ -c (i.e., AL = -c, AU = ∞)
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
c = [-3.0, -5.0]
l = [0.0, 0.0]
u = [Inf, Inf]
obj_constant = 0.0

# Set up parameters
params = HPRLP_parameters()
params.use_gpu = false      # Use CPU for this small problem
params.stoptol = 1e-4      # Convergence tolerance
params.time_limit = 60     # Maximum 60 seconds
params.verbose = true      # Print solver output

# Solve
result = run_lp(A, AL, AU, c, l, u, obj_constant, params)

# Check results
println("Status: ", result.output_type)
println("Optimal value: ", result.primal_obj)
println("Solution: x₁ = ", result.x[1], ", x₂ = ", result.x[2])
println("Iterations: ", result.iter)
println("Solve time: ", result.time, " seconds")
```

**Expected output:**
- Optimal value: -26.4
- Solution: x₁ ≈ 2.8, x₂ ≈ 3.6

### Using JuMP

The same problem using the JuMP modeling interface:

```julia
using JuMP
using HPRLP

model = Model(HPRLP.Optimizer)

# Set optimizer attributes
set_optimizer_attribute(model, "stoptol", 1e-4)
set_optimizer_attribute(model, "use_gpu", false)
set_optimizer_attribute(model, "verbose", true)

# Define variables and constraints
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, -3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

# Solve
optimize!(model)

# Get results
println("Status: ", termination_status(model))
println("Optimal value: ", objective_value(model))
println("Solution: x₁ = ", value(x1), ", x₂ = ", value(x2))
println("Solve time: ", solve_time(model), " seconds")
```

## Next Steps

- Learn about [solving MPS files](guide/mps_files.md)
- Explore the [Direct API](guide/direct_api.md) in detail
- See more [JuMP integration examples](guide/jump_integration.md)
- Check out the full [API Reference](api.md)
- Browse additional [Examples](examples.md)

## Performance Tips

1. **JIT Compilation**: The first run will be slow due to Julia's JIT compilation. For benchmarking, run the solver twice or use a warm-up phase.

2. **GPU Usage**: For large problems (typically > 10,000 variables/constraints), GPU acceleration can provide significant speedups.

3. **Scaling**: The default scaling methods (Ruiz and Pock-Chambolle) improve numerical stability. Keep them enabled unless you have specific reasons to disable them.

4. **Tolerance**: The default `stoptol = 1e-4` balances accuracy and speed. Increase for tighter solutions or decrease for faster approximate solutions.
