# Direct API Usage

The direct API allows you to solve LP problems by passing matrices and vectors directly, without using MPS files or modeling languages.

## Basic Example

```julia
using HPRLP
using SparseArrays

# Problem: min -3x₁ - 5x₂
#          s.t. x₁ + 2x₂ ≤ 10
#               3x₁ + x₂ ≤ 12
#               x₁, x₂ ≥ 0

# Convert to standard form: AL ≤ Ax ≤ AU
A = sparse([-1.0 -2.0; -3.0 -1.0])  # Note the negation
AL = [-10.0, -12.0]
AU = [Inf, Inf]
c = [-3.0, -5.0]
l = [0.0, 0.0]
u = [Inf, Inf]

params = HPRLP_parameters()
params.use_gpu = false

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Optimal value: ", result.primal_obj)
println("Solution: ", result.x)
```

## Standard Form Convention

HPRLP uses the convention:

```math
\begin{array}{ll}
\min \quad & c^T x \\
\text{s.t.} \quad & AL \leq Ax \leq AU \\
& l \leq x \leq u
\end{array}
```

### Converting Common Forms

#### From ≤ Inequalities

Original: ``a^T x \leq b``

Standard form:
- Row of ``A``: ``a^T``
- ``AL_i = -\infty``
- ``AU_i = b``

Or equivalently:
- Row of ``A``: ``-a^T``
- ``AL_i = -b``
- ``AU_i = +\infty``

#### From ≥ Inequalities

Original: ``a^T x \geq b``

Standard form:
- Row of ``A``: ``a^T``
- ``AL_i = b``
- ``AU_i = +\infty``

Or equivalently:
- Row of ``A``: ``-a^T``
- ``AL_i = -\infty``
- ``AU_i = -b``

#### From Equalities

Original: ``a^T x = b``

Standard form:
- Row of ``A``: ``a^T``
- ``AL_i = b``
- ``AU_i = b``

#### Two-Sided Constraints

Original: ``l_i \leq a^T x \leq u_i``

Standard form:
- Row of ``A``: ``a^T``
- ``AL_i = l_i``
- ``AU_i = u_i``

## Complete Example with All Constraint Types

```julia
using HPRLP
using SparseArrays

# Problem with mixed constraints:
# min   x₁ + 2x₂ + 3x₃
# s.t.  x₁ + x₂ + x₃ = 5      (equality)
#       x₁ + 2x₂ ≤ 8          (upper bound)
#       2x₁ + x₃ ≥ 3          (lower bound)
#       1 ≤ x₂ + x₃ ≤ 6       (two-sided)
#       0 ≤ x₁ ≤ 5, x₂ ≥ 0, x₃ free

# Constraint matrix
A = sparse([
    1.0  1.0  1.0;   # x₁ + x₂ + x₃ = 5
    1.0  2.0  0.0;   # x₁ + 2x₂ ≤ 8
    2.0  0.0  1.0;   # 2x₁ + x₃ ≥ 3
    0.0  1.0  1.0    # 1 ≤ x₂ + x₃ ≤ 6
])

# Constraint bounds
AL = [5.0, -Inf, 3.0, 1.0]
AU = [5.0, 8.0, Inf, 6.0]

# Objective
c = [1.0, 2.0, 3.0]

# Variable bounds (free variables: l = -Inf, u = Inf)
l = [0.0, 0.0, -Inf]
u = [5.0, Inf, Inf]

params = HPRLP_parameters()
params.use_gpu = false
params.verbose = true

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("\nResults:")
println("Status: ", result.output_type)
println("Objective: ", result.primal_obj)
println("Solution: x = ", result.x)
```

## Working with Dense Matrices

If your problem uses dense matrices, convert to sparse:

```julia
using SparseArrays

# Dense matrix
A_dense = [1.0 2.0 3.0;
           4.0 5.0 6.0;
           7.0 8.0 9.0]

# Convert to sparse
A_sparse = sparse(A_dense)

# Then solve as usual
result = run_lp(A_sparse, AL, AU, c, l, u, 0.0, params)
```

## Constructing Sparse Matrices

### From Triplet Format

```julia
# Row indices, column indices, values
rows = [1, 1, 2, 2, 3]
cols = [1, 2, 2, 3, 3]
vals = [1.0, 2.0, 3.0, 4.0, 5.0]

# Create sparse matrix
A = sparse(rows, cols, vals, 3, 3)  # 3×3 matrix
```

### Pattern Matrices

```julia
using SparseArrays

# Identity matrix
n = 100
A = sparse(I, n, n)

# Tridiagonal matrix
A = spdiagm(-1 => ones(n-1), 0 => 2*ones(n), 1 => ones(n-1))

# Random sparse matrix
A = sprand(1000, 500, 0.01)  # 1000×500, 1% density
```

## GPU Acceleration

For large problems, enable GPU acceleration:

```julia
params = HPRLP_parameters()
params.use_gpu = true
params.device_number = 0  # First GPU

# The solver automatically transfers data to GPU
result = run_lp(A, AL, AU, c, l, u, 0.0, params)
```

When to use GPU:
- Problem has > 10,000 variables or constraints
- Constraint matrix has > 100,000 nonzeros
- You have a CUDA-capable GPU available

## Parameter Tuning

### Convergence Tolerance

```julia
params = HPRLP_parameters()

# Quick approximate solution
params.stoptol = 1e-3

# High accuracy solution
params.stoptol = 1e-8

result = run_lp(A, AL, AU, c, l, u, 0.0, params)
```

### Scaling Options

```julia
params = HPRLP_parameters()

# Disable all scaling (if data is already well-scaled)
params.use_Ruiz_scaling = false
params.use_Pock_Chambolle_scaling = false
params.use_bc_scaling = false

# Or enable all (default, recommended)
params.use_Ruiz_scaling = true
params.use_Pock_Chambolle_scaling = true
params.use_bc_scaling = true
```

### Iteration and Time Limits

```julia
params = HPRLP_parameters()
params.max_iter = 100000       # Maximum iterations
params.time_limit = 1800       # 30 minutes
params.check_iter = 100        # Check convergence every 100 iterations
```

## Accessing Solution Information

```julia
result = run_lp(A, AL, AU, c, l, u, 0.0, params)

# Optimization status
if result.output_type == "OPTIMAL"
    println("Found optimal solution!")
    
    # Objective value
    obj = result.primal_obj
    
    # Primal solution
    x = result.x
    
    # Dual solutions
    y = result.y  # Constraint duals
    z = result.z  # Variable bound duals
    
    # Performance metrics
    println("Iterations: ", result.iter)
    println("Time: ", result.time, " seconds")
    println("Residuals: ", result.residuals)
    println("Gap: ", result.gap)
    
elseif result.output_type == "TIME_LIMIT"
    println("Time limit reached")
    println("Best objective found: ", result.primal_obj)
    
elseif result.output_type == "MAX_ITER"
    println("Iteration limit reached")
    println("Current objective: ", result.primal_obj)
end
```

## Accuracy Tracking

The solver tracks when different accuracy levels are reached:

```julia
result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Time to 1e-4 accuracy: ", result.time_4, " seconds")
println("Iterations to 1e-4: ", result.iter_4)

println("Time to 1e-6 accuracy: ", result.time_6, " seconds")
println("Iterations to 1e-6: ", result.iter_6)

println("Time to 1e-8 accuracy: ", result.time_8, " seconds")
println("Iterations to 1e-8: ", result.iter_8)
```

## Silent Mode

Suppress all output:

```julia
params = HPRLP_parameters()
params.verbose = false
params.warm_up = false

result = run_lp(A, AL, AU, c, l, u, 0.0, params)
# No output, only returns result
```

## Example: Portfolio Optimization

```julia
using HPRLP
using SparseArrays

# Simple portfolio optimization
# min -μᵀw (negative expected return)
# s.t. Σᵢ wᵢ = 1  (fully invested)
#      wᵢ ≥ 0    (long-only)
#      wᵢ ≤ 0.2  (max 20% per asset)

n_assets = 100
expected_returns = rand(n_assets) .* 0.1  # 0-10% expected return

# Constraints: sum(w) = 1
A = sparse(ones(1, n_assets))
AL = [1.0]
AU = [1.0]

# Objective: minimize -μᵀw (i.e., maximize μᵀw)
c = -expected_returns

# Variable bounds: 0 ≤ w ≤ 0.2
l = zeros(n_assets)
u = 0.2 * ones(n_assets)

params = HPRLP_parameters()
params.use_gpu = false
params.verbose = false

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Expected portfolio return: ", -result.primal_obj)
println("Weights: ", result.x)
println("Number of positions: ", count(result.x .> 1e-6))
```
