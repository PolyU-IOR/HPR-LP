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

# Step 1: Build the model
model = build_from_Abc(A, c, AL, AU, l, u)

# Step 2: Set parameters
params = HPRLP_parameters()
params.use_gpu = false

# Step 3: Optimize
result = optimize(model, params)

println("Status: ", result.status)
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

| Original Constraint | Set ``A`` row | Set ``AL_i`` | Set ``AU_i`` |
|---------------------|---------------|--------------|--------------|
| ``a^T x \leq b`` | ``a^T`` | ``-\infty`` | ``b`` |
| ``a^T x \geq b`` | ``a^T`` | ``b`` | ``+\infty`` |
| ``a^T x = b`` | ``a^T`` | ``b`` | ``b`` |
| ``L \leq a^T x \leq U`` | ``a^T`` | ``L`` | ``U`` |

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

# Build and solve
model = build_from_Abc(A, c, AL, AU, l, u)

params = HPRLP_parameters()
params.use_gpu = false
params.verbose = true

result = optimize(model, params)

println("\nResults:")
println("Status: ", result.status)
println("Objective: ", result.primal_obj)
println("Solution: x = ", result.x)
```

## Working with Dense Matrices

Dense matrices are automatically converted to sparse format:

```julia
using SparseArrays

# Dense matrix (will be converted automatically with a warning)
A_dense = [1.0 2.0 3.0;
           4.0 5.0 6.0;
           7.0 8.0 9.0]

# build_from_Abc will convert it to sparse automatically
model = build_from_Abc(A_dense, c, AL, AU, l, u)

# Then solve as usual
params = HPRLP_parameters()
result = optimize(model, params)
```

## Warm-Start with Initial Solutions

Provide initial primal/dual solutions to warm-start the solver:

```julia
# From previous solve or heuristic
x0 = [1.0, 2.0]
y0 = [0.5, 0.3]

params = HPRLP_parameters()
params.initial_x = x0
params.initial_y = y0  # Optional

model = build_from_Abc(A, c, AL, AU, l, u)
result = optimize(model, params)
```

Useful for solving sequences of related problems or re-solving with different parameters.

## Auto-Save Best Solution

Automatically save the best solution found during optimization to HDF5:

```julia
params = HPRLP_parameters()
params.auto_save = true
params.save_filename = "my_optimization.h5"

result = optimize(model, params)
```

Useful for long optimizations that might be interrupted or reach time limits.
```

## See Also

- [Parameters](parameters.md) - Complete guide to solver parameters
- [Output & Results](output_results.md) - Understanding solver output and results