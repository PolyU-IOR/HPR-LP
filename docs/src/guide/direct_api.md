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

params = HPRLP_parameters()
params.use_gpu = false
params.verbose = true

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("\nResults:")
println("Status: ", result.status)
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

## See Also

- [Parameters](parameters.md) - Complete guide to solver parameters
- [Output & Results](output_results.md) - Understanding solver output and results