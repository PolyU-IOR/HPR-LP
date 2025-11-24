# Solver Parameters

HPRLP provides extensive customization through the `HPRLP_parameters` type. This guide explains all available parameters and their effects on solver behavior.

## Parameter Summary Table

| Parameter | Type | Default | Range/Values | Purpose |
|-----------|------|---------|--------------|---------|
| `stoptol` | Float64 | 1e-4 | > 0 | Convergence tolerance |
| `max_iter` | Int | ~2.1B | > 0 | Maximum iterations |
| `time_limit` | Float64 | 3600.0 | > 0 | Time limit (seconds) |
| `check_iter` | Int | 150 | > 0 | Convergence check interval |
| `use_Ruiz_scaling` | Bool | true | true/false | Ruiz equilibration |
| `use_Pock_Chambolle_scaling` | Bool | true | true/false | Pock-Chambolle scaling |
| `use_bc_scaling` | Bool | true | true/false | b/c vector scaling |
| `use_gpu` | Bool | true | true/false | GPU acceleration |
| `device_number` | Int | 0 | ≥ 0 | GPU device selection |
| `warm_up` | Bool | true | true/false | GPU warm-up |
| `print_frequency` | Int | -1 | -1 or > 0 | Print interval (-1=auto) |
| `verbose` | Bool | true | true/false | Enable output |


## Creating Parameters

```julia
using HPRLP

# Create with default values
params = HPRLP_parameters()

# Customize as needed
params.stoptol = 1e-6
params.use_gpu = true
params.verbose = true
```

## Convergence Parameters

### `stoptol::Float64`
**Default:** `1e-4`

Stopping tolerance for convergence. The solver terminates when the optimality conditions are satisfied within this tolerance.

```julia
params.stoptol = 1e-9  # Higher accuracy (slower)
params.stoptol = 1e-2  # Lower accuracy (faster)
```

### `max_iter::Int`
**Default:** `typemax(Int32)` (≈2.1 billion)

Maximum number of iterations before stopping.

```julia
params.max_iter = 100000  # Limit to 100k iterations
```

### `time_limit::Float64`
**Default:** `3600.0` (1 hour)

Maximum solve time in seconds.

```julia
params.time_limit = 600.0   # 10 minutes
params.time_limit = 7200.0  # 2 hours
```

### `check_iter::Int`
**Default:** `150`

Interval for checking convergence criteria. Lower values check more frequently (higher overhead), higher values check less often (potentially more wasted iterations).

```julia
params.check_iter = 100  # Check more frequently
params.check_iter = 200  # Check less frequently
```

## Scaling Parameters

Scaling improves numerical stability and convergence. HPRLP supports three types of scaling that can be combined.

### `use_Ruiz_scaling::Bool`
**Default:** `true`

Enable Ruiz equilibration scaling, which balances the rows and columns of the constraint matrix.

```julia
params.use_Ruiz_scaling = true   # Recommended
params.use_Ruiz_scaling = false  # Disable if problem is already well-scaled
```

### `use_Pock_Chambolle_scaling::Bool`
**Default:** `true`

Enable Pock-Chambolle scaling, a diagonal scaling for primal-dual algorithms.

```julia
params.use_Pock_Chambolle_scaling = true  # Recommended
```

### `use_bc_scaling::Bool`
**Default:** `true`

Enable scaling for the objective vector (c) and constraint bounds (b).

```julia
params.use_bc_scaling = true  # Recommended for most problems
```

!!! note "Scaling Recommendations"
    - Keep all three scaling options enabled (default) for most problems
    - Disable only if you have a well-conditioned problem or specific numerical concerns
    - All three can be used simultaneously for best results

## GPU Parameters

### `use_gpu::Bool`
**Default:** `true`

Enable GPU acceleration. Requires CUDA-capable GPU.

```julia
params.use_gpu = true   # Use GPU (much faster for large problems)
params.use_gpu = false  # Use CPU only
```

!!! tip "When to Use GPU"
    - **Large problems** (>10,000 variables): Significant speedup
    - **Small problems** (<1,000 variables): CPU may be faster due to overhead
    - **Multiple GPUs available**: Use `device_number` to select specific GPU

### `device_number::Int`
**Default:** `0`

GPU device number to use (0-indexed). Only relevant when `use_gpu = true`.

```julia
params.device_number = 0  # First GPU
params.device_number = 1  # Second GPU
```

### `warm_up::Bool`
**Default:** `true`

To ensure accurate timing, due to Julia's JIT compilation, a warm-up phase can be performed before the actual solve. This is recommended for benchmarking.

```julia
params.warm_up = true   # Accurate timing (recommended)
params.warm_up = false  # Skip warm-up
```

## Output Parameters

### `verbose::Bool`
**Default:** `true`

Enable detailed solver output during optimization.

```julia
params.verbose = true   # Print iteration log
params.verbose = false  # Silent mode
```

### `print_frequency::Int`
**Default:** `-1` (automatic)

Control how often to print iteration information. When set to `-1`, frequency is automatically determined.

```julia
params.print_frequency = -1    # Auto (default)
params.print_frequency = 100   # Print every 100 iterations
params.print_frequency = 1     # Print every iteration (very verbose)
```

## Common Configurations

### High Accuracy
```julia
params = HPRLP_parameters()
params.stoptol = 1e-9
params.time_limit = 15000.0
```

### Fast Approximate Solutions
```julia
params = HPRLP_parameters()
params.stoptol = 1e-4
params.time_limit = 300.0
```

### CPU-Only (No GPU Available)
```julia
params = HPRLP_parameters()
params.use_gpu = false
```

### Production/Batch Processing
```julia
params = HPRLP_parameters()
params.verbose = false
params.warm_up = false
```

### Debugging/Analysis
```julia
params = HPRLP_parameters()
params.verbose = true
params.print_frequency = 1
params.check_iter = 1
params.max_iter = 100
```