# Solving MPS Files

HPRLP solves MPS (Mathematical Programming System) models through JuMP's file reader. This is the recommended path after removal of the old `build_from_mps` helper.

## Quick Start

```julia
using JuMP, HPRLP

# Step 1: Read model from MPS file
model = read_from_file("path/to/problem.mps")

# Step 2: Attach HPRLP
set_optimizer(model, HPRLP.Optimizer)

# Step 3: Configure solver parameters
set_attribute(model, "stoptol", 1e-4)
set_attribute(model, "use_gpu", true)

# Step 4: Optimize
optimize!(model)

# Optional: access the HPRLP-specific result struct
result = unsafe_backend(model).results
```

## Working with MPS Files

### Basic Usage

```julia
using JuMP, HPRLP

# Read the model
model = read_from_file("model.mps")

# Attach HPRLP
set_optimizer(model, HPRLP.Optimizer)

# Set up solver attributes
set_attribute(model, "stoptol", 1e-4)
set_attribute(model, "use_gpu", true)

# Solve
optimize!(model)

result = unsafe_backend(model).results

if result.status == "OPTIMAL"
    println("Found optimal solution!")
    println("Objective: ", result.primal_obj)
    println("Solution vector: ", result.x)
else
    println("Solver stopped with status: ", result.status)
end
```

### With Custom Parameters

```julia
using JuMP, HPRLP

# Read model
model = read_from_file("large_problem.mps")

# Attach HPRLP
set_optimizer(model, HPRLP.Optimizer)

# Configure attributes
set_attribute(model, "stoptol", 1e-9)      # Higher accuracy
set_attribute(model, "time_limit", 3600.0) # 1 hour time limit
set_attribute(model, "use_gpu", true)      # Enable GPU
set_attribute(model, "verbose", true)      # Show progress
set_attribute(model, "warm_up", true)      # Enable warmup for accurate timing

# Solve
optimize!(model)

result = unsafe_backend(model).results
```

!!! tip "Parameter Reference"
    For detailed explanations of all parameters, see the [Parameters](parameters.md) guide.

## Common MPS Sources

### NETLIB

The classic NETLIB LP test set:
- Download from: http://www.netlib.org/lp/data/
- Contains problems of various sizes and characteristics
- Standard benchmark for LP solvers

### MIPLIB

Mixed-Integer Programming Library (LP relaxations):
- Download from: https://miplib.zib.de/
- Includes continuous LP problems
- Challenging real-world instances

### Hans Mittelmann's Benchmark

Collection of LP and optimization problems:
- http://plato.asu.edu/ftp/lptestset/
- Regularly updated
- Various problem classes

## Performance Tips

1. **GPU vs CPU**: 
   - Use GPU for large problems (> 10,000 variables/constraints)
   - Use CPU for small to medium problems or when GPU is unavailable

2. **Tolerance**:
   - Use `1e-6` or `1e-8` for high-accuracy requirements

3. **Time Limits**:
   - Set reasonable time limits for batch processing
   - Default is 3600 seconds (1 hour)

4. **Scaling**:
   - Keep scaling enabled for better numerical stability
   - Disable only if you have pre-scaled data

## See Also

- [Parameters](parameters.md) - Complete guide to solver parameters and configuration
- [Output & Results](output_results.md) - Understanding and interpreting solver results