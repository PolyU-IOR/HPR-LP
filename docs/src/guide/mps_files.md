# Solving MPS Files

HPRLP can directly read and solve linear programming problems in MPS (Mathematical Programming System) format, a widely-used industry standard format.

## Quick Start

```julia
using HPRLP

# Configure solver parameters
params = HPRLP_parameters()
params.stoptol = 1e-4        # Set stopping tolerance

# Solve MPS file
result = run_single("path/to/problem.mps", params)
```

## Working with MPS Files

### Basic Usage

```julia
using HPRLP

# Simple solve with defaults
params = HPRLP_parameters()
result = run_single("model.mps", params)

if result.output_type == "OPTIMAL"
    println("Found optimal solution!")
    println("Objective: ", result.primal_obj)
    println("Solution vector: ", result.x)
else
    println("Solver stopped with status: ", result.output_type)
end
```

### With Custom Parameters

```julia
params = HPRLP_parameters()

# Solver settings
params.stoptol = 1e-9          # Higher accuracy
params.time_limit = 3600       # 1 hour time limit
params.use_gpu = true          # Enable GPU
params.verbose = true          # Show progress

result = run_single("large_problem.mps", params)
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