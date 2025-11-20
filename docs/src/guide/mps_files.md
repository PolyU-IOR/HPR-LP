# Solving MPS Files

HPRLP can directly read and solve linear programming problems in MPS (Mathematical Programming System) format, a widely-used industry standard format.

## Quick Start

```julia
using HPRLP

# Configure solver parameters
params = HPRLP_parameters()
params.use_gpu = false
params.stoptol = 1e-4
params.verbose = true

# Solve MPS file
result = run_single("path/to/problem.mps", params)

# Access results
println("Objective value: ", result.primal_obj)
println("Status: ", result.output_type)
println("Solve time: ", result.time, " seconds")
```

## MPS Format Overview

The MPS format defines an LP problem with these sections:

- **NAME**: Problem name
- **ROWS**: Constraint types (N=objective, E=equality, L=≤, G=≥)
- **COLUMNS**: Nonzero entries in constraint matrix
- **RHS**: Right-hand side values
- **BOUNDS**: Variable bounds
- **ENDATA**: End marker

### Example MPS File

```mps
NAME          EXAMPLE
ROWS
 N  OBJ
 L  ROW1
 L  ROW2
COLUMNS
    X1        ROW1      1.0
    X1        ROW2      3.0
    X1        OBJ      -3.0
    X2        ROW1      2.0
    X2        ROW2      1.0
    X2        OBJ      -5.0
RHS
    RHS1      ROW1     10.0
    RHS1      ROW2     12.0
BOUNDS
 LO BND1      X1        0.0
 LO BND1      X2        0.0
ENDATA
```

This represents:
```math
\begin{array}{ll}
\min \quad & -3x_1 - 5x_2 \\
\text{s.t.} \quad & x_1 + 2x_2 \leq 10 \\
& 3x_1 + x_2 \leq 12 \\
& x_1, x_2 \geq 0
\end{array}
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
params.stoptol = 1e-6          # Higher accuracy
params.time_limit = 7200       # 2 hour time limit
params.max_iter = 1000000      # Maximum iterations

# Performance settings
params.use_gpu = true          # Enable GPU
params.device_number = 0       # Use first GPU

# Scaling options
params.use_Ruiz_scaling = true
params.use_Pock_Chambolle_scaling = true
params.use_bc_scaling = true

# Output control
params.verbose = true          # Show progress
params.print_frequency = 100   # Print every 100 iterations

result = run_single("large_problem.mps", params)
```

### Silent Mode

For batch processing or when embedding in larger applications:

```julia
params = HPRLP_parameters()
params.verbose = false  # No output
params.warm_up = false  # Skip warm-up for speed

result = run_single("problem.mps", params)
```

## Warm-up Phase

For better performance when solving multiple similar problems:

```julia
params = HPRLP_parameters()
params.warm_up = true  # Enable warm-up (default)

# First call: includes warm-up compilation
result1 = run_single("problem1.mps", params)

# Subsequent calls: already compiled, faster
result2 = run_single("problem2.mps", params)
result3 = run_single("problem3.mps", params)
```

## Batch Processing

Process multiple MPS files:

```julia
using HPRLP

function solve_batch(files::Vector{String}, params::HPRLP_parameters)
    results = Dict{String, HprLP_results}()
    
    for (i, file) in enumerate(files)
        println("Solving $i/$(length(files)): $file")
        
        try
            result = run_single(file, params)
            results[file] = result
            
            println("  Status: $(result.output_type)")
            println("  Objective: $(result.primal_obj)")
            println("  Time: $(result.time)s")
        catch e
            println("  ERROR: $e")
        end
    end
    
    return results
end

# Usage
params = HPRLP_parameters()
params.verbose = false
params.time_limit = 600  # 10 minutes per problem

files = ["problem1.mps", "problem2.mps", "problem3.mps"]
results = solve_batch(files, params)
```

## Checking Problem Statistics

The solver prints problem information when `verbose = true`:

```
READING FILE ... problem.mps
READING FILE time: 0.15 seconds
FORMULATING LP ...
problem information: nRow = 1000, nCol = 2000, nnz A = 10000
                     number of equalities = 100
                     number of inequalities = 900
FORMULATING LP time: 0.05 seconds
```

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
   - Default `stoptol = 1e-4` is good for most applications
   - Use `1e-6` or `1e-8` for high-accuracy requirements
   - Lower tolerances significantly increase solve time

3. **Time Limits**:
   - Set reasonable time limits for batch processing
   - Default is 3600 seconds (1 hour)

4. **Scaling**:
   - Keep scaling enabled for better numerical stability
   - Disable only if you have pre-scaled data

## Troubleshooting

### File Not Found

```julia
# Check file path
if !isfile("problem.mps")
    error("File not found!")
end

result = run_single("problem.mps", params)
```

### Time Limit Reached

```julia
result = run_single("problem.mps", params)

if result.output_type == "TIME_LIMIT"
    println("Time limit reached after $(result.time) seconds")
    println("Current objective: $(result.primal_obj)")
    println("Residuals: $(result.residuals)")
    
    # Could increase time limit and retry
    params.time_limit = 7200
    result = run_single("problem.mps", params)
end
```

### Iteration Limit

```julia
if result.output_type == "MAX_ITER"
    println("Maximum iterations reached")
    println("Increase max_iter or check problem formulation")
end
```
