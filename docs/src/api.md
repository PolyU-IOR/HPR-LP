# API Reference

Complete API documentation for HPRLP. For detailed guides, see:
- [Parameters Guide](guide/parameters.md) - Detailed parameter explanations
- [Output & Results](guide/output_results.md) - Understanding solver output

## Main Functions

```@docs
build_from_mps
build_from_Abc
optimize
HPRLP.Optimizer
```

## Types

```@docs
HPRLP_parameters
HPRLP_results
```

## Quick Reference

### Solving Problems

**Direct API (Matrix Form):**
```julia
# Step 1: Build the model
model = build_from_Abc(A, c, AL, AU, l, u)

# Step 2: Set parameters
params = HPRLP_parameters()
params.stoptol = 1e-6

# Step 3: Optimize
result = optimize(model, params)
```

**MPS Files:**
```julia
# Step 1: Build the model from file
model = build_from_mps("problem.mps")

# Step 2: Set parameters
params = HPRLP_parameters()

# Step 3: Optimize
result = optimize(model, params)
```

**JuMP:**
```julia
model = Model(HPRLP.Optimizer)
# ... add variables and constraints ...
optimize!(model)
```

### Common Parameter Settings

```julia
params = HPRLP_parameters()
params.stoptol = 1e-6           # Convergence tolerance
params.use_gpu = true           # Enable GPU
params.verbose = false          # Silent mode
params.time_limit = 3600        # Time limit (seconds)
params.warm_up = true           # Enable warmup for accurate timing
params.initial_x = x0           # Initial primal solution
params.initial_y = y0           # Initial dual solution
params.auto_save = true         # Auto-save best solution
params.save_filename = "opt.h5" # HDF5 file for auto-save
```

### Accessing Results

```julia
result.status         # "OPTIMAL", "MAX_ITER", or "TIME_LIMIT"
result.primal_obj     # Primal objective value
result.x              # Primal solution vector
result.y              # Dual solution vector (constraints)
result.z              # Dual solution vector (bounds)
result.iter           # Total iterations
result.time           # Solve time (seconds)
result.residuals      # Final residual (max of primal, dual, gap)
```

## See Also

- [User Guide](guide/input_overview.md) - Comprehensive usage guides
- [Examples](examples.md) - Complete working examples
