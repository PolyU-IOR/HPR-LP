# API Reference

Complete API documentation for HPRLP. For detailed guides, see:
- [Parameters Guide](guide/parameters.md) - Detailed parameter explanations
- [Output & Results](guide/output_results.md) - Understanding solver output

## Main Functions

```@docs
run_lp
run_single
HPRLP.Optimizer
```

## Types

```@docs
HPRLP_parameters
HPRLP_results
```

## Quick Reference

### Solving Problems

**Direct API:**
```julia
result = run_lp(A, AL, AU, c, l, u, c0, params)
```

**MPS Files:**
```julia
result = run_single("problem.mps", params)
```

**JuMP:**
```julia
model = Model(HPRLP.Optimizer)
optimize!(model)
```

### Common Parameter Settings

```julia
params = HPRLP_parameters()
params.stoptol = 1e-6      # Convergence tolerance
params.use_gpu = true      # Enable GPU
params.verbose = false     # Silent mode
params.time_limit = 3600   # Time limit (seconds)
```

### Accessing Results

```julia
result.output_type    # "OPTIMAL", "MAX_ITER", or "TIME_LIMIT"
result.primal_obj     # Objective value
result.x              # Solution vector
result.iter           # Iterations
result.time           # Solve time (seconds)
result.residuals      # Solution quality
```

## See Also

- [User Guide](guide/input_overview.md) - Comprehensive usage guides
- [Examples](examples.md) - Complete working examples
