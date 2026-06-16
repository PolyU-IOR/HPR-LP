# GPUPresolver

`GPUPresolver` is a Julia package for **LP/QP presolve and postsolve**.
It reads an optimization model, applies presolve reductions, returns a reduced
problem, and can recover original-space solutions through postsolve.

This repository is organized as a **library first**. The main entrypoint is the
Julia API.

## What The Package Does

- reads LP/QP models from `.mps` / `.mps.gz` files or from in-memory Julia data structures
- runs presolve with the GPU-native backend
- returns a reduced problem
- restores primal/dual variables after the reduced problem is solved

## Problem Types

The package currently supports two problem classes:

- `LP` (linear programming)
- `QP` (quadratic programming)

### LP form

LP models are interpreted as:

$$
\begin{aligned}
\min_x \quad & c^\top x + c_0 \\
\text{subject to} \quad & AL \le Ax \le AU \\
& l \le x \le u
\end{aligned}
$$

where:

- `A` is the constraint matrix
- `c` is the linear objective vector
- `AL`, `AU` are row lower and upper bounds
- `l`, `u` are variable lower and upper bounds

### QP form

QP models are interpreted as:

$$
\begin{aligned}
\min_x \quad & \frac{1}{2}x^\top Qx + c^\top x + c_0 \\
\text{subject to} \quad & AL \le Ax \le AU \\
& l \le x \le u
\end{aligned}
$$

where `Q` is the quadratic objective matrix.

If input `Q` is not symmetric, presolve uses the symmetric part:

$$
Q \leftarrow \tfrac{1}{2}(Q + Q^\top)
$$

## Install

From Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/GPU-Presolver.git", rev="main")
```

Then:

```julia
using GPUPresolver
```

## Quick Start

```julia
using GPUPresolver

setup = GPUPresolver.load_default_presolve_setup()

result = GPUPresolver.run_presolve(
    "/path/to/model.mps.gz";   # or "/path/to/model_qp.mps.gz"
    problem_type=setup.problem_type,
    config=setup.config,
)

println(result.status)
println(result.presolve_time)

reduced_problem = result.reduced_problem
```

## Package API

The main API is:

```julia
run_presolve(...)
```

It returns a `GPUPresolver.PresolveResult` with:

- `status`
- `presolve_time`
- `backend`
- `reduced_problem`
- `state`

Use `state` later with:

```julia
run_postsolve(...)
```

## Postsolve Example

```julia
using GPUPresolver

result = GPUPresolver.run_presolve(
    "/path/to/model.mps";
    problem_type="LP",
    config=GPUPresolver.PresolveConfig(backend="GPU", verbose=false, device_number=0),
)

# Solve the reduced problem yourself and prepare:
# x_red, y_red, z_red

if result.status == "OK" && result.state !== nothing
    x_org, y_org, z_org = GPUPresolver.run_postsolve(result.state, x_red, y_red, z_red)
end
```

## Configuration

The package ships with a default configuration file:

- `config/default.toml`

This file is intended to describe package runtime and presolve behavior, for
example:

- problem type
- backend selection
- device number
- verbosity
- scheduler mode
- iteration/time limits
- tolerances
- rule switches
- QP-specific controls

The default configuration currently uses the tiered GPU scheduler with
structural L1 substitution enabled in GPU-only mode. FME projection remains
disabled by default.

It intentionally does **not** describe dataset paths or experiment output
locations. Those belong to the calling code.

You can also load the same TOML file through the Julia API:

```julia
using GPUPresolver

setup = GPUPresolver.load_default_presolve_setup()

result = GPUPresolver.run_presolve(
    "/path/to/model.mps.gz";
    problem_type=setup.problem_type,
    config=setup.config,
)
```

This returns a ready-to-use setup object with:

- `setup.problem_type`
- `setup.config`
- `setup.presolve_params`
- `setup.rule_switches`

## Notes

- File-based and in-memory LP/QP workflows are both supported.
