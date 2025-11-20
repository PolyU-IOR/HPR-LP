# HPRLP.jl Documentation

**A GPU-accelerated Julia package for solving Linear Programming problems using the Halpern Peaceman-Rachford (HPR) method.**

## Overview

HPRLP.jl is a high-performance linear programming solver that leverages GPU acceleration to solve large-scale LP problems efficiently. It implements the Halpern Peaceman-Rachford splitting method with adaptive restart strategies.

## Features

- ✅ **GPU Acceleration**: Native CUDA support for solving large-scale problems
- ✅ **CPU Fallback**: Automatic CPU mode when GPU is not available
- ✅ **Multiple Interfaces**: 
  - Direct API with matrix inputs
  - MPS file format support
  - JuMP integration via MOI wrapper
- ✅ **Flexible Scaling**: Ruiz and Pock-Chambolle scaling methods
- ✅ **Adaptive Algorithms**: Automatic restart strategies and step-size selection
- ✅ **Silent Mode**: Control output verbosity for embedding in larger applications

## Problem Formulation

HPRLP solves linear programming problems of the form:

```math
\begin{array}{ll}
\min_{x \in \mathbb{R}^n} \quad & \langle c, x \rangle \\
\text{subject to} \quad & L \leq A x \leq U, \\
& l \leq x \leq u
\end{array}
```

where:
- ``x \in \mathbb{R}^n`` is the decision variable
- ``c \in \mathbb{R}^n`` is the objective coefficient vector
- ``A \in \mathbb{R}^{m \times n}`` is the constraint matrix
- ``L, U \in \mathbb{R}^m`` are lower and upper bounds on constraints
- ``l, u \in \mathbb{R}^n`` are lower and upper bounds on variables

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("HPRLP")  # Once registered
```

Or from GitHub:
```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/HPR-LP")
```

### Simple Example

```julia
using HPRLP
using SparseArrays

# Define LP: min -3x₁ - 5x₂ s.t. x₁ + 2x₂ ≤ 10, 3x₁ + x₂ ≤ 12, x ≥ 0
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
c = [-3.0, -5.0]
l = [0.0, 0.0]
u = [Inf, Inf]

params = HPRLP_parameters()
params.use_gpu = false  # Use CPU
params.verbose = false  # Silent mode

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Optimal value: ", result.primal_obj)
println("Solution: x = ", result.x)
```

### With JuMP

```julia
using JuMP, HPRLP

model = Model(HPRLP.Optimizer)
set_silent(model)  # Optional: suppress output

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, -3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

optimize!(model)
println("Objective: ", objective_value(model))
println("x1 = ", value(x1), ", x2 = ", value(x2))
```

## Documentation Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/mps_files.md",
    "guide/direct_api.md",
    "guide/jump_integration.md",
    "api.md",
    "examples.md",
]
Depth = 2
```

## Citation

If you use HPRLP in your research, please cite:

```bibtex
@article{chen2025hprlp,
  title={HPR-LP: An implementation of an HPR method for solving linear programming},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={Mathematical Programming Computation},
  volume={17},
  year={2025},
  doi={10.1007/s12532-025-00292-0}
}
```

## License

HPRLP.jl is licensed under the MIT License. See [LICENSE](https://github.com/PolyU-IOR/HPR-LP/blob/main/LICENSE) for details.
