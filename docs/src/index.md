# HPRLP.jl Documentation

**A Julia implementation of the Halpern Peaceman-Rachford (HPR) method for solving linear programming (LP) problems on the GPU.**

## Overview

HPRLP.jl is a high-performance linear programming solver that leverages GPU acceleration to solve large-scale LP problems efficiently. It implements the Halpern Peaceman-Rachford splitting method with adaptive restart strategy and penalty parameter selection.

## Features

- ✅ **GPU Acceleration**: Native CUDA support for solving large-scale problems
- ✅ **CPU Support**: Support CPU mode when GPU is not available
- ✅ **Multiple Inputs**: 
  - Direct API with matrix inputs
  - MPS file format support
  - JuMP integration via MOI wrapper
- ✅ **Flexible Scaling**: Ruiz, Pock-Chambolle, and scalar scaling methods
- ✅ **Adaptive Algorithms**: Automatic restart strategy and penalty parameter selection

## Problem Formulation

HPRLP solves linear programming problems of the form:

```math
\begin{array}{ll}
\underset{x \in \mathbb{R}^n}{\min} \quad & \langle c, x \rangle \\
\text{s.t.} \quad & L \leq A x \leq U, \\
& l \leq x \leq u .
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

From GitHub (recommended for applications):
```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/HPR-LP")
```

Locally (recommended for development):
```bash
git clone https://github.com/PolyU-IOR/HPR-LP.git
cd HPR-LP
julia --project=. -e 'using Pkg; Pkg.instantiate()'
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
params.stoptol = 1e-9  # Set stopping tolerance

result = run_lp(A, AL, AU, c, l, u, 0.0, params)

println("Optimal value: ", result.primal_obj)
println("Solution: x = ", result.x)
```

### With JuMP

```julia
using JuMP, HPRLP

model = Model(HPRLP.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, -3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

set_attribute(model, "stoptol", 1e-9)  # Set stopping tolerance

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
@article{chen2025hpr,
  title={HPR-LP: An implementation of an HPR method for solving linear programming.},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={Mathematical Programming Computation},
  pages={1--28},
  year={2025},
  publisher={Springer}
}
```

## License

HPRLP.jl is licensed under the MIT License. See [LICENSE](https://github.com/PolyU-IOR/HPR-LP/blob/main/LICENSE) for details.
