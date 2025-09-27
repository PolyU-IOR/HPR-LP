# HPR-LP: A GPU Solver for Linear Programming

> **A Julia implementation of the Halpern Peaceman-Rachford (HPR) method for solving linear programming (LP) problems on the GPU.**

---

## LP Problem Formulation

<div align="center">

$$
\begin{array}{ll}
\underset{x \in \mathbb{R}^n}{\min} \quad & \langle c, x \rangle \\
\text{s.t.} \quad & L \leq A x \leq U, \\
& l \leq x \leq u .
\end{array}
$$

</div>

---

## Numerical Results on an NVIDIA A100-SXM4-80GB GPU

**Numerical performance of HPR-LP and** [**cuPDLP.jl**](https://github.com/jinwen-yang/cuPDLP.jl) **(downloaded on July 24th, 2024) on 49 instances of **[**Mittelmann's LP benchmark set**](https://plato.asu.edu/ftp/lpfeas.html)** without Gurobi's presolve. Time limit 15000 seconds.**

<table>
  <thead>
    <tr>
      <th align="middle">Tolerance</th>
      <th align="middle">1e-4</th>
      <th align="middle">1e-4</th>
      <th align="middle">1e-6</th>
      <th align="middle">1e-6</th>
      <th align="middle">1e-8</th>
      <th align="middle">1e-8</th>
    </tr>
    <tr>
      <th align="middle">Solvers</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="middle">cuPDLP.jl</td>
      <td align="middle">76.9</td>
      <td align="middle">42</td>
      <td align="middle">156.2</td>
      <td align="middle">41</td>
      <td align="middle">277.9</td>
      <td align="middle">40</td>
    </tr>
    <tr>
      <td align="middle">HPR-LP (v0.1.0)</td>
      <td align="middle">30.2</td>
      <td align="middle">47</td>
      <td align="middle">69.1</td>
      <td align="middle">44</td>
      <td align="middle">103.8</td>
      <td align="middle">43</td>
    </tr>
    <tr>
      <td align="middle">HPR-LP (v0.1.2)</td>
      <td align="middle">25.7</td>
      <td align="middle">47</td>
      <td align="middle">53.2</td>
      <td align="middle">44</td>
      <td align="middle">82.1</td>
      <td align="middle">44</td>
    </tr>
  </tbody>
</table>

**Numerical performance of HPR-LP and cuPDLP.jl on 18 LP relaxations (>10M nonzeros in $A$) from **[**MIPLIB 2017**](https://miplib.zib.de/)** without Gurobi's presolve. Time limit 18000 seconds.**

<table>
  <thead>
    <tr>
      <th align="middle">Tolerance</th>
      <th align="middle">1e-4</th>
      <th align="middle">1e-4</th>
      <th align="middle">1e-6</th>
      <th align="middle">1e-6</th>
      <th align="middle">1e-8</th>
      <th align="middle">1e-8</th>
    </tr>
    <tr>
      <th align="middle">Solvers</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
      <th align="middle">SGM10</th>
      <th align="middle">Solved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="middle">cuPDLP.jl</td>
      <td align="middle">129.8</td>
      <td align="middle">16</td>
      <td align="middle">253.3</td>
      <td align="middle">15</td>
      <td align="middle">442.2</td>
      <td align="middle">14</td>
    </tr>
    <tr>
      <td align="middle">HPR-LP (v0.1.0)</td>
      <td align="middle">117.6</td>
      <td align="middle">17</td>
      <td align="middle">260.7</td>
      <td align="middle">15</td>
      <td align="middle">428.6</td>
      <td align="middle">14</td>
    </tr>
    <tr>
      <td align="middle">HPR-LP (v0.1.2)</td>
      <td align="middle">60.9</td>
      <td align="middle">17</td>
      <td align="middle">122.5</td>
      <td align="middle">17</td>
      <td align="middle">204.2</td>
      <td align="middle">17</td>
    </tr>
  </tbody>
</table>

---

# Getting Started

## Prerequisites

Before using HPR-LP, make sure the following dependencies are installed:

- **Julia** (Recommended version: `1.10.4`)
- **CUDA** (Required for GPU acceleration; install the appropriate version for your GPU and Julia)
- Required Julia packages

> To install the required Julia packages and build the HPR-LP environment, run:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

> To verify that CUDA is properly installed and working with Julia, run:
```julia
using CUDA
CUDA.versioninfo()
```

---

## Usage 1: Test Instances in MPS Format

### Setting Data and Result Paths

> Before running the scripts, please modify **`run_single_file.jl`** or **`run_dataset.jl`** in the scripts directory to specify the data path and result path according to your setup.

### Running a Single Instance

To test the script on a single instance (`.mps` file):

```bash
julia --project scripts/run_single_file.jl
```

### Running All Instances in a Directory

To process all `.mps` files in a directory:

```bash
julia --project scripts/run_dataset.jl
```

---

## Usage 2: Define Your LP Model in Julia Scripts

### Example 1: Build and Export an LP Model Using JuMP

This example demonstrates how to construct an LP model using the JuMP modeling language in Julia and export it to MPS format for use with the HPR-LP solver.

```bash
julia --project demo/demo_JuMP.jl
```

The script:
- Builds a linear programming (LP) model.
- Saves the model as an MPS file.
- Uses HPR-LP to solve the LP instance.

> **Remark:** If the model may be infeasible or unbounded, you can use HiGHS to check it.

```julia
using JuMP, HiGHS
## read a model from file (or create in other ways)
mps_file_path = "xxx" # your file path
model = read_from_file(mps_file_path)
## set HiGHS as the optimizer
set_optimizer(model, HiGHS.Optimizer)
## solve it
optimize!(model)
```

---

### Example 2: Define an LP Instance Directly in Julia

This example demonstrates how to construct and solve a linear programming problem directly in Julia without relying on JuMP.

```bash
julia --project demo/demo_Abc.jl
```

---

## Note on First-Time Execution Performance

You may notice that solving a single instance — or the first instance in a dataset — appears slow. This is due to Julia’s Just-In-Time (JIT) compilation, which compiles code on first execution.

> **💡 Tip for Better Performance:**  
> To reduce repeated compilation overhead, it’s recommended to run scripts from an **IDE like VS Code** or the **Julia REPL** in the terminal.

#### Start Julia REPL with the project environment:

```bash
julia --project
```

Then, at the Julia REPL, run demo/demo_Abc.jl (or other scripts):

```julia
include("demo/demo_Abc.jl")
```

> **CAUTION:**  
> If you encounter the error message:  
> `Error: Error during loading of extension AtomixCUDAExt of Atomix, use Base.retry_load_extensions() to retry`.
>
> Don’t panic — this is usually a transient issue. Simply wait a few moments; the extension typically loads successfully on its own.

---

## Parameters

Below is a list of the parameters in HPR-LP along with their default values and usage:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Default Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>warm_up</code></td><td><code>false</code></td><td>Determines if a warm-up phase is performed before main execution.</td></tr>
    <tr><td><code>time_limit</code></td><td><code>3600</code></td><td>Maximum allowed runtime (seconds) for the algorithm.</td></tr>
    <tr><td><code>stoptol</code></td><td><code>1e-6</code></td><td>Stopping tolerance for convergence checks.</td></tr>
    <tr><td><code>device_number</code></td><td><code>0</code></td><td>GPU device number (only relevant if <code>use_gpu</code> is true).</td></tr>
    <tr><td><code>max_iter</code></td><td><code>typemax(Int32)</code></td><td>Maximum number of iterations allowed.</td></tr>
    <tr><td><code>check_iter</code></td><td><code>150</code></td><td>Number of iterations to check residuals.</td></tr>
    <tr><td><code>use_Ruiz_scaling</code></td><td><code>true</code></td><td>Whether to apply Ruiz scaling.</td></tr>
    <tr><td><code>use_Pock_Chambolle_scaling</code></td><td><code>true</code></td><td>Whether to use the Pock-Chambolle scaling.</td></tr>
    <tr><td><code>use_bc_scaling</code></td><td><code>true</code></td><td>Whether to use the scaling for b and c.</td></tr>
    <tr><td><code>use_gpu</code></td><td><code>true</code></td><td>Whether to use GPU or not.</td></tr>
    <tr><td><code>print_frequency</code></td><td><code>-1</code> (auto)</td><td>Print the log every <code>print_frequency</code> iterations.</td></tr>
  </tbody>
</table>

---

# Result Explanation

After solving an instance, you can access the result variables as shown below:

```julia
# Example from /demo/demo_Abc.jl
println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
```

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Variable</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Iteration Counts</b></td><td><code>iter</code></td><td>Total number of iterations performed by the algorithm.</td></tr>
    <tr><td></td><td><code>iter_4</code></td><td>Number of iterations required to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>iter_6</code></td><td>Number of iterations required to achieve an accuracy of 1e-6.</td></tr>
    <tr><td><b>Time Metrics</b></td><td><code>time</code></td><td>Total time in seconds taken by the algorithm.</td></tr>
    <tr><td></td><td><code>time_4</code></td><td>Time in seconds taken to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>time_6</code></td><td>Time in seconds taken to achieve an accuracy of 1e-6.</td></tr>
    <tr><td></td><td><code>power_time</code></td><td>Time in seconds used by the power method.</td></tr>
    <tr><td><b>Objective Values</b></td><td><code>primal_obj</code></td><td>The primal objective value obtained.</td></tr>
    <tr><td></td><td><code>gap</code></td><td>The gap between the primal and dual objective values.</td></tr>
    <tr><td><b>Residuals</b></td><td><code>residuals</code></td><td>Relative residuals of the primal feasibility, dual feasibility, and duality gap.</td></tr>
    <tr><td><b>Algorithm Status</b></td><td><code>output_type</code></td><td>The final status of the algorithm:<br/>- <code>OPTIMAL</code>: Found optimal solution<br/>- <code>MAX_ITER</code>: Max iterations reached<br/>- <code>TIME_LIMIT</code>: Time limit reached</td></tr>
    <tr><td><b>Solution Vectors</b></td><td><code>x</code></td><td>The final solution vector <code>x</code>.</td></tr>
    <tr><td></td><td><code>y</code></td><td>The final solution vector <code>y</code>.</td></tr>
    <tr><td></td><td><code>z</code></td><td>The final solution vector <code>z</code>.</td></tr>
  </tbody>
</table>

---

## Citation

```bibtex
@article{chen2024hpr,
  title={HPR-LP: An implementation of an HPR method for solving linear programming},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={arXiv preprint arXiv:2408.12179},
  year={2024}
}
```
