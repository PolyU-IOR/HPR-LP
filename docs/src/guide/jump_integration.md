# JuMP Integration

HPRLP integrates seamlessly with JuMP through the MathOptInterface (MOI), allowing you to use HPRLP as a backend solver for JuMP models.

## Basic Usage

```julia
using JuMP
using HPRLP

model = Model(HPRLP.Optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)
@objective(model, Min, -3x - 5y)
@constraint(model, x + 2y <= 10)
@constraint(model, 3x + y <= 12)

optimize!(model)

println("Optimal value: ", objective_value(model))
println("x = ", value(x), ", y = ", value(y))
```

## Setting Solver Attributes

### Using `set_optimizer_attribute`

```julia
model = Model(HPRLP.Optimizer)

# Standard MOI attributes
set_silent(model)                      # Suppress output
set_time_limit_sec(model, 3600.0)     # 1 hour time limit

# HPRLP-specific attributes
set_optimizer_attribute(model, "stoptol", 1e-6)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "device_number", 0)
set_optimizer_attribute(model, "use_Ruiz_scaling", true)
set_optimizer_attribute(model, "warm_up", true)
set_optimizer_attribute(model, "max_iter", 100000)

# New features: warm-start and auto-save
set_optimizer_attribute(model, "initial_x", x0)  # Warm-start primal
set_optimizer_attribute(model, "initial_y", y0)  # Warm-start dual
set_optimizer_attribute(model, "auto_save", true)
set_optimizer_attribute(model, "save_filename", "optimization.h5")
```

!!! tip "Parameter Reference"
    For detailed explanations of all parameters, see the [Parameters](parameters.md) guide.

## Querying Results

### Termination Status

```julia
optimize!(model)

status = termination_status(model)

if status == MOI.OPTIMAL
    println("Optimal solution found!")
elseif status == MOI.TIME_LIMIT
    println("Time limit reached")
elseif status == MOI.ITERATION_LIMIT
    println("Iteration limit reached")
end
```

### Objective and Solutions

```julia
if has_values(model)
    obj_val = objective_value(model)
    x_val = value(x)
    y_val = value(y)
    
    println("Objective: $obj_val")
    println("x = $x_val, y = $y_val")
end
```

### Solve Time

```julia
time = solve_time(model)
println("Solved in $time seconds")
```

## Silent Mode

Suppress all solver output:

```julia
model = Model(HPRLP.Optimizer)
set_silent(model)

# Build model...
optimize!(model)
# No output from solver
```

Or equivalently:

```julia
model = Model(HPRLP.Optimizer)
set_optimizer_attribute(model, "verbose", false)
optimize!(model)
```

## Common Patterns

### Production Portfolio

```julia
using JuMP, HPRLP

# Production planning
products = 1:5
resources = 1:3

profit = rand(5) .* 10
resource_usage = rand(3, 5)
resource_capacity = rand(3) .* 100

model = Model(HPRLP.Optimizer)
set_silent(model)

@variable(model, production[products] >= 0)
@objective(model, Max, sum(profit[p] * production[p] for p in products))

for r in resources
    @constraint(model, 
        sum(resource_usage[r,p] * production[p] for p in products) 
        <= resource_capacity[r]
    )
end

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("Optimal profit: ", objective_value(model))
    for p in products
        println("Product $p: ", value(production[p]))
    end
end
```

### Network Flow

```julia
using JuMP, HPRLP

# Simple network flow
nodes = 1:5
arcs = [(1,2), (1,3), (2,4), (3,4), (4,5)]
capacity = Dict(arcs .=> rand(length(arcs)) .* 10)
supply = [10.0, 0, 0, 0, -10]  # Source at 1, sink at 5

model = Model(HPRLP.Optimizer)
set_silent(model)

@variable(model, 0 <= flow[a in arcs] <= capacity[a])
@objective(model, Min, sum(flow[a] for a in arcs))  # Min total flow

# Flow conservation
for n in nodes
    incoming = [a for a in arcs if a[2] == n]
    outgoing = [a for a in arcs if a[1] == n]
    
    @constraint(model,
        sum(flow[a] for a in incoming) - 
        sum(flow[a] for a in outgoing) == -supply[n]
    )
end

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    for a in arcs
        if value(flow[a]) > 1e-6
            println("Arc $a: ", value(flow[a]))
        end
    end
end
```

### Diet Problem

```julia
using JuMP, HPRLP

# Classic diet problem
foods = ["Bread", "Milk", "Eggs", "Meat", "Cake"]
nutrients = ["Calories", "Protein", "Fat"]

cost = [2.0, 3.5, 2.5, 8.0, 5.0]
nutrition = [
    300 200 100 400 500;  # Calories
     10  15  12  30   5;  # Protein (g)
      5  10   8  20  15   # Fat (g)
]
min_nutrient = [2000, 50, 30]  # Daily requirements

model = Model(HPRLP.Optimizer)
set_silent(model)

@variable(model, servings[1:5] >= 0)
@objective(model, Min, sum(cost[i] * servings[i] for i in 1:5))

for n in 1:3
    @constraint(model,
        sum(nutrition[n,f] * servings[f] for f in 1:5) >= min_nutrient[n]
    )
end

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("Minimum cost: \$", objective_value(model))
    for (i, food) in enumerate(foods)
        if value(servings[i]) > 1e-6
            println("$food: ", value(servings[i]), " servings")
        end
    end
end
```

## Reading MPS Files with JuMP

You can read MPS files and solve them with HPRLP via JuMP:

```julia
using JuMP, HPRLP

model = read_from_file("problem.mps")
set_optimizer(model, HPRLP.Optimizer)

# Set attributes
set_optimizer_attribute(model, "stoptol", 1e-6)

# Solve
optimize!(model)

println("Status: ", termination_status(model))
if has_values(model)
    println("Objective: ", objective_value(model))
end
```

## Comparison with Other Solvers

Easy to switch between solvers:

```julia
using JuMP, HPRLP, HiGHS

# Build model
function build_model()
    model = Model()
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @objective(model, Min, -3x - 5y)
    @constraint(model, x + 2y <= 10)
    @constraint(model, 3x + y <= 12)
    return model
end

# Solve with HPRLP
model1 = build_model()
set_optimizer(model1, HPRLP.Optimizer)
optimize!(model1)
println("HPRLP: ", objective_value(model1))

# Solve with HiGHS
model2 = build_model()
set_optimizer(model2, HiGHS.Optimizer)
optimize!(model2)
println("HiGHS: ", objective_value(model2))
```

## Performance Tips

1. **Set silent mode** for faster execution in production
2. **Disable warm-up** if solving many small problems
3. **Use GPU** for problems with > 10,000 variables
4. **Adjust tolerance** based on application needs
5. **Reuse models** when solving similar problems repeatedly

## Troubleshooting

### Slow First Run

Julia's JIT compilation causes slow first runs. Use warm-up:
```julia
set_optimizer_attribute(model, "warm_up", true)  # Default
```

## See Also

- [Parameters](parameters.md) - Complete guide to all solver parameters and their effects
- [Output & Results](output_results.md) - Understanding solver output and solution quality