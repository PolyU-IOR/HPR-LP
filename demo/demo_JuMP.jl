using JuMP
using SparseArrays
import HPRLP

# =============================================================================
# Example: Using the MOI Optimizer with JuMP
# =============================================================================

model = Model(HPRLP.Optimizer)

# Set solver attributes
set_attribute(model, "stoptol", 1e-4)
set_attribute(model, "time_limit", 3600.0)
set_attribute(model, "use_gpu", true)
set_attribute(model, "device_number", 0)
set_attribute(model, "warm_up", true)

function simple_example(model)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    @objective(model, Min, -3x1 - 5x2)

    @constraint(model, 1x1 + 2x2 <= 10)
    @constraint(model, 3x1 + 1x2 <= 12)
end

# For more examples, please refer to the JuMP documentation: https://jump.dev/JuMP.jl/stable/tutorials/linear/introduction/
simple_example(model)

# Solve using MOI
optimize!(model)

# Check solution status
println("Termination status: ", termination_status(model))
# Print solve time and iterations
println("Solve time: ", solve_time(model), " seconds")
# Get solution
if has_values(model)
    println("Objective value: ", objective_value(model))
    println("x1 = ", value(model[:x1]))
    println("x2 = ", value(model[:x2]))
end

## Usage 2: read a model from file
mps_file_path = "model.mps" # your file path
model = read_from_file(mps_file_path)
## set HPRLP as the optimizer
set_optimizer(model, HPRLP.Optimizer)
## solve it
optimize!(model)