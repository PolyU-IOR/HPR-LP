import HPRLP
using JuMP

file_name = "model.mps" # Replace with the actual path to your LP file

# Build the model from MPS file
model = read_from_file(file_name)

## set HPRLP as the optimizer
set_optimizer(model, HPRLP.Optimizer)

set_attribute(model, "stoptol", 1e-4)
set_attribute(model, "time_limit", 3600.0)
set_attribute(model, "use_gpu", true)
set_attribute(model, "device_number", 0)
set_attribute(model, "warm_up", true)

# Optimize the model
optimize!(model)
result = unsafe_backend(model).results

println("Objective value: ", result.primal_obj)
println("Status: ", result.status)
