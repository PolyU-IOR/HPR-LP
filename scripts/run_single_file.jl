import HPRLP
using JuMP

file_name = "model.mps" # Replace with the actual path to your LP file

# Build the model from MPS file
# model = read_from_file(file_name)
model = HPRLP.build_from_mps(file_name)
params = HPRLP.HPRLP_parameters()
params.stoptol = 1e-4
params.time_limit = 3600.0
params.use_gpu = true
params.device_number = 0
params.warm_up = false


# Optimize the model
result = HPRLP.optimize(model, params)

println("Objective value: ", result.primal_obj)
println("Status: ", result.status)
