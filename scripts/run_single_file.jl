import HPRLP

file_name = "model.mps" # Replace with the actual path to your LP file

# Build the model from MPS file
model = HPRLP.build_from_mps(file_name)

# Set up parameters
params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.use_gpu = true
params.warm_up = true

# Optimize the model
result = HPRLP.optimize(model, params)

println("Objective value: ", result.primal_obj)
println("Status: ", result.output_type)
