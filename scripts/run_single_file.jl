import HprLP

file_name = "model.mps" # Replace with the actual path to your LP file

params = HprLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.use_gpu = true
params.warm_up = true

result = HprLP.run_single(file_name, params)

println("Objective value: ", result.primal_obj)