import HPRLP

file_name = "model.mps" # Replace with the actual path to your LP file

# Build the model from MPS file
model = HPRLP.build_from_mps(file_name)
# model = HPRLP.read_from_hdf5("model.h5") # Read the model from HDF5 format if it exists

# HPRLP.save_lp_to_hdf5("model.h5", model) # Save the model to HDF5 format for future use

# Set up parameters
params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-8 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0

params.use_gpu = true
params.warm_up = false
params.presolve = "NONE"

params.max_iter = 50000
# params.auto_save = true
# params.save_filename = "test.h5"

params.debug_restart = true
params.debug_sigma = true

# Optimize the model
result = HPRLP.optimize(model, params)

params.max_iter = 99999999
result = HPRLP.optimize_from_autosave(model, params; filename="test.h5")

println("Objective value: ", result.primal_obj)
println("Status: ", result.status)
