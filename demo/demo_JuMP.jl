using JuMP
using SparseArrays
import HPRLP

model = Model()

function simple_example(model)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    @objective(model, Min, -3x1 - 5x2)

    @constraint(model, 1x1 + 2x2 <= 10)
    @constraint(model, 3x1 + 1x2 <= 12)
end

# For more examples, please refer to the JuMP documentation: https://jump.dev/JuMP.jl/stable/tutorials/linear/introduction/
simple_example(model)

# Export the model to an MPS file (optional)
# write_to_file(model, "model.mps")
# Read the model from an MPS file (not commonly needed)
# model = read_from_file(mps_file_path)

params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.use_gpu = true
params.warm_up = true

# Run the HPR-LP algorithm for the JuMP model
HPRLP_result = HPRLP.run_JuMP_model(model, params)
# Optionally, if you want to run from MPS file, uncomment below:
# result = HPRLP.run_single("model.mps", params)

# Note: objective was already adjusted for maximization in extract_lp_data
println("\nObjective value: ", HPRLP_result.primal_obj)
println("x1 = ", HPRLP_result.x[1])
println("x2 = ", HPRLP_result.x[2])

