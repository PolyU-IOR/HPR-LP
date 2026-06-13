import HPRLP

data_path = "xxx" # Replace with the actual path to your dataset
result_path = "xxx" # Replace with the actual path where you want to save the results

# Batch construction options
batch_size = 4
instance_mode = :copy # :copy or :perturbed_obj
obj_perturbation_scale = 1.0e-3
random_seed = 1234

# Current true-batched GPU path is presolve-free and GPU-only.
params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4
# params.max_iter = 10_000
params.device_number = 0
params.use_gpu = true
params.warm_up = true
params.verbose = true
params.presolve = "NONE"

HPRLP.run_batched_dataset(
    data_path,
    result_path,
    params;
    batch_size=batch_size,
    instance_mode=instance_mode,
    obj_perturbation_scale=obj_perturbation_scale,
    random_seed=random_seed,
)

# The results consist of the following files:
# - HPRLP_batched_result.csv: one aggregate row per LP file plus SGM10/solved summary rows
# - HPRLP_batched_columns.csv: one row per batch column for each LP file
# - HPRLP_batched_log.txt: a log file containing the output of HPRLP
# If one instance fails, the runner records an ERROR row, reclaims memory, and continues with the next file.