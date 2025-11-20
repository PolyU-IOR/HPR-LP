import HPRLP

data_path = "xxx" # Replace with the actual path to your dataset
result_path = "xxx" # Replace with the actual path where you want to save the results

params = HPRLP.HPRLP_parameters()
params.time_limit = 3600
params.stoptol = 1e-4 # can be adjusted as needed to higher accuracy such as 1e-9
params.device_number = 0
params.warm_up = true
params.use_gpu = true

HPRLP.run_dataset(data_path, result_path, params)

# The results consist of the following files:
# - HPRLP_result.csv: a CSV file containing the results of the experiment
# - HPRLP_log.txt: a log file containing the output of HPRLP