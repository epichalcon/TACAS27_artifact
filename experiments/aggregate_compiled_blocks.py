
'''
Step 2. aggregate all the compiled blocks to the statsdir/blocks.csv file
'''

import sys 
import os 
import glob
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

HEAD_SEQ='numid,block_id,previous_solution,timeout,initial_n_instrs,initial_estimated_size,initial_estimated_gas,rules,initial_length,saved_length,chosen_block_tag,solver_time_in_sec,saved_size,saved_gas,model_found,shown_optimal,solution_found,optimized_n_instrs,optimized_length,optimized_estimated_size,optimized_estimated_gas,outcome,forves_checker'

HEAD_BLOCKS=",block_id,old_instrs,new_instrs,old_gas,new_gas,old_size,new_size,old_length,new_length,gasol_checker,gasol_checker_reason,forves_checker,saved_gas,saved_size,saved_length"

#FILTER=sys.argv[1]

INPUTDIR = f'intermediate'
OUTPUTDIR='statsdir'

cmd_mkdir=f'mkdir -p {OUTPUTDIR}'
os.system(cmd_mkdir)

# Create an empty DataFrame to store the combined data
combined_seq_data = pd.DataFrame()
combined_blocks_data = pd.DataFrame()

# Get all matching CSV files
input_dirs = glob.glob(f'{INPUTDIR}/*/*/csvs/')

not_processed = 0

# Process each file
for input_dir in input_dirs:
    try:
        if "optimize_run" not in input_dir:
            continue
        contract = input_dir.split("/")[1]
        file_seq = f"{input_dir}/{contract}_seq.csv"
        file_blocks = f"{input_dir}/{contract}_blocks.csv"
        # Read the current CSV file
        current_data = pd.read_csv(file_seq, sep=',')
        current_blocks_data = pd.read_csv(file_blocks, sep=',')
        
        # Skip header rows if they contain "solution"
        current_data = current_data[~current_data.astype(str).apply(lambda x: x.str.contains('solution', case=False)).any(axis=1)]
        current_blocks_data = current_blocks_data[~current_blocks_data.astype(str).apply(lambda x: x.str.contains('solution', case=False)).any(axis=1)]
        
        # Append to the combined DataFrame
        combined_blocks_data = pd.concat([combined_blocks_data, current_blocks_data], ignore_index=True)
        combined_seq_data = pd.concat([combined_seq_data, current_data], ignore_index=True)
    except Exception as e:
        #print(f"Error processing file {file}: {e}")
        not_processed += 1

print("not processed", not_processed)

# Ensure all expected columns exist (from HEAD_SEQ)
expected_columns = HEAD_SEQ.split(',')
for col in expected_columns:
    if col not in combined_seq_data.columns:
        combined_seq_data[col] = None  # Add missing columns with None values

# Reorder columns to match HEAD_SEQ
combined_seq_data = combined_seq_data[expected_columns]

# Save the combined data to the output CSV file
# The rest of the script remains largely the same
data = combined_seq_data.copy()


# Ensure all expected columns exist (from HEAD_SEQ)
expected_columns = HEAD_BLOCKS.split(',')
for col in expected_columns:
    if col not in combined_blocks_data.columns:
        combined_blocks_data[col] = None  # Add missing columns with None values

# Reorder columns to match HEAD_SEQ
combined_blocks_data = combined_blocks_data[expected_columns]

# Save the combined data to the output CSV file
# The rest of the script remains largely the same
data_block = combined_blocks_data.copy()

data.loc[data["outcome"] == "no_model", "optimized_n_instrs"] = data["initial_n_instrs"]
data.loc[data["outcome"] == "no_model", "solution_found"] = data["previous_solution"]

data.to_csv("statsdir/blocks.csv", index=False, header=True,sep=';')

print(f"Total number of blocks: {len(data)}")