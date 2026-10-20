'''
Takes the contents of intermediate/processed_blocks and generates a csv with 
the data from all the processed blocks
Input:      intermediate/processed_blocks
Output:     statsdir/splitted_blocks/
'''

from numpy import block
import pandas as pd
import sys
import re
from pathlib import Path

SIZE=30

# Define the base path
script_dir = Path(__file__).resolve().parent
base_path = script_dir / "check_for_optimality"
output_base = script_dir / "statsdir" / "splitted_blocks_check_for_optimality"

modes = {
        "sat": [],
        }

def find_line_after_sequence(file_path, search_phrase):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        index = next((i for i, line in enumerate(lines) 
                      if line.find(search_phrase) != -1), None)

        if index is not None and index + 1 < len(lines):
            return lines[index+1].strip()
    except (FileNotFoundError, StopIteration):
        pass
    return None


def extract_data(file_path, phrase):
    try:
        content = file_path.read_text()
        match = re.search(fr"{phrase}\s*(\d+\.?\d*)", content)
        if match and match.group(1):
            str_n = match.group(1)
            return float(str_n) if '.' in str_n else int(str_n)

    except (IOError, ValueError):
        return None 


# Ensure the output directory exists
output_base.mkdir(parents=True, exist_ok=True)

over_path = script_dir / "statsdir" / f"blocks_cleaned.csv"
try: 
    over_csv = pd.read_csv(over_path, delimiter=";")
    # Keep only the last occurrence of each block_id
    over_csv = over_csv.drop_duplicates(subset=["block_id"], keep='last')
    over_csv.set_index("block_id", inplace=True)
except:
    print(f"file {over_path} does not exist")

# Iterate through execution mode directories
for execution_mode in base_path.iterdir():
    print(execution_mode)
    stats_path = execution_mode / "stats"
    blocks_path = execution_mode / "blocks"
    logs_path = execution_mode / "logs"

    n_files = 0


    if stats_path.is_dir() and blocks_path.is_dir():
        data = modes[execution_mode.name]

        # Iterate through CSV files in stats/
        for csv_file in stats_path.glob("*.csv"):
            n_files += 1
            block_id = csv_file.stem  # Get filename without extension
            #print(block_id)

            # Read CSV and check row count
            results = pd.read_csv(csv_file)

            log_file = logs_path / f"{block_id}.log"

            if results.empty:
                continue

            execution_result = "found"


            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as log:
                    log_content = log.read()
                    if "UNSATISFIABLE" in log_content:
                        execution_result = "unsat"
                    elif "UNKNOWN" in log_content:
                            execution_result = "not_found"
                    elif "segmentation fault" in log_content or "Error:" in log_content:
                            execution_result = "mzk fault"
                    elif "type error" in log_content:
                            execution_result = "gasol error"


            if  "optimized_n_instrs" not in results.columns:
                results["optimized_n_instrs"] = results["initial_n_instrs"] - results["saved_length"]
                results["optimized_length"] = results["initial_length"] - results["saved_length"]
                results["optimized_estimated_size"] = results["initial_estimated_size"] - results["saved_size"]
                results["optimized_estimated_gas"] = results["initial_estimated_gas"] - results["saved_gas"]

            if execution_result != "found" or results["outcome"].any() == "no_model":

                row_data = {
                    "block_id": block_id,
                    "previous_solution": " ".join(results["previous_solution"]),
                    "timeout": results["timeout"].iloc[0],  # Same for both rows

                    "initial_n_instrs"          : over_csv.loc[block_id, "optimized_n_instrs"],
                    "initial_length"            : over_csv.loc[block_id, "optimized_length"],
                    "initial_estimated_size"    : over_csv.loc[block_id, "optimized_estimated_size"],
                    "initial_estimated_gas"     : over_csv.loc[block_id, "optimized_estimated_gas"],
                    "optimized_n_instrs"        : over_csv.loc[block_id, "optimized_n_instrs"],
                    "optimized_length"          : over_csv.loc[block_id, "optimized_length"],
                    "optimized_estimated_size"  : over_csv.loc[block_id, "optimized_estimated_size"],
                    "optimized_estimated_gas"   : over_csv.loc[block_id, "optimized_estimated_gas"],
                    "solver_time_in_sec"        : results["solver_time_in_sec"].sum(), 
                    "saved_length"              : 0, 
                    "saved_size"                : 0,
                    "saved_gas"                 : 0,
                    "solution_found"            : over_csv.loc[block_id, "solution_found"],
                    "model_found"               : False,
                    "partially_optimal"         : False,
                    "improvement_percentage"    : 0,
                    "tmp/inst_ratio"            : results["solver_time_in_sec"].sum()/over_csv.loc[block_id, "initial_n_instrs"],
                }
            else:
                block_file = blocks_path / f"{block_id}.txt"
                try:
                    solution_found = " ".join(results["solution_found"])
                except:
                    solution_found = block_file.read_text().strip() if block_file.exists() else "",
                row_data = {
                    "block_id"                  : block_id,
                    "previous_solution"         : " ".join(results["previous_solution"]),
                    "timeout"                   : results["timeout"].iloc[0],
                    **{col: results[col].sum() for col in ["initial_n_instrs", "initial_estimated_size", 
                                                        "initial_estimated_gas", "initial_length", 
                                                        "solver_time_in_sec", "saved_length", 
                                                        "saved_size", "saved_gas",
                                                        "optimized_n_instrs", "optimized_length",
                                                        "optimized_estimated_size", "optimized_estimated_gas"]},
                    "model_found"               : results["model_found"].all(),  
                    "partially_optimal"         : results["shown_optimal"].all(),
                    "solution_found"            : solution_found,
                }

                row_data["improvement_percentage"] = row_data["saved_gas"] / row_data["initial_estimated_gas"] * 100


                row_data["tmp/inst_ratio"] = row_data["solver_time_in_sec"]/row_data["initial_n_instrs"]

            row_data["result_optim"] = execution_result

            data.append(row_data)
    print(n_files)

for mode, data in modes.items():
    # Create DataFrame
    df_final = pd.DataFrame(data)

    # Define output file path
    output_file = output_base / f"{mode}.csv"

    # Save DataFrame to CSV
    df_final.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
