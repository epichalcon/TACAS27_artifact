#!/usr/bin/env python3

'''
Optimizes the blocks form statsdir/cleaned_blocks by applying max-sat.
'''

import csv
import subprocess
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from tqdm import tqdm

GASOL_GREEDY_EXEC = "/app/gasol-optimizer/gasol_asm.py"
TIMEOUT = 600
WORKERS = 4

def process_block(block_id, block_code, res_name, command, sub_res_dir):
    logs_dir = f"{sub_res_dir}/logs"
    blocks_dir = f"{sub_res_dir}/blocks"
    stats_dir = f"{sub_res_dir}/stats"
    err_dir = f"{sub_res_dir}/err"

    # Write block to file
    block_file = f"block_{res_name}_{block_id}.txt"
    with open(block_file, 'w') as f:
        f.write(block_code)

    cmd = f"timeout --foreground {TIMEOUT} python3 {GASOL_GREEDY_EXEC} -bl {block_file} {command}"

    # Run command
    log_file = f"{logs_dir}/{block_id}.log"
    err_file = f"{err_dir}/{block_id}.log"
    try:
        with open(log_file, 'w') as log, open(err_file, 'w') as err:
            subprocess.run(['bash', '-c', cmd], stdout=log, stderr=err)
    except subprocess.TimeoutExpired:
        with open(err_file, 'a') as err:
            err.write("Timeout\n")

    # Move files
    optimized_file = f"block_{res_name}_{block_id}_optimized.json_solc"
    stats_file = f"block_{res_name}_{block_id}_statistics_seq.csv"
    stats_block_file = f"block_{res_name}_{block_id}_statistics_blocks.csv"
    if os.path.exists(optimized_file):
        shutil.move(optimized_file, f"{blocks_dir}/{block_id}.txt")
    if os.path.exists(stats_file):
        shutil.move(stats_file, f"{stats_dir}/{block_id}.csv")
        shutil.move(stats_block_file, f"{stats_dir}/{block_id}_blocks.csv")

    # Clean up
    for f in [block_file, optimized_file, stats_file, stats_block_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    dir_path = os.getcwd()

    csv_file = f"statsdir/training_blocks_cleaned.csv"
    res_dir = f"{dir_path}/processed_blocks/"
    os.makedirs(res_dir, exist_ok=True)

    mode_list = [
        ("sat", "--sat --length"),
    ]

    all_tasks = []
    for res_name, command in mode_list:
        sub_res_dir = f"{res_dir}/{res_name}"
        stats_dir = f"{sub_res_dir}/stats"
        blocks_dir = f"{sub_res_dir}/blocks"
        logs_dir = f"{sub_res_dir}/logs"
        err_dir = f"{sub_res_dir}/err"
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(blocks_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)

        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # skip header
            for row in reader:
                block_id = row[1]
                block_code = row[16]  
                all_tasks.append((block_id, block_code, res_name, command, sub_res_dir))

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process_block, *task) for task in all_tasks]
        with tqdm(total=len(all_tasks), desc="Processing blocks") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")
                pbar.update(1)

    
