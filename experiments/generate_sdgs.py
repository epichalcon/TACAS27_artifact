#!/usr/bin/env python3


'''
Generates de sdg json of the pre and post greedy blocks.
'''

import csv
import subprocess
import os
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from tqdm import tqdm
from utils.notify import send_msg

GASOL_GREEDY_EXEC = "~/gasol-optimizer/gasol_asm.py"
TIMEOUT = 600
WORKERS = 64

def process_block(block_id, block_code, res_dir):
    logs_dir = f"{res_dir}/logs"
    sdg_dir = f"{res_dir}/sdgs"
    err_dir = f"{res_dir}/err"

    # Write block to file
    block_file = f"block_{block_id}.txt"
    with open(block_file, 'w') as f:
        f.write(block_code)

    cmd = f"source ~/gasol-optimizer/venv/bin/activate && timeout --foreground {TIMEOUT} python3 {GASOL_GREEDY_EXEC} -bl {block_file} -sdg"

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
    sdg_file = f"block_{block_id}_sdg.json"
    if os.path.exists(sdg_file):
        shutil.move(sdg_file, f"{sdg_dir}/{block_id}.json")



def process_blocks_and_clean(block_id, previous_solution, opt_solution,  pre_res_dir, post_res_dir):

    process_block(block_id, previous_solution, pre_res_dir)
    process_block(block_id, opt_solution, post_res_dir)

    optimized_file = f"block_{block_id}_optimized.json_solc"
    stats_file = f"block_{block_id}_statistics_seq.csv"
    stats_block_file = f"block_{block_id}_statistics_blocks.csv"
    sdg_file = f"block_{block_id}_sdg.json"
    block_file = f"block_{block_id}.txt"
    
    # Clean up
    for f in [block_file, optimized_file, stats_file, stats_block_file, sdg_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    dir_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Generate SDG JSON files for blocks in a semicolon-delimited CSV."
    )
    parser.add_argument(
        "--input",
        default="statsdir/blocks_cleaned.csv",
        help="Path to the input blocks CSV",
    )

    parser.add_argument(
        "--pre-dir",
        default="pre_optimized",
        help="Directory for the pre optimized sdgs",
    )
    parser.add_argument(
        "--post-dir",
        default="post_optimized",
        help="Directory for the post optimized sdgs",
    )
    args = parser.parse_args()

    csv_file = args.input


    res_dir = f"{dir_path}/sdgs/"
    os.makedirs(res_dir, exist_ok=True)

    pre_optimized = f"{res_dir}/{args.pre_dir}"
    post_optimized = f"{res_dir}/{args.post_dir}"

    for dir in [pre_optimized, post_optimized]:
        

        os.makedirs(dir, exist_ok=True)

        sdgs = f"{dir}/sdgs"
        logs_dir = f"{dir}/logs"
        err_dir = f"{dir}/err"

        os.makedirs(sdgs, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)

    all_tasks = []


    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            block_id = row[1]
            previous_solution = row[2]  # index of previous_solution column
            opt_solution = row[16]  # new_instrs
            all_tasks.append((block_id, previous_solution, opt_solution,  pre_optimized, post_optimized))

    all_tasks = [all_tasks[0]]
        
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process_blocks_and_clean, *task) for task in all_tasks]
        with tqdm(total=len(all_tasks), desc="Generating sdgs") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")
                pbar.update(1)

    
    for f in ["graph", "graph.dot", "graph.png"]:
        if os.path.exists(f):
            os.remove(f)

    send_msg("sdg processing finished")
    
