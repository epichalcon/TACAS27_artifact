"""
Step 1: takes the contracts from the corresponding folder and compiles the source code.
"""


THREADS = 64

import argparse
import os
import shutil
import subprocess
import glob
import sys
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

DIR_PATH = "intermediate/"
DATE = "20251112"
DATE = "20260609"
DATE = "20260922"
DATE = "20240226"
SOL_DIR = f"contracts-{DATE}"
GASOL_GREEDY_EXEC = "/app/gasol-optimizer/gasol_asm.py"
COMPILATION_OPTION = f"compilation_options_{DATE}.csv"
COMPILERS_DIR = "/app/solc-bin-linux-amd64/"
TIMEOUT = 1200
WORKERS = 64
ORIGINAL_CWD = os.getcwd()

# Parameters
SOL_PARAMS = "--combined-json asm"
LOGSPARAM = "logs_optim"
EXECID = "getblocks"
OPTIM_PARAMS = "-length"
OPTIM_PARAMS = "-length --greedy"

compilation_options_csv = pd.read_csv(COMPILATION_OPTION, index_col=False)

def create_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def clean_up(patterns):
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(SOL_DIR, pattern)):
            os.remove(file_path)


def get_sol_files(filter):
    # Find SOL files
    print(f"LS: ls {SOL_DIR}/* | grep {filter} | grep '.sol' | grep ^0x")

    sol_files = []
    for file_path in glob.glob(os.path.join(SOL_DIR, "*")):
        if (filter in file_path and 
            file_path.endswith(".sol") and 
            os.path.basename(file_path).startswith("0x")):
            sol_files.append(os.path.basename(file_path))
    print(SOL_DIR)

    return sol_files


def prepare_dir(resdir):
    results_ok = os.path.join(resdir, "exec_OK")
    results_unknown = os.path.join(resdir, "exec_ERROR")
    
    if os.path.exists(resdir):
        shutil.rmtree(resdir)
    
    os.makedirs(os.path.join(DIR_PATH, "results"), exist_ok=True)
    os.makedirs(resdir, exist_ok=True)
    os.makedirs(results_ok, exist_ok=True)
    os.makedirs(results_unknown, exist_ok=True)
    
    return results_ok, results_unknown


def get_compiler(fname):
    compiler = list(compilation_options_csv[compilation_options_csv["address"] == fname]["CompilerVersion"])[0]
    compiler = compiler.split("+")[0]
    #compiler = "v0.8.24"
    compiler = [comp for comp in glob.glob(os.path.join(COMPILERS_DIR, "*")) if os.path.basename(comp).startswith(f"solc-linux-amd64-{compiler}")][0]

    return compiler

def get_file_type(fname):
    return list(compilation_options_csv[compilation_options_csv["address"] == fname]["origin"])[0]


def store_results(base_dir, paths):
    os.makedirs(base_dir, exist_ok=True)
        
    for src, dest_file in paths:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(base_dir, dest_file))


def compile_single_contract(sol, optimize=False):

    fname = Path(sol).stem
    solfile = os.path.join(SOL_DIR, sol)

    compiler = get_compiler(fname)


    contract_path = os.path.join(DIR_PATH, fname)

    if optimize:
        contract_path = os.path.join(contract_path, "optimize_run")
    else:
        contract_path = os.path.join(contract_path, "not_optimized")

    contract_jsondir = os.path.join(contract_path, "jsons")

    os.makedirs(contract_jsondir, exist_ok=True)

    if get_file_type(fname) == "json":
        result = subprocess.run(
            f"timeout --foreground 120 {compiler} {solfile} --standard-json",
            shell=True,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            f"{compiler} {solfile} {SOL_PARAMS} {'--optimize' if optimize else ''}",
            shell=True,
            capture_output=True,
            text=True,
        )


    json_file_path = os.path.join(contract_jsondir, f"{fname}.json")

    # 2. Write the initial stdout to the file
    with open(json_file_path, "w") as f:
        f.write(result.stdout)

    if get_file_type(fname) == "json":
        with open(json_file_path, "r") as f:
            compiled = json.load(f)
        
        compiled["version"] = compiler
        
        with open(json_file_path, "w") as f:
            json.dump(compiled, f, indent=4)

    # store results
    if result.returncode == 0:
        res = "ok"
        store_results(os.path.join(contract_path, "result_OK"), [
            ("exception_info.csv", "exception_info.csv"),
            (solfile, sol)
        ])
    else:
        res = "error"
        store_results(os.path.join(contract_path, "result_ERROR"), [(solfile, sol)])

        with open(os.path.join(os.path.join(contract_path, "result_ERROR"), "error.log"), "w") as f:
            f.write(result.stderr)
    return res



def optimize_single_contract(fname, optimized=False):

    contract_path = os.path.join(DIR_PATH, fname)

    if optimized:
        contract_path = os.path.join(contract_path, "optimize_run")
    else:
        contract_path = os.path.join(contract_path, "not_optimized")


    contract_jsondir = os.path.join(contract_path, "jsons")
    contract_csvdir = os.path.join(contract_path, "csvs")
    contract_logsdir = os.path.join(contract_path, "logs")
    contract_sdgdir = os.path.join(contract_path, "sdgs")

    compiled_file = os.path.join(contract_jsondir, f'{fname}.json')

    seq_file = os.path.join(contract_csvdir, f"{fname}_seq.csv")
    blocks_file = os.path.join(contract_csvdir, f"{fname}_blocks.csv")

    optimized_file = f"{fname}_optimized.json_solc"

    os.makedirs(os.path.join(contract_csvdir), exist_ok=True)
    os.makedirs(os.path.join(contract_logsdir), exist_ok=True)

    with open(compiled_file, "r") as f:
        if len(f.read()) == 0:
            return "error"

    # Run the Python optimization script with timeout
    cmd = f"source ~/gasol-optimizer/venv/bin/activate && timeout --foreground {TIMEOUT} python3 {GASOL_GREEDY_EXEC} {compiled_file} {OPTIM_PARAMS} -csv {seq_file} -block-csv {blocks_file}"
    #print(cmd)
    
    res = None
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            timeout=int(TIMEOUT),
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout"

    with open(os.path.join(contract_logsdir, "output.log"), "w") as f:
        f.write(result.stdout)

    if result.returncode == 0:
        res = "ok"
        try:
            store_results(contract_jsondir, [
                (optimized_file, "optimized.json_solc"),
            ])
            os.remove(optimized_file)
        except Exception as e:
            print(e)

        store_results(contract_sdgdir, [
            (f"{fname}_sdg.json", "sdg.json"),
        ])

    else:
        if res is None:
            res = "error"

        with open(os.path.join(contract_logsdir, "error.log"), "w") as f:
            f.write(result.stderr)

    return res



def compile_contracts(sol_files):
    tasks = []
    '''
    tasks += [
        (sol_file, False)
        for sol_file in sol_files
    ]
    '''

    tasks += [
        (sol_file, True)
        for sol_file in sol_files
    ]

    max_workers = min(mp.cpu_count(), WORKERS)
    print(f"Processing {len(tasks)} contracts with {max_workers} parallel workers ...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compile_single_contract, *task): task[0] for task in tasks}
        successful = []
        failed_contracts = []
    
        # Wait for all futures to complete and collect results
        with tqdm(total=len(tasks), desc="Compiling contracts") as pbar:
            for future in as_completed(futures):
                contract_id = futures[future]
                try:
                    result = future.result()
                    if result == "ok":
                        successful.append(contract_id)
                    else:
                        failed_contracts.append((contract_id, result))
                except Exception as e:
                    failed_contracts.append((contract_id, str(e)))
                pbar.update(1)
                pbar.set_postfix(ok=len(successful), error=len(failed_contracts))


    # Print summary
    print(f"\n=== FINAL COMPILATION SUMMARY ===")
    print(f"Total contracts commpiled: {len(tasks)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed_contracts)}")
    
    if failed_contracts:
        print(f"\nFailed contracts (first 10):")
        for block_id, error in failed_contracts[:10]:
            print(f"  {block_id}: {error}")


def optimize_contracts(sol_files):
    tasks = []
    '''
    tasks += [
        (Path(sol_file).stem, False)
        for sol_file in sol_files
    ]
    '''

    tasks += [
        (Path(sol_file).stem, True)
        for sol_file in sol_files
    ]

    max_workers = min(mp.cpu_count(), WORKERS)
    print(f"Optimizing {len(tasks)} contracts with {max_workers} parallel workers ...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(optimize_single_contract, *task): task[0] for task in tasks}
        successful = []
        failed_contracts = []
        timeout = []
    
        # Wait for all futures to complete and collect results
        with tqdm(total=len(tasks), desc="Optimizing contracts") as pbar:
            for future in as_completed(futures):
                contract_id = futures[future]
                try:
                    result = future.result()
                    if result == "ok":
                        successful.append(contract_id)
                    elif result == "timeout":
                        timeout.append(contract_id)
                    else:
                        failed_contracts.append((contract_id, result))
                except Exception as e:
                    failed_contracts.append((contract_id, str(e)))

                pbar.update(1)
                pbar.set_postfix(ok=len(successful), error=len(failed_contracts), timeout=len(timeout))


    # Print summary
    print(f"\n=== FINAL OPTIMIZATION SUMMARY ===")
    print(f"Total contracts optimized: {len(tasks)}")
    print(f"Successful: {len(successful)}")
    print(f"Timeout: {len(timeout)}")
    print(f"Failed: {len(failed_contracts)}")
    
    if failed_contracts:
        print(f"\nFailed contracts (first 10):")
        for block_id, error in failed_contracts[:10]:
            print(f"  {block_id}: {error}")


def main(filter, option):

    if option == "both":
        option = ["compile", "optimize"]
    else:
        option = [option]

    
    # Clean up SOL_DIR
    clean_up(['*.*~', '*.evm', '*.disasm'])
    
    
    # Getting and compiling the contracts
    sol_files = get_sol_files(filter)

    if len(sol_files) == 0:
        print("No contracts to be processed")
        return
    
    if "compile" in option:
        compile_contracts(sol_files)
    if "optimize" in option:
        optimize_contracts(sol_files)


    clean_up(['*.*~'])

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Obtain contract blocks: ")

    parser.add_argument("--filter",  default="0x",
                        help="contract filter 0x...")

    parser.add_argument("--action", choices=["compile", "optimize", "both"],  default="both",
                        help="Only compile the contracts, only optimize the contracts or perform both")

    parser.add_argument("--greedy", action="store_true",
                        help="apply greedy")


    args = parser.parse_args()

    OPTIM_PARAMS = "--greedy"
    
    
    main(args.filter, args.action)
