#prepare blocks
python3 process_contracts.py --action both
python3 aggregate_compiled_blocks.py
python3 clean_blocks_csv.py
python3 apply_max_sat.py
python3 aggregate_max_sat_blocks.py

#apply model and compile results
python3 generate_sdgs.py --pre-dir pre_optimized_2024 --post-dir tmp 
rm -r sdgs/tmp
python3 evaluate_sdgs.py --input pre_optimized_2024 --output-csv statsdir/2024_contracts.csv
#python3 scripts/summarize_comparison.py --input statsdir/2024_contracts.csv --output statsdir/2024_contracts_summary.csv

#Check sat results
#python3 scripts/summarize_sat_comparison.py --input statsdir/2024_contracts.csv --output statsdir/2024_sat_summary.csv

#Validate model optimality with sat
python3 apply_max_sat_temp_to_check_optimality.py
python3 aggregate_max_sat_blocks_check_for_optimality.py
#python3 scripts/summarize_sat_csv.py --input statsdir/splitted_blocks_check_for_optimality/sat.csv --output statsdir/sat_summary_check_for_optimality.csv

python3 unify_statsdir_csv.py --cleaned --sat --model statsdir/2024_contracts.csv --optimal --output global_2024.csv

python3 scripts/summarize --output report_2024.csv