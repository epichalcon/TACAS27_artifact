

#Training ------------------------

python3 compiler_distribution.py --input ../../compilation_options_20260609.csv --output ../pngs/training_compiler_distribution.png

python3 optimality.py --input ../../statsdir/training_global.csv --output ../pngs/training_optimality.png --letter "(a) Block outcomes by original length"

python3 times.py --input ../../statsdir/training_global.csv --output ../pngs/training_times.png --legend --letter "(b) Time by original Length"


#Experiments----------------------

python3 joiner.py --first_csv  ../../statsdir/september_global_2.csv --second_csv ../../statsdir/global_2024.csv --output_csv ../../statsdir/joined_global.csv



python3 compiler_distribution.py --input ../../compilation_options_20260922.csv --input2  ../../compilation_options_20240226.csv --output ../pngs/joined_compiler_distribution.png


python3 times.py --input ../../statsdir/joined_global.csv --output ../pngs/joined_times.png --bigger-letters --letter "(a) SuperStack time"
python3 times.py --input ../../statsdir/joined_global.csv --time-column model_time --output ../pngs/ml_times.png --bigger-letters --letter "(b) G2S-EVML time"


python3 optimality.py --input ../../statsdir/joined_global.csv --output ../pngs/joined_optimality.png --letter "(c) SuperStack outcome"
python3 ml_optimality.py --input ../../statsdir/joined_global.csv --output ../pngs/ml_optimality.png --letter "(d) G2S-EVML outcome"


#Sat check ----------------------

python3 ml_sat_optimality.py --input ../../statsdir/joined_global.csv --output ../pngs/ml_sat_optimality.png --letter "(a) Model optimality-check outcom"
python3 times.py --input ../../statsdir/joined_global.csv --time-column optimality_check_time --output ../pngs/ml_sat_times.png --letter "(b) Model optimality-check time"