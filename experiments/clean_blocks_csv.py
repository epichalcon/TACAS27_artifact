'''
Eliminates duplicated blocks
'''
import pandas as pd

df = pd.read_csv("statsdir/blocks.csv", sep=";")
df = df.drop_duplicates(subset="previous_solution", keep="first")
df.to_csv("statsdir/blocks_cleaned.csv", sep=";", index=False)
