import pandas as pd 
"""
This script reads a TSV file and saves it as a CSV file.
"""


df = pd.read_table("data/test.tsv", sep="\t")

# Save as CSV to the output folder
df.to_csv("output/test_converted.csv", index=False)