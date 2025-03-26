import pandas as pd 

df = pd.read_table("data/test.tsv", sep="\t")

# Save as CSV to the output folder
df.to_csv("output/test_converted.csv", index=False)