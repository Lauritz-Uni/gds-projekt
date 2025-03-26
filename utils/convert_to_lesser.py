import pandas as pd

columns_to_read = ["label", "content-tokens_stemmed"]

df = pd.read_csv("output/995,000_rows_processed_test.csv", usecols=columns_to_read)

df.to_csv("output/reduced_test.csv", index=False)

df = pd.read_csv("output/995,000_rows_processed_val.csv", usecols=columns_to_read)

df.to_csv("output/reduced_val.csv", index=False)

print("finished") 