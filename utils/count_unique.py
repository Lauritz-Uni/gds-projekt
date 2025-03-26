import pandas as pd
from collections import Counter
import dask.dataframe as dd

# Load dataset
print("Loading dataset...")
train_data = dd.read_csv('output/reduced_train.csv', usecols=["content-tokens_stemmed"])

# Convert to Pandas
train_data = train_data.compute()

# Count unique words
print("Counting unique words...")
word_counts = Counter(" ".join(train_data["content-tokens_stemmed"].astype(str)).split())

# Total unique words
total_unique_words = len(word_counts)

print(f"Total Unique Words in Dataset: {total_unique_words}")