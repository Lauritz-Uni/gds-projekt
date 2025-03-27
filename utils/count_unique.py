import pandas as pd
from collections import Counter
import dask.dataframe as dd
#Script to count unique words, to figure out the amount of features to use in the model. 

# Load dataset
print("Loading dataset...")
train_data = dd.read_csv('output/reduced_train.csv', usecols=["type"])

# Convert to Pandas
train_data = train_data.compute()

# Count unique words
print("Counting unique words...")
word_counts = Counter(" ".join(train_data["type"].astype(str)).split())

# Total unique words
total_unique_words = len(word_counts)

print(f"Total Unique Words in Dataset: {total_unique_words}")

#ca. 700,000 unique words in the fake news dataset, so therefore we chose to use 50,000 features. 
#3069 in processed liar dataset
#6060 in unprocessed liar dataset

#6 unique types in liar after preprocessing.
#6 unique types in liar before preprocessing. 