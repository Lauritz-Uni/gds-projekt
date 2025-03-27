import pandas as pd 
"""
This script reads a TSV file and saves it as a CSV file.
"""


df = pd.read_table("data/test.tsv", sep="\t", header=None, names=[
        'id', 'label', 'statement', 'subjects', 'speaker', 'job_title', 
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])

# Save as CSV to the output folder
df.to_csv("output/test_converted.csv", index=False)