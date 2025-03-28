import pandas as pd 
"""
This script reads a TSV file and saves it as a CSV file.
"""

def convert_tsv_to_csv(input_file, output_file):
    columns = [
        'id', 'label', 'content', 'subjects', 'speaker', 'job_title',
        'state', 'party', 'barely_true', 'false', 'half_true',
        'mostly_true', 'pants_fire', 'context'
    ]

    df = pd.read_table("./data/"+input_file, sep="\t", names=columns)

    df.to_csv("./output/"+output_file, index=False)

if __name__ == "__main__":
    files = [["test.tsv", "test.csv"],
             ["train.tsv", "train.csv"],
             ["valid.tsv", "valid.csv"]]

    for file in files:
        print(f"converting {file[0]} to {file[1]}")
        convert_tsv_to_csv(file[0], file[1])
    print("finished")