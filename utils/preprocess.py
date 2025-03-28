import pandas as pd
import os
"""
This script reads a TSV file and saves it as a CSV file.
"""

def convert_tsv_to_csv(input_file, output_file):
    columns = [
        'id', 'type', 'content', 'subjects', 'speaker', 'job_title',
        'state', 'party', 'barely_true', 'false', 'half_true',
        'mostly_true', 'pants_fire', 'context'
    ]

    df = pd.read_table("./data/"+input_file, sep="\t", names=columns)

    df.to_csv("./output/"+output_file, index=False)

def main():
    files = [["test.tsv", "liar_test.csv"],
             ["train.tsv", "liar_train.csv"],
             ["valid.tsv", "liar_valid.csv"]]

    for file in files:
        print(f"converting {file[0]} to {file[1]}")
        convert_tsv_to_csv(file[0], file[1])

    os.chdir("rust-preprocess")
    os.system(f"cargo run --release -- --input ../output/{files[0][1]},../output/{files[1][1]},../output/{files[2][1]} --output ../output/liar_processed.csv --three-files")
    os.system(f"cargo run --release -- --input ../data/995,000_rows.csv --output ../output/995,000_rows_processed.csv")
    os.system(f"cargo clean")
    os.chdir("..")

if __name__ == "__main__":
    main()
    
    print("finished")