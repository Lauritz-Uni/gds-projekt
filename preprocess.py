import pandas as pd
import os
from utils.pandas_csv_reader import read_csv_file
"""
This script is used to preprocess the data.
"""

def convert_tsv_to_csv(input_file, output_file):
    columns = [
        'id', 'type', 'content', 'subjects', 'speaker', 'job_title',
        'state', 'party', 'barely_true', 'false', 'half_true',
        'mostly_true', 'pants_fire', 'context'
    ]

    df = pd.read_table("./data/"+input_file, sep="\t", names=columns)

    df.to_csv("./output/"+output_file, index=False)

def run_preprocessor(args: str):
    os.chdir("rust-preprocess")
    os.system("cargo run --release -- "+args)
    os.chdir("..")

    print("\n Preprocessing complete\n\n")

def combine_995k_and_articles_data():
    print("Combining 995,000 rows and articles data")
    orig_df = read_csv_file("data/995,000_rows.csv")
    articles_df = read_csv_file("data/articles_data.csv")

    articles_df['type'] = 'reliable'

    combined_df = pd.concat([orig_df, articles_df], ignore_index=True)[orig_df.columns]

    combined_df.to_csv("data/combined_995,000_rows.csv", index=False)

    print(orig_df['content'].isna().sum())

    print(articles_df['content'].isna().sum())

    print(combined_df['content'].isna().sum())

    print(combined_df.columns)

    print("Combined 995,000 rows and articles data saved to data/combined_995,000_rows.csv")



def main():
    
    files = [["test.tsv", "liar_test.csv"],
             ["train.tsv", "liar_train.csv"],
             ["valid.tsv", "liar_valid.csv"]]

    for file in files:
        print(f"converting {file[0]} to {file[1]}")
        convert_tsv_to_csv(file[0], file[1])

    run_preprocessor(f"--input ../output/{files[0][1]},../output/{files[1][1]},../output/{files[2][1]} --output ../output/liar_processed.csv --three-files")

    combine_995k_and_articles_data()
    run_preprocessor(f"--input ../data/combined_995,000_rows.csv --output ../output/995,000_rows_processed.csv")

    

if __name__ == "__main__":
    main()

    #combine_995k_and_articles_data()
    
    print("finished")
    print("If building the preprocessor failed at any point, try rerunning it again\n We found there are issues on windows with compiling the crates, but they will be fixed on as second run")