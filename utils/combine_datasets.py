import pandas as pd

def combine_csv_files(file1_path, file2_path, output_path):
    """
    Combine two CSV files into a single CSV file.
    
    Args:
    file1_path (str): Path to the first CSV file
    file2_path (str): Path to the second CSV file
    output_path (str): Path where the combined CSV will be saved
    """
    # Read the first CSV file
    df1 = pd.read_csv(file1_path)
    
    # Read the second CSV file
    df2 = pd.read_csv(file2_path)
    
    # Combine the dataframes vertically (concatenate rows)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined dataset saved to {output_path}")
    print(f"Total rows in combined dataset: {len(combined_df)}")
    print(f"Columns in combined dataset: {list(combined_df.columns)}")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    first_file = ("output/995,000_rows_train.csv")
    second_file = ("output/reduced_articles_data_processed_reliable.csv")
    output_file = ("output/combined_dataset.csv")
    
    combine_csv_files(first_file, second_file, output_file)