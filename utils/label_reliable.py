import pandas as pd

def add_label_column(input_file, output_file):
    """
    Add a 'label' column with 'reliable' value to the input CSV file.
    
    Args:
    input_file (str): Path to the input CSV file
    output_file (str): Path where the new CSV with label column will be saved
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add 'label' column with 'reliable' for all rows
    df['label'] = 'reliable'
    
    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"File processed successfully!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns in file: {list(df.columns)}")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = r'data\articles_data_processed.csv'
    output_file = r'output\articles_data_processed_reliable.csv'
    
    add_label_column(input_file, output_file)