import pandas as pd

def convert_to_lesser():
    columns_to_read = ["label", "content-tokens_stemmed"]

    df = pd.read_csv("output/995,000_rows_processed_train.csv", usecols=columns_to_read)

    df.to_csv("output/reduced_train.csv", index=False)

    df2 = pd.read_csv("output/995,000_rows_processed_test.csv", usecols=columns_to_read)

    df2.to_csv("output/reduced_test.csv", index=False)

    df3 = pd.read_csv("output/995,000_rows_processed_val.csv", usecols=columns_to_read)

    df3.to_csv("output/reduced_val.csv", index=False)

    print("finished") 


    columns_to_read_liar = ["type","content-tokens_stemmed"]

    df1 = pd.read_csv("output/liar_test.csv", usecols=columns_to_read_liar)

    df1.to_csv("output/reduced_liar.csv", index=False)

convert_to_lesser()