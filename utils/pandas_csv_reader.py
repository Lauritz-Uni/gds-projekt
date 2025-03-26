import pandas
import os


def read_csv_file(filepath) -> pandas.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    return pandas.read_csv(filepath)


def is_processed_file(df: pandas.DataFrame) -> bool:
    if 'content-tokens_no_stop' in df.columns:
        return True
    elif 'label' in df.columns:
        return False
    raise ValueError(f"Dataframe does not contain split or processed data.")


def is_split_file(df: pandas.DataFrame):
    if 'content-tokens_no_stop' in df.columns:
        return False
    elif 'label' in df.columns:
        return True
    raise ValueError(f"Dataframe does not contain split or processed data.")