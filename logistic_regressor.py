import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, verbose=0)

"""
If ran directly, change input files at the bottom of the script.
"""

def format_text(text):
    """
    Split into tokens.
    """
    return text.split()

def load_datasets(train_csv, valid_csv, test_csv):
    """
    Train a logistic regression model on the given CSV files and
    evaluate its performance on the test set.
    """
    print("[#] Loading datasets...")

    # Load datasets
    train_df = pd.read_csv(train_csv, dtype={'content-tokens_stemmed': str})
    valid_df = pd.read_csv(valid_csv, dtype={'content-tokens_stemmed': str})
    test_df = pd.read_csv(test_csv, dtype={'content-tokens_stemmed': str})

    print("[#] Setting y...")

    # Preprocess labels: 'reliable' -> 1, others -> 0
    y_train = (train_df['label'] == 'reliable').astype(int)
    y_valid = (valid_df['label'] == 'reliable').astype(int)
    y_test = (test_df['label'] == 'reliable').astype(int)

    print("[#] Setting X...")

    # Handle missing values efficiently and convert to string
    train_text = train_df['content-tokens_stemmed'].parallel_apply(format_text)
    valid_text = valid_df['content-tokens_stemmed'].parallel_apply(format_text)
    test_text = test_df['content-tokens_stemmed'].parallel_apply(format_text)

    return train_text, valid_text, test_text, y_train, y_valid, y_test


def train_logistic_regressor(train_text, valid_text, test_text, y_train, y_valid, y_test):
    """
    Train a logistic regression model on the given CSV files and
    evaluate its performance on the test set.
    """
    print("[#] Creating vectorizer...")
    # Build vocabulary from training data and transform texts
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,  # No-op, expects pre-split lists
        token_pattern=None,
        lowercase=False,
        max_features=10000,
        binary=True,
        dtype=np.uint8
    )

    print("[#] Fitting vectorizer...")

    X_train = vectorizer.fit_transform(train_text)
    X_valid = vectorizer.transform(valid_text)
    X_test = vectorizer.transform(test_text)

    print("[#] Creating model...")

    # Train logistic regression model with optimized hyperparameters
    model = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=1.0,
        max_iter=1000,
        random_state=42,
        verbose=True
    )

    print("[#] Fitting model...")

    model.fit(X_train, y_train)

    print("[#] Testing model...")

    # Predict on test set and evaluate
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)

    print("\nVectorizer Hyperparameters Used:")
    print(f"- min_df: {vectorizer.get_params()['min_df']}")
    print(f"- max_features: {vectorizer.get_params()['max_features']}")
    print(f"- binary: {vectorizer.get_params()['binary']}")
    print(f"- lowercase: {vectorizer.get_params()['lowercase']}")
    print(f"- token_pattern: {vectorizer.get_params()['token_pattern']}")
    print(f"- dtype: {vectorizer.get_params()['dtype']}")

    print("\nModel hyperparameters Used:")
    print(f"- solver: {model.get_params()['solver']}")
    print(f"- penalty: {model.get_params()['penalty']}")
    print(f"- C: {model.get_params()['C']}")
    print(f"- max_iter: {model.get_params()['max_iter']}")
    print(f"- random_state: {model.get_params()['random_state']}")
    print(f"- verbose: {model.get_params()['verbose']}")

    print()

    print(f"\nTest F1 Score: {test_f1:.4f}\n")

    print(f"{"-"*50}\nClassification Report:\n{'-'*50}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    default_train_csv = './output/995,000_rows_processed_train.csv'
    default_valid_csv = './output/995,000_rows_processed_val.csv'
    default_test_csv = './output/995,000_rows_processed_test.csv'

    train_text, valid_text, test_text, y_train, y_valid, y_test = load_datasets(default_train_csv, default_valid_csv, default_test_csv)
    train_logistic_regressor(train_text, valid_text, test_text, y_train, y_valid, y_test)