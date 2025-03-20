import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import numpy as np

default_train_csv = './output/reduced_train.csv'
default_valid_csv = './output/reduced_val.csv'
default_test_csv = './output/reduced_test.csv'

def train_logistic_regressor(train_csv, valid_csv, test_csv):
    """
    Train a logistic regression model on the given CSV files and
    evaluate its performance on the test set.
    """
    print("[#] Loading datasets...")

    # Load datasets
    train_df = pd.read_csv(train_csv, dtype={'content-tokens_stemmed': str})
    # valid_df = pd.read_csv(valid_csv, dtype={'content-tokens_stemmed': str})
    test_df = pd.read_csv(test_csv, dtype={'content-tokens_stemmed': str})

    print("[#] Training model...")

    # Preprocess labels: 'reliable' -> 1, others -> 0
    y_train = (train_df['label'] == 'reliable').astype(int)
    # y_valid = (valid_df['label'] == 'reliable').astype(int)
    y_test = (test_df['label'] == 'reliable').astype(int)

    print("[#] Vectorizing texts...")

    # Build vocabulary from training data and transform texts
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(),
        token_pattern=None,
        lowercase=False,
        max_features=10000,
        binary=True
    )

    print("[#] Fitting vectorizer...")

    X_train = vectorizer.fit_transform(train_df['content-tokens_stemmed'].apply(lambda x: np.str_(x)))
    # X_valid = vectorizer.transform(valid_df['content-tokens_stemmed'].apply(lambda x: np.str_(x)))
    X_test = vectorizer.transform(test_df['content-tokens_stemmed'].apply(lambda x: np.str_(x)))

    print("[#] Fitting model...")

    # Train logistic regression model with optimized hyperparameters
    model = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("[#] Testing model...")

    # Predict on test set and evaluate
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)

    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nHyperparameters Used:")
    print(f"- solver: {model.get_params()['solver']}")
    print(f"- penalty: {model.get_params()['penalty']}")
    print(f"- C: {model.get_params()['C']}")
    print(f"- max_iter: {model.get_params()['max_iter']}")
    print(f"- random_state: {model.get_params()['random_state']}")

    class_report = classification_report(y_test, y_pred)

    print(f"\n{"-"*50}\nClassfication Report:\n{class_report}")


if __name__ == "__main__":
    train_logistic_regressor(default_train_csv, default_valid_csv, default_test_csv)