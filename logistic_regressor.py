import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import numpy as np
from pandarallel import pandarallel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time

pandarallel.initialize(progress_bar=True, verbose=0)

def format_text(text):
    """Split into tokens."""
    return text.split()

def load_datasets(train_csv, valid_csv, test_csv):
    """
    Load datasets and preprocess labels.
    """
    print("[#] Loading datasets...")
    
    # Load datasets with efficient data types
    train_df = pd.read_csv(train_csv, dtype={'content-tokens_stemmed': str})
    valid_df = pd.read_csv(valid_csv, dtype={'content-tokens_stemmed': str})
    test_df = pd.read_csv(test_csv, dtype={'content-tokens_stemmed': str})

    print("[#] Setting y...")
    
    # Preprocess labels: 'reliable' -> 1, others -> 0
    y_train = (train_df['label'] == 'reliable').astype(np.int8)
    y_valid = (valid_df['label'] == 'reliable').astype(np.int8)
    y_test = (test_df['label'] == 'reliable').astype(np.int8)

    print("[#] Setting X...")
    
    # Handle missing values and convert to string
    train_text = train_df['content-tokens_stemmed'].fillna('').apply(format_text)
    valid_text = valid_df['content-tokens_stemmed'].fillna('').apply(format_text)
    test_text = test_df['content-tokens_stemmed'].fillna('').apply(format_text)

    return train_text, valid_text, test_text, y_train, y_valid, y_test

def optimize_model(train_text, valid_text, test_text, y_train, y_valid, y_test):
    """
    Optimize and train a logistic regression model with hyperparameter tuning.
    """
    
    print("[#] Setting up vectorizer...")
    vectorizer = CountVectorizer(
            tokenizer=lambda x: x,
            token_pattern=None,
            lowercase=False,
            binary=True,
            dtype=np.uint8,
            max_features=10000,
            min_df=5, # Ignore words that appear in less than 5 documents
            max_df=0.95 # Ignore words that appear in more than 95% of documents
        )
    
    print("[#] Vectorizing data...")
    train_text = vectorizer.fit_transform(train_text)

    # 3. Parallel transform for validation/test
    print("[#] Transforming data...")
    valid_text = vectorizer.transform(valid_text)
    test_text = vectorizer.transform(test_text)

    print("[#] Creating pipeline...")
    # Create pipeline for easier parameter tuning
    pipeline = Pipeline([
        ('classifier', LogisticRegression(
            max_iter=10000,
            random_state=42,
            class_weight='balanced',
            verbose=0
        ))
    ])
    
    # Define parameter grid for optimization
    param_grid = {
        'classifier__C': np.logspace(-1, 2, 10),  # Regularization strength
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'liblinear']
    }
    
    print("[#] Performing grid search...")
    print("[#] This may take a while...")
    
    # Use validation set for evaluation during grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,  # Use all available cores
        verbose=3
    )
    
    # Combine train and validation for more training data (optional)
    # X_combined = train_text + valid_text
    # y_combined = np.concatenate([y_train, y_valid])
    
    # Fit the model with timing
    start_time = time.time()
    grid_search.fit(valid_text, y_valid)
    print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
    
    # Evaluate on validation set
    valid_pred = grid_search.predict(valid_text)
    valid_f1 = f1_score(y_valid, valid_pred)
    
    # Print best parameters
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"- {param}: {value}")
    
    print(f"\nValidation F1 Score: {valid_f1:.4f}")
    
    # Retrain on combined train+validation data with best parameters
    print("\n[#] Retraining on combined train+validation data with best parameters...")
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(train_text, y_train)
    
    # Evaluate on test set
    test_pred = best_pipeline.predict(test_text)
    test_f1 = f1_score(y_test, test_pred)
    
    print("\nFinal Model Evaluation:")
    print(f"\nTest F1 Score: {test_f1:.4f}\n")
    print(f"{'-'*50}\nClassification Report:\n{'-'*50}")
    print(classification_report(y_test, test_pred))
    
    return best_pipeline

if __name__ == "__main__":
    default_train_csv = './output/995,000_rows_processed_train.csv'
    default_valid_csv = './output/995,000_rows_processed_val.csv'
    default_test_csv = './output/995,000_rows_processed_test.csv'
    
    # Load data
    train_text, valid_text, test_text, y_train, y_valid, y_test = load_datasets(
        default_train_csv, default_valid_csv, default_test_csv
    )
    
    # Optimize and evaluate model
    best_model = optimize_model(train_text, valid_text, test_text, y_train, y_valid, y_test)