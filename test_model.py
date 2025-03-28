import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
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

def optimize_model(train_text, valid_text, test_text, y_train, y_valid, y_test, is_995k=True):
    """
    Optimize and train a RandomForest model with hyperparameter tuning.
    """
    
    print("[#] Setting up vectorizer...")
    vectorizer = CountVectorizer(
            lowercase=True,
            binary=True,
            dtype=np.uint8,
            max_features=10000,
            min_df=5, # Ignore words that appear in less than 5 documents
            max_df=0.95 # Ignore words that appear in more than 95% of documents
        )
    
    if is_995k:
        print("[*] Using custom tokenizer since input is 955k rows...")
        vectorizer.set_params({'tokenizer':lambda x: x, 'token_pattern':None})
    else:
        print("[*] Using default tokenizer...")
    
    print("[#] Vectorizing data...")
    train_text = vectorizer.fit_transform(train_text)

    # Parallel transform for validation/test
    print("[#] Transforming data...")
    valid_text = vectorizer.transform(valid_text)
    test_text = vectorizer.transform(test_text)

    print("[#] Creating pipeline...")
    # Create pipeline for easier parameter tuning
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        ))
    ])

    print("\n[#] Running test fit with default parameters...")
    try:
        test_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=30,
            verbose=2
        )
        start_time = time.time()
        test_model.fit(train_text, y_train)
        test_pred = test_model.predict(test_text)
        test_f1 = f1_score(y_test, test_pred)
        print(f"Test fit completed in {time.time() - start_time:.2f} seconds")
        print(f"Test F1 score: {test_f1:.4f}")
        print(f"{'-'*50}\nTest fit classification Report:\n{'-'*50}")
        print(classification_report(y_test, test_pred))
        print("Test fit successful! Proceeding with grid search...")
    except Exception as e:
        print(f"\n[!] Error during test fit: {str(e)}")
        print("[!] Fix the error before proceeding with grid search!")
        return None
    
    # Define parameter grid for optimization
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
        'classifier__max_depth': [10, 20, 30],  # Maximum depth of the tree
        'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'classifier__max_features': ['sqrt', 'log2']  # Number of features to consider at every split
    }
    
    print("[#] Performing grid search...")
    
    # Use validation set for evaluation during grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,  # Use all available cores
        verbose=3
    )
    
    # Fit the model with timing
    start_time = time.time()
    grid_search.fit(train_text, y_train)
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
    best_pipeline.fit(np.concatenate([train_text, valid_text]), np.concatenate([y_train, y_valid]))
    
    # Evaluate on test set
    test_pred = best_pipeline.predict(test_text)
    test_f1 = f1_score(y_test, test_pred)
    
    print("\nFinal Model Evaluation:")
    print(f"\nTest F1 Score: {test_f1:.4f}\n")
    print(f"{'-'*50}\nClassification Report:\n{'-'*50}")
    print(classification_report(y_test, test_pred))
    
    return best_pipeline

if __name__ == "__main__":
    default_train_csv = './output/train.csv'
    default_valid_csv = './output/val.csv'
    default_test_csv = './output/test.csv'
    
    # Load data
    train_text, valid_text, test_text, y_train, y_valid, y_test = load_datasets(
        default_train_csv, default_valid_csv, default_test_csv
    )
    
    # Optimize and evaluate model
    best_model = optimize_model(train_text, valid_text, test_text, y_train, y_valid, y_test)