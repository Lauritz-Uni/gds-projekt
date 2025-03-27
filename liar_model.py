import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

train_path = './data/train.tsv'
val_path = './data/valid.tsv'
test_path = './data/test.tsv'

# Custom transformer to handle text columns separately
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

# Load the data
train_df = pd.read_csv(train_path, sep='\t', header=None)
val_df = pd.read_csv(val_path, sep='\t', header=None)
test_df = pd.read_csv(test_path, sep='\t', header=None)

# Assign column names
columns = [
    'id', 'label', 'statement', 'subjects', 'speaker', 'job_title',
    'state', 'party', 'barely_true', 'false', 'half_true',
    'mostly_true', 'pants_fire', 'context'
]
train_df.columns = columns
val_df.columns = columns
test_df.columns = columns

# Prepare features and target
X_train = train_df.drop(columns=['id', 'label'])
y_train = train_df['label']
X_val = val_df.drop(columns=['id', 'label'])
y_val = val_df['label']
X_test = test_df.drop(columns=['id', 'label'])
y_test = test_df['label']

# Define preprocessing for different column types
text_features = ['statement', 'subjects', 'context']
categorical_features = ['speaker', 'job_title', 'state', 'party']
numeric_features = ['barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire']

# Create separate pipelines for each text feature
text_pipelines = []
for col in text_features:
    text_pipelines.append(
        (f'tfidf_{col}', Pipeline([
            ('selector', TextSelector(col)),
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
        ]))
    )

# Combine all text features
text_transformer = FeatureUnion(text_pipelines)

# Other transformers
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine all preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features),
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])

# Create the full pipeline with a classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced'))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
print("Validation F1 Score:", f1_score(y_val, y_val_pred, average='weighted'))
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test)
print("\nTest F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
print(classification_report(y_test, y_test_pred))