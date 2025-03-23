import time
start_time = time.time() #Starting timer
print("Program started")

import sys
sys.path.append(".")
import modin.pandas as pd
import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

print("test")

from utils.f1_score_model import categorize_reliable_or_fake

print("test")

pandarallel.initialize(progress_bar=True, nb_workers=4)

print("Pandarrell initialized, reading files")

#reading files with dask, using reduced data-sets to lessen the load for the script (look at conver_to_lesser.py)
train_data = dd.read_csv('output/reduced_train.csv', usecols=["label", "content-tokens_stemmed"])
test_data = dd.read_csv('output/reduced_test.csv', usecols=["label", "content-tokens_stemmed"])
val_data = dd.read_csv('output/reduced_val.csv', usecols=["label", "content-tokens_stemmed"])

print("files read")

print("Converting to pandas dataframe")

# Convert to Pandas
train_data = train_data.compute()
test_data = test_data.compute()
val_data = val_data.compute()

# Convert preprocessed text into TF-IDF features. 50.000 features since there are 700.000 unique words (look at count_unique.py)
vectorizer = TfidfVectorizer(max_features=50_000)

print("applying vectorizor")

tfidf_train = vectorizer.fit_transform(train_data['content-tokens_stemmed'].astype(str))
tfidf_test = vectorizer.transform(test_data["content-tokens_stemmed"].astype(str))
tfidf_val = vectorizer.transform(val_data["content-tokens_stemmed"].astype(str))

print("assignming labels")

# Parallelize label assignment with pandarallel
train_data["binary_type"] = train_data["label"].parallel_apply(categorize_reliable_or_fake)
test_data["binary_type"] = test_data["label"].parallel_apply(categorize_reliable_or_fake)
val_data["binary_type"] = val_data["label"].parallel_apply(categorize_reliable_or_fake)

print("Done with assigning labels, converting labels to numpy arrays")

# Convert labels to numpy arrays
training_labels = train_data["binary_type"].to_numpy()
test_labels = test_data["binary_type"].to_numpy()
val_labels = val_data["binary_type"].to_numpy()

print("Training Model")

# Train Naive Bayes Classifier
model = ComplementNB()
model.fit(tfidf_train, training_labels)

print("Predicting values")

# Test Model
test_pred = model.predict(tfidf_test)
train_pred = model.predict(tfidf_train)

print("evaluating model")

# Evaluate Model
print(f"Model Accuracy for training data: {accuracy_score(training_labels, train_pred) * 100:.2f} %")
print(f"Model Accuracy for test data: {accuracy_score(test_labels, test_pred) * 100:.2f} %")
print(f"Model Report:\n\n{classification_report(test_labels, test_pred)}")

end_time = time.time() #ending timer
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time:.2f} seconds")