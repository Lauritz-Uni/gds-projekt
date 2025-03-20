# Now you can import the module
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils import top10k
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def get_top_words(csv_data, column="content-tokens_stemmed", top_n=10_000):
    counter = Counter()

    for document in csv_data[column].dropna():  # Ignore NaN values
        words = document.split()  # Assume words are space-separated
        counter.update(words)

    top_words = [word for word, _ in counter.most_common(top_n)]
    return top_words

"""

train_data = pd.read_csv("data/995,000_rows_processed_train.csv")
test_data = pd.read_csv("data/995,000_rows_processed_test.csv")

# List of 10.000 most frequent words in all texts
top_words = top10k.get_top_words(train_data)
##
"""
# Function for labeling reliable as 1 and any other label as fake/0
def categorize_reliable_or_fake(content_type):
    if isinstance(content_type, float):
        return 0
    return 1 if str(content_type).lower() == "reliable" else 0

"""
train_data["binary type"] = train_data["type"].apply(categorize_reliable_or_fake)
test_data["binary type"] = test_data["type"].apply(categorize_reliable_or_fake)

text_column = "content-tokens_stemmed"

train_data[text_column] = train_data[text_column].fillna("")
test_data[text_column] = test_data[text_column].fillna("")

vectorizer = CountVectorizer(vocabulary=top_words)
x_train = vectorizer.fit_transform(train_data[text_column])
x_test = vectorizer.transform(test_data[text_column])

y_train = train_data["binary type"]
y_test = test_data["binary type"]

model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(x_train, y_train)

print("[#] Testing model")
y_pred = model.predict(x_test)

print(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.4f}") """


""" # Counts the words of combined_data_cs
word_counts = big_data["content"].apply(count_words)

# Convert dictionary to Dataframe
x = pd.DataFrame(word_counts.tolist()).fillna(0) # Converts from NaN to 0
y = big_data["content-tokens_stemmed"]

 # Train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

print("[#] Fitting the model")
# Train the model
model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(y, big_data["binary type"])

print("[#] Testing model")
# Predict/evaluate
y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"F1 score: {f1:.4f}") """




""" fake_words = " ".join(fake_csv_file["content"]).split()
reliable_words = " ".join(reliable_csv_file["content"]).split()

fake_word_counts = Counter(word for word in fake_words if word in word_list)
reliable_word_counts = Counter(word for word in reliable_words if word in word_list)



print("Fake word counts")
for word, count in fake_word_counts.items():
    print(f"{word}: {count}")
    
print("Reliable word counts")
for word, count in reliable_word_counts.items():
    print(f"{word}: {count}") """


# Define file paths
train_file = "data/995,000_rows_processed_train.csv"
test_file = "data/995,000_rows_processed_test.csv"

# Load and preprocess dataset
def process_chunks(file):
    chunk_size = 100_000  # Load in chunks to save memory
    chunks = []
    
    for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
        chunk = chunk[chunk["type"].apply(lambda x: isinstance(x, str))]  # Remove NaNs and floats
        chunk["binary type"] = chunk["type"].apply(lambda x: 1 if x.lower() == "reliable" else 0)
        chunk["content-tokens_stemmed"] = chunk["content-tokens_stemmed"].fillna("")
        chunks.append(chunk)

    return pd.concat(chunks)

def main():
    train_data = process_chunks(train_file)
    test_data = process_chunks(test_file)

    #**Extract Top 10,000 Words From Training Data**
    top_words = get_top_words(train_data, column="content-tokens_stemmed")

    # **Use These Words in CountVectorizer**
    vectorizer = CountVectorizer(vocabulary=top_words)
    x_train = vectorizer.fit_transform(train_data["content-tokens_stemmed"])
    x_test = vectorizer.transform(test_data["content-tokens_stemmed"])

    # Labels
    y_train = train_data["binary type"]
    y_test = test_data["binary type"]

    # Train the model
    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    print(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.4f}")


if __name__ == "__main__":
    main()