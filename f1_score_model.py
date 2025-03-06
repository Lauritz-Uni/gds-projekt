import code_pre_processor
import pandas as pd
from collections import Counter
import top10k
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# List of 10.000 most frequent words in all texts
word_list = top10k.get_top_words()

# Reading csv files
fake_csv_file = code_pre_processor.read_csv_file("fake_articles_sample.csv") 
reliable_csv_file = code_pre_processor.read_csv_file("reliable_articles_sample.csv")      


# Combined files
combined_data_csv = pd.concat([fake_csv_file, reliable_csv_file])

# Function for labeling reliable as 1 and any other label as fake/0
def categorize_reliable_or_fake(content_type):
    if "reliable" == content_type.lower():
        return 1
    else:
        return 0

combined_data_csv["binary type"] = combined_data_csv["type"].apply(categorize_reliable_or_fake)


# Counts words
def count_words(text):
    words = text.split() # Tokenizes text
    return {word: words.count(word) for word in word_list if word in words}

# Counts the words of combined_data_csv
word_counts = combined_data_csv["content"].apply(count_words)

# Convert dictionary to Dataframe
x = pd.DataFrame(word_counts.tolist()).fillna(0) # Converts from NaN to 0
y = combined_data_csv["content"]

# Train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("[#] Fitting the model")
# Train the model
model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(x_train, y_train)

print("[#] Testing model")
# Predict/evaluate
y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"F1 score: {f1:.4f}")




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