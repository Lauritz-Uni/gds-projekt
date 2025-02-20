import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_csv_file(file_path):
    # Load the dataset
    csv_data = pd.read_csv(file_path)

    return csv_data

def replace_fields(csv_data, column):
    url_pattern = ''

    for document in csv_data[column]:
        pass
    re.findall(url_pattern, csv_data)

    return csv_data

# Function to clean and tokenize text
def tokenize(text):
    if pd.isna(text):
        return []
    
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Use NLTK's word_tokenize for tokenization
    
    return tokens

def stopword_removal(tokens):
    stopword_list = set(stopwords.words('english'))
    tokens_no_stopwords = []
    for word in tokens:
        if word not in stopword_list:
            tokens_no_stopwords.append(word)
    return tokens_no_stopwords


def process_text(file_path, column_to_process):
    print(f"[#] Processing {file_path} at column {column_to_process}")

    # Ensure necessary NLTK resources are available
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    csv_data = read_csv_file(file_path)

    #Remove special fields
    csv_data = replace_fields(csv_data, column_to_process)

    # Apply tokenization
    csv_data[column_to_process+"-tokens"] = csv_data[column_to_process].apply(tokenize)

    # Remove stopwords using NLTK's stopwords list
    csv_data[column_to_process+"-tokens_no_stopwords"] = csv_data[column_to_process+"-tokens"].apply(stopword_removal)

    # Apply stemming
    csv_data[column_to_process+"tokens_stemmed"] = csv_data[column_to_process+"-tokens_no_stopwords"].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

    print(f"[!] Done processing text")

    return csv_data


def get_vocabulary_size(csv_data, column):
    # Compute vocabulary size before stopword removal
    vocab_before_stopwords = set(word for tokens in csv_data[column+"-tokens"] for word in tokens)
    vocab_size_before_stopwords = len(vocab_before_stopwords)

    # Compute vocabulary size after stopword removal
    vocab_after_stopwords = set(word for tokens in csv_data[column+"-tokens_no_stopwords"] for word in tokens)
    vocab_size_after_stopwords = len(vocab_after_stopwords)

    # Compute reduction rate after stopword removal
    reduction_rate_stopwords = (1 - vocab_size_after_stopwords / vocab_size_before_stopwords) * 100

    # Compute vocabulary size after stemming
    vocab_after_stemming = set(word for tokens in csv_data[column+"-tokens_stemmed"] for word in tokens)
    vocab_size_after_stemming = len(vocab_after_stemming)

    # Compute reduction rate after stemming
    reduction_rate_stemming = (1 - vocab_size_after_stemming / vocab_size_after_stopwords) * 100

    # Display results
    results = {
        "Vocabulary Size Before Stopwords": vocab_size_before_stopwords,
        "Vocabulary Size After Stopwords": vocab_size_after_stopwords,
        "Reduction Rate After Stopwords (%)": reduction_rate_stopwords,
        "Vocabulary Size After Stemming": vocab_size_after_stemming,
        "Reduction Rate After Stemming (%)": reduction_rate_stemming,
    }

    return results

def print_vocabulary_size(results_to_print):
    for key in results_to_print:
        print(f"{key}: {results_to_print[key]}")


if __name__ == "__main__":
    test_path = 'data/news_sample.csv'

    csv_data = process_text(test_path, 'content')

    vocab_size = get_vocabulary_size(csv_data, 'content')

    print_vocabulary_size(vocab_size)