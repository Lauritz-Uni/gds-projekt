import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function to clean and tokenize text
def process_text(text):
    if pd.isna(text):
        return []
    
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Use NLTK's word_tokenize for tokenization
    
    return tokens

# Load the dataset
file_path = "data/news_sample.csv"
df = pd.read_csv(file_path)

# Apply tokenization
df["tokens"] = df["content"].apply(process_text)

# Compute vocabulary size before stopword removal
vocab_before_stopwords = set(word for tokens in df["tokens"] for word in tokens)
vocab_size_before_stopwords = len(vocab_before_stopwords)

# Remove stopwords using NLTK's stopwords list
stopword_list = set(stopwords.words('english'))
df["tokens_no_stopwords"] = df["tokens"].apply(lambda tokens: [word for word in tokens if word.lower() not in stopword_list])

# Compute vocabulary size after stopword removal
vocab_after_stopwords = set(word for tokens in df["tokens_no_stopwords"] for word in tokens)
vocab_size_after_stopwords = len(vocab_after_stopwords)

# Compute reduction rate after stopword removal
reduction_rate_stopwords = (1 - vocab_size_after_stopwords / vocab_size_before_stopwords) * 100

# Apply stemming
df["tokens_stemmed"] = df["tokens_no_stopwords"].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Compute vocabulary size after stemming
vocab_after_stemming = set(word for tokens in df["tokens_stemmed"] for word in tokens)
vocab_size_after_stemming = len(vocab_after_stemming)

# Compute reduction rate after stemming
reduction_rate_stemming = (1 - vocab_size_after_stemming / vocab_size_before_stopwords) * 100

# Display results
results = {
    "Vocabulary Size Before Stopwords": vocab_size_before_stopwords,
    "Vocabulary Size After Stopwords": vocab_size_after_stopwords,
    "Reduction Rate After Stopwords (%)": reduction_rate_stopwords,
    "Vocabulary Size After Stemming": vocab_size_after_stemming,
    "Reduction Rate After Stemming (%)": reduction_rate_stemming,
}

[print(result, results[result]) for result in results]

print(df['tokens_stemmed'][0])
