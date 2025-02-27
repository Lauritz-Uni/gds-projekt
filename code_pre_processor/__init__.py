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

def replace_fields(csv_data: pd.DataFrame, column: str):
    # Url pattern from https://gist.github.com/gruber/8891611
    url_pattern = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"

    def new_url_column(text):
        data = re.findall(url_pattern, text)
        return data

    def remove_urls(text):
        return re.sub(url_pattern, '<URL>', text)
    

    csv_data[column+"-urls"] = csv_data[column].apply(new_url_column)
    print(f"Length of {column}-urls column: {len(csv_data[column+"-urls"])}")

    csv_data[column+"-removed_fields"] = csv_data[column].apply(remove_urls)

    print(csv_data[column+"-removed_fields"][1])

    return csv_data

# Function to clean and tokenize text
def tokenize(text):
    if pd.isna(text):
        return []
    
    # TODO: fix replace_fields such that <URL> and other fields are not tokenized
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"\W(?!<[A-Z]*>)", " ", text)  # Remove punctuation
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
    csv_data[column_to_process+"-tokens"] = csv_data[column_to_process+"-removed_fields"].apply(tokenize)

    # Remove stopwords using NLTK's stopwords list
    csv_data[column_to_process+"-tokens_no_stopwords"] = csv_data[column_to_process+"-tokens"].apply(stopword_removal)

    # Apply stemming
    csv_data[column_to_process+"-tokens_stemmed"] = csv_data[column_to_process+"-tokens_no_stopwords"].apply(lambda tokens: [stemmer.stem(word) for word in tokens])
    print(csv_data[column_to_process+"-tokens_stemmed"][1])
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