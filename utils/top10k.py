from collections import Counter
import time
import pandas as pd

def wordFrequency(csv_data, column):
    counter = Counter()
    for document in csv_data[column]:
        if isinstance(document, str):
            words = document.split()
        elif isinstance(document, list):
            words = document
        else:
            continue
        counter.update(words)
    return counter

def get_top_words(csv_data: pd.DataFrame, content: str = 'content'):
    counter = wordFrequency(csv_data, content+'-tokens_stemmed')

    top_words = [word for word, _ in counter.most_common(10000)]
    return top_words
    
if __name__ == "__main__":
    csv_data = pd.read_csv('data/news_sample_processed.csv')
    start_time = time.time()
    print(get_top_words(csv_data)[:20])
    print(f"Time taken to get top words: {time.time() - start_time:.2f}s")
