from collections import Counter
import code_pre_processor as cpp
import time

def wordFrequency(csv_data, content_column):
    counter = Counter()

    for document in csv_data[content_column]:
        counter.update(document)
    return counter

def get_top_words(filepath: str = 'data/news_sample.csv', content: str = 'content'):
    processor = cpp.TextProcessor()
    csv_data = processor.full_pipeline(filepath, 'content')

    counter = wordFrequency(csv_data, content+'-tokens_no_stop')

    top_words = [word for word, _ in counter.most_common(10000)]
    return top_words
    
if __name__ == "__main__":
    start_time = time.time()
    print(get_top_words()[:20])
    print(f"Time taken to get top words: {time.time() - start_time:.2f}s")
