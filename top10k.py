from collections import Counter
import code_pre_processor as cpp
import time

def wordFrequency(csv_data, content_column):
    counter = Counter()

    for document in csv_data[content_column]:
        counter.update(document)
    return counter

start_time = time.time()

processor = cpp.TextProcessor()
csv_data = processor.full_pipeline('data/news_sample.csv', 'content')

counter = wordFrequency(csv_data, 'content-tokens_no_stop')

top_words = [word for word, _ in counter.most_common(10000)]
print(top_words)

print("--- %s seconds ---" % (time.time() - start_time))