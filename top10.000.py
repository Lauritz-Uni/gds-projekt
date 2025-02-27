from collections import Counter
import code_pre_processor as cpp
import time

def wordFrequency(csv_data, content_column):
    counterList = []

    for document in csv_data[content_column]:
        wordCounter = Counter()
        for word in document:
            wordCounter.update([word])
        counterList.append(wordCounter)
    return counterList

start_time = time.time()

processor = cpp.TextProcessor()
csv_data = processor.full_pipeline('data/news_sample.csv', 'content')

counterList = wordFrequency(csv_data, 'content-tokens_no_stop')

mergedCounter = Counter()
for counter in counterList:
    mergedCounter.update(counter)

top_words = {word for word, _ in mergedCounter.most_common(10000)}
top_words = list(top_words)[:100]
print(top_words)

print("--- %s seconds ---" % (time.time() - start_time))