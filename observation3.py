#An observation on the top 40 words in the file, across all different articles, and texts. 

import csv
import re
import matplotlib.pyplot as plt
from collections import Counter
import code_pre_processor as cpp

def wordFrequency(csv_data, content_column):
    counterList = []

    for document in csv_data[content_column]:
        wordCounter = Counter()
        for word in document:
            wordCounter.update([word])
        counterList.append(wordCounter)
    return counterList

processor = cpp.TextProcessor()
csv_data = processor.full_pipeline('data/news_sample.csv', 'content')

counterList = wordFrequency(csv_data, 'content-tokens_no_stop')

mergedCounter = Counter()
for counter in counterList:
    mergedCounter.update(counter)

recurring_words = {word: count for word, count in mergedCounter.items() if count > 1}

print("Most Common Recurring Words Across Documents:")
for word, count in mergedCounter.most_common(40):
    print(f"{word}: {count}")

# Plot top 20 recurring words
plt.figure(figsize=(12, 5))
words, counts = zip(*mergedCounter.most_common(40))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 40 Recurring Words Across Documents")
plt.ylabel("Count")
plt.show()
