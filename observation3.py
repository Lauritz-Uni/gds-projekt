#An observation on the top 20 words in the file, across all different articles, and texts. 

import csv
import re
import matplotlib.pyplot as plt
from collections import Counter
import code_pre_processor

def wordFrequency(csv_data, content_column):
    wordCount = Counter()
    print(csv_data[content_column][0])
    for document in csv_data[content_column]:
        pass


csv_data = code_pre_processor.process_text('./data/news_sample.csv', 'content')
wordFrequency(csv_data, 'content-tokens_no_stopwords')