import code_pre_processor
import pandas as pd
from collections import Counter


# List of 10.000 most frequent words in all texts
word_list = {"word", "list"}                                                                      #put list here

# Reading csv files
fake_csv_file = code_pre_processor.read_csv_file("fake_articles_sample.csv")            #replace path with fake text path       
reliable_csv_file = code_pre_processor.read_csv_file("reliable_articles_sample.csv")     #replace path with reliable text path       




fake_words = fake_csv_file.split()
reliable_words = reliable_csv_file.split()

fake_word_counts = Counter(word for word in fake_words if word in word_list)
reliable_word_counts = Counter(word for word in reliable_words if word in word_list)

for word, count in fake_word_counts.items():
    print(f"{word}: {count}")

for word, count in reliable_word_counts.items():
    print(f"{word}: {count}")