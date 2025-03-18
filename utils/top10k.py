from collections import Counter
import time
import pandas as pd

def wordFrequency(csv_data, column):
    counter = Counter()

    print(csv_data)

    print(f"[#] Counting words {column}")
    print(f"[#] Number of documents: {len(csv_data[column])}")
    
    for i, document in enumerate(csv_data[column]):
        if type(document) != str:
            print(i, type(document))
            print(document)
            print(f"Length of csv_data: {len(csv_data[column])}")
            print(csv_data.loc[i,:])
            new_csv_data = pd.read_csv('data/995,000_rows.csv')

            for i, id in enumerate(new_csv_data["url"]):
                if id == "http://katehon.com/ru/tags/la-guerra-hibrida":
                    print(i)
                    print(new_csv_data.loc[i,:])
                    break
            raise
        counter.update(document.split(" "))
    return counter

def get_top_words(csv_data: pd.DataFrame, content: str = 'content-tokens_no_stop'):
    counter = wordFrequency(csv_data, content)

    top_words = [word for word, _ in counter.most_common(10000)]
    return top_words
    
if __name__ == "__main__":
    csv_data = pd.read_csv('data/news_sample_processed.csv')
    start_time = time.time()
    print(get_top_words(csv_data)[:20])
    print(f"Time taken to get top words: {time.time() - start_time:.2f}s")
