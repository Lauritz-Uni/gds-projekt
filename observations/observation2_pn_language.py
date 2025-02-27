import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download vader lexicon
nltk.download('vader_lexicon')


# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Read text file
file_path = ["Your are a good boy", "Hello you are stupid"]  # Change this to file path

# Analyzing and printing
for i, text in enumerate(file_path):
    sentiment_scores = sia.polarity_scores(text)
    
    # Printing
    print(f"Text {i+1} sentiment score: ", sentiment_scores)
    print("-" * 50)


