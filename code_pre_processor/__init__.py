import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urlextract import URLExtract
from typing import Dict, List, Set

"""
This module provides a text processing pipeline for analyzing vocabulary statistics in a dataset.
The pipeline includes the following steps:
1. Field Extraction: Extract special fields like URLs, dates, emails, and numbers.
2. Field Replacement: Replace special fields with placeholders.
3. Tokenization: Tokenize text while preserving placeholders.
4. Stopword Removal: Remove stopwords from token list.
5. Stemming: Apply Porter stemming to tokens.
6. Vocabulary Analysis: Calculate vocabulary statistics like vocabulary size, stopword reduction rate, and stemming reduction rate.


How to use the code:
import code_pre_processor as cpp

processor = cpp.TextProcessor()
test_path = 'data/news_sample.csv'
processed_data = processor.full_pipeline(test_path, 'content')

analyzer = cpp.VocabularyAnalyzer()
stats = analyzer.get_vocabulary_stats(processed_data, 'content')
analyzer.print_stats(stats)
"""

# ======================
# Constants & Patterns
# ======================

PATTERNS = {
    'url': None,  # Handled by URLExtract
    'date': (
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        r'|\b\d{4}[.-]\d{1,2}[.-]\d{1,2}\b'
        r'|\b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{4}\b'
        r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{1,2},\s\d{4}\b'
        r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{1,2}\b'
        r'|\b\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b'
    ),
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'number': r'\b\d+\b'
}

PLACEHOLDERS = {
    'url': '<URL>',
    'date': '<DATE>',
    'email': '<EMAIL>',
    'number': '<NUMBER>'
}

# ======================
# File Reading Function
# ======================

def read_csv_file(file_path: str) -> pd.DataFrame:
        """Read CSV file"""
        return pd.read_csv(file_path)

# ======================
# Text Processing Class
# ======================

class TextProcessor:
    def __init__(self):
        self.extractor = URLExtract()
        self.stemmer = PorterStemmer()
        self._ensure_nltk_resources()
        
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ['punkt', 'punkt_tab', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)

    # ======================
    # Field Replacement Logic
    # ======================

    def _extract_field(self, text: str, field_type: str) -> List[str]:
        """Generic field extraction method"""
        if field_type == 'url':
            return self.extractor.find_urls(text)
        return re.findall(
            PATTERNS[field_type], 
            text, 
            flags=re.IGNORECASE if field_type == 'date' else 0
        )

    def _replace_fields(self, text: str) -> str:
        """Replace all special fields with placeholders
        
        Example:

        Input: "Hello https://www.google.com world"

        Output: "Hello \\<URL\\> world"
        """
        for field_type in PATTERNS.keys():
            matches = self._extract_field(text, field_type)
            placeholder = PLACEHOLDERS[field_type]
            for match in matches:
                text = text.replace(match, placeholder)
        return text

    def process_fields(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Add columns with extracted fields and cleaned text"""
        # Add extraction columns
        for field_type in PATTERNS.keys():
            df[f"{column}-{field_type}s"] = df[column].apply(
                lambda x: self._extract_field(x, field_type)
            )
        
        # Add cleaned text column
        df[f"{column}-cleaned"] = df[column].apply(self._replace_fields)
        return df
    
    def is_placeholder(self, token: str) -> bool:
        """Check if a token matches one of the predefined placeholders."""
        return token in PLACEHOLDERS.values()

    # ======================
    # Tokenization & Stemming
    # ======================

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving placeholders
        
        Example:

        Input: "Hello \\<URL\\> world"

        Output: ["hello", "\\<URL\\>", "world"]
        """
        if pd.isna(text):
            return []
        
        # Regex pattern to match placeholders (e.g., <URL>, <DATE>)
        placeholder_pattern = r'(<\w+>)'
        
        # Split text into parts: placeholders and other text
        parts = re.split(placeholder_pattern, text)
        
        tokens = []
        for part in parts:
            if not part:
                continue
            # Check if the part is a placeholder
            if re.fullmatch(placeholder_pattern, part):
                tokens.append(part)
            else:
                # Process non-placeholder text with word_tokenize and cleaning
                sub_tokens = word_tokenize(part)
                for token in sub_tokens:
                    # Remove punctuation and lowercase while preserving underscores
                    cleaned = re.sub(r'[^\w\s]', '', token.lower())
                    if cleaned:
                        tokens.append(cleaned)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words or self.is_placeholder(token)]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens"""
        def stem(token):
            if self.is_placeholder(token):
                return token
            return self.stemmer.stem(token)
        return [stem(token) for token in tokens]

    # ======================
    # Full Processing Pipeline
    # ======================

    def full_pipeline(self, file_path: str, target_column: str) -> pd.DataFrame:
        """Complete text processing pipeline"""
        print(f"[#] Processing {file_path} for column {target_column}")
        
        df = read_csv_file(file_path)
        df = self.process_fields(df, target_column)
        
        # Tokenization steps
        df[f"{target_column}-tokens"] = df[f"{target_column}-cleaned"].apply(self.tokenize)
        df[f"{target_column}-tokens_no_stop"] = df[f"{target_column}-tokens"].apply(self.remove_stopwords)
        df[f"{target_column}-tokens_stemmed"] = df[f"{target_column}-tokens_no_stop"].apply(self.stem_tokens)
        print(df[f"{target_column}"][1])
        print(df[f"{target_column}-tokens"][1])
        print(df[f"{target_column}-tokens_stemmed"][1])
        
        print("[!] Processing complete")
        return df

# ======================
# Vocabulary Analysis
# ======================

class VocabularyAnalyzer:
    @staticmethod
    def get_vocabulary_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Calculate vocabulary statistics"""
        vocab_sizes = {}
        
        for stage in ['tokens', 'tokens_no_stop', 'tokens_stemmed']:
            vocab = set(word for tokens in df[f"{column}-{stage}"] for word in tokens)
            vocab_sizes[stage] = len(vocab)
        
        return {
            'vocabulary_size_raw': vocab_sizes['tokens'],
            'vocabulary_size_no_stopwords': vocab_sizes['tokens_no_stop'],
            'vocabulary_size_stemmed': vocab_sizes['tokens_stemmed'],
            'stopword_reduction_rate': (
                (1 - vocab_sizes['tokens_no_stop'] / vocab_sizes['tokens']) * 100
            ),
            'stemming_reduction_rate': (
                (1 - vocab_sizes['tokens_stemmed'] / vocab_sizes['tokens_no_stop']) * 100
            )
        }

    @staticmethod
    def print_stats(stats: Dict[str, float]):
        """Print formatted vocabulary statistics"""
        for key, value in stats.items():
            if 'rate' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

# ======================
# Main Execution
# ======================

if __name__ == "__main__":
    processor = TextProcessor()
    analyzer = VocabularyAnalyzer()
    

    test_path = 'data/news_sample.csv'
    processed_data = processor.full_pipeline(test_path, 'content')


    print("[#] Calculating vocabulary statistics")    
    stats = analyzer.get_vocabulary_stats(processed_data, 'content')
    analyzer.print_stats(stats)
    print("[!] Analysis complete")