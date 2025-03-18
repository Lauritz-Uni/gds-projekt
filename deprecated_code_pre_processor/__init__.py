import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urlextract import URLExtract
from typing import Dict, List, Tuple, Any
import time
import itertools
import json
import ast

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

COLUMNS = ['-tokens', '-tokens_no_stop', '-tokens_stemmed', '-urls', '-dates', '-emails', '-numbers']

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

def read_csv_file(file_path: str, column='content') -> pd.DataFrame:
        """Read CSV file"""
        csv_data = pd.read_csv(file_path)
        try:
            csv_data[column+'-urls'][0]
            for col in COLUMNS:
                csv_data[column+col] = csv_data[column+col].apply(ast.literal_eval)
        except KeyError:
            print("# No processed data found in the csv file")
            return None
        return csv_data

# ======================
# Text Processing Class
# ======================

class TextProcessor:
    def __init__(self):
        """Initialize text processor with required resources"""
        self.extractor = URLExtract()
        self.stemmer = PorterStemmer()
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        
        
        # Precompile regex patterns
        self.date_pattern = re.compile(PATTERNS['date'], re.IGNORECASE)
        self.email_pattern = re.compile(PATTERNS['email'])
        self.number_pattern = re.compile(PATTERNS['number'])
        self.combined_pattern = re.compile(
            rf'({PATTERNS["date"]})|({PATTERNS["email"]})|({PATTERNS["number"]})',
            re.IGNORECASE
        )
        self.placeholder_regex = re.compile(r'(<\w+>)')
        self.clean_token_regex = re.compile(r'[^\w\s]')
        
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are downloaded"""
        resources = ['punkt','punkt_tab', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)

    # ======================
    # Field Replacement Logic
    # ======================

    def _replace_and_extract(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Replace special fields with placeholders and extract them"""
        extracted = {k: [] for k in PATTERNS.keys()}
        if not text:
            return text, extracted
        
        # Extract and replace URLs
        urls = self.extractor.find_urls(text)
        extracted['url'] = urls
        for url in urls:
            text = text.replace(url, PLACEHOLDERS['url'])
        
        # Extract and replace other fields
        matches = []
        for match in self.combined_pattern.finditer(text):
            if match.group(1):
                field = 'date'
            elif match.group(2):
                field = 'email'
            elif match.group(3):
                field = 'number'
            else:
                continue
            extracted[field].append(match.group())
            matches.append((match.start(), match.end(), field))
        
        # Replace from last to first to avoid offset issues
        for start, end, field in sorted(matches, reverse=True, key=lambda x: x[0]):
            text = text[:start] + PLACEHOLDERS[field] + text[end:]
        
        return text, extracted

    def process_fields(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Process special fields in the given column"""
        results = df[column].apply(self._replace_and_extract)
        df[f"{column}-cleaned"] = results.apply(lambda x: x[0])
        extracted = results.apply(lambda x: x[1])
        for field in PATTERNS.keys():
            df[f"{column}-{field}s"] = extracted.apply(lambda e: e.get(field, []))
        return df
    
    # ======================
    # Tokenization & Stemming
    # ======================

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving placeholders"""
        if pd.isna(text):
            return []
        
        tokens = []
        parts = self.placeholder_regex.split(text)
        for i, part in enumerate(parts):
            if not part:
                continue
            if i % 2 == 1:  # Placeholder
                tokens.append(part)
            else:           # Regular text
                cleaned_part = self.clean_token_regex.sub('', part.lower())
                if cleaned_part:
                    tokens.extend(word_tokenize(cleaned_part))
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [t for t in tokens if t in PLACEHOLDERS.values() or t not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens"""
        return [t if t in PLACEHOLDERS.values() else self.stemmer.stem(t) for t in tokens]
    
    # =====================
    # Full Pipeline
    # =====================

    def full_pipeline(self, file_path: str, target_column: str, out_path=None, chunk_size = None) -> pd.DataFrame:
        """Run full text processing pipeline on the given file and column"""
        print(f"[#] Processing {file_path} for column {target_column}")
        start_time = time.time()
        
        if chunk_size is None:
            print(f"[#] Reading csv file")
            csv_start_time = time.time()
            df = pd.read_csv(file_path)
            df = self.process_fields(df, target_column)
            print(f"[!] Read csv file in {time.time() - csv_start_time:.2f}s")
            
            # Parallel processing steps
            print("[#] Running text processing pipeline")
            text_start_time = time.time()
            df[f"{target_column}-tokens"] = df[f"{target_column}-cleaned"].apply(self.tokenize)
            df[f"{target_column}-tokens_no_stop"] = df[f"{target_column}-tokens"].apply(self.remove_stopwords)
            df[f"{target_column}-tokens_stemmed"] = df[f"{target_column}-tokens_no_stop"].apply(self.stem_tokens)
            print(f"[!] Completed text processing in {time.time() - text_start_time:.2f}s")
            if out_path:
                print("saving to csv file")
                df.to_csv(out_path, index=1)
        else:
            # Chunked processing
            print(f"[#] Reading and processing in chunks of {chunk_size}")
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
            accumulated_chunks = []
            
            for i, chunk in enumerate(chunk_reader):
                print(f"[#] Processing chunk {i+1}")
                chunk = self.process_fields(chunk, target_column)
                
                # Process tokens for current chunk
                chunk[f"{target_column}-tokens"] = chunk[f"{target_column}-cleaned"].apply(self.tokenize)
                chunk[f"{target_column}-tokens_no_stop"] = chunk[f"{target_column}-tokens"].apply(self.remove_stopwords)
                chunk[f"{target_column}-tokens_stemmed"] = chunk[f"{target_column}-tokens_no_stop"].apply(self.stem_tokens)
                
                if out_path:
                    # Write chunks to disk incrementally
                    header = (i == 0)  # Write header only for first chunk
                    chunk.to_csv(out_path, mode='a' if i > 0 else 'w', header=header, index=False)
                else:
                    accumulated_chunks.append(chunk)
            
            df = pd.concat(accumulated_chunks, ignore_index=True) if not out_path else None

        
        print(f"[!] Full pipeline completed in {time.time() - start_time:.2f}s")

        return df

class VocabularyAnalyzer:
    @staticmethod
    def get_vocabulary_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Calculate vocabulary statistics for the given column"""
        vocab_sizes = {}
        for stage in ['tokens', 'tokens_no_stop', 'tokens_stemmed']:
            col = f"{column}-{stage}"
            vocab = set(itertools.chain.from_iterable(df[col]))
            vocab_sizes[stage] = len(vocab)
        
        return {
            'vocabulary_size_raw': vocab_sizes['tokens'],
            'vocabulary_size_no_stopwords': vocab_sizes['tokens_no_stop'],
            'vocabulary_size_stemmed': vocab_sizes['tokens_stemmed'],
            'stopword_reduction_rate': (1 - vocab_sizes['tokens_no_stop'] / vocab_sizes['tokens']) * 100,
            'stemming_reduction_rate': (1 - vocab_sizes['tokens_stemmed'] / vocab_sizes['tokens_no_stop']) * 100
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