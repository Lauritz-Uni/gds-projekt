import code_pre_processor

filepath = 'data/995,000_rows.csv'
content_column = 'content'
processor = code_pre_processor.TextProcessor()
csv_data = processor.full_pipeline(filepath, content_column, out_path='data/995,000_rows_processed.csv',chunk_size=1000)