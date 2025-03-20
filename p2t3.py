import code_pre_processor

print("test")
filepath = 'data/articles_data.csv'
content_column = 'content'
processor = code_pre_processor.TextProcessor()
csv_data = processor.full_pipeline(filepath, content_column, out_path='data/articles_data_processed.csv', chunk_size=None)
