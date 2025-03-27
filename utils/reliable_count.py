
import pandas as pd

# Read the file
df = pd.read_csv(r'C:\Users\theod\Documents\GitHub\gds-projekt\data\995,000_rows.csv')

# Count the number of 'reliable' types
reliable_count = (df['type'] == 'reliable').sum()

# Print just the count as an integer
print(reliable_count)