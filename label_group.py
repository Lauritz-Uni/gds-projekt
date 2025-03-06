import code_pre_processor
import pandas as pd


csv_data = code_pre_processor.read_csv_file("data/news_sample.csv")

# Lists for reliable and fake id's
reliable = []
fake = []


# Appends all id's of the files with type "Reliable" into a list and all other id's of types into a list for fake articles
def label(csv_data, type, id):
    for index, row in csv_data.iterrows():
        if row[type] == "reliable":
            reliable.append(row[id])
        elif row[type] in ["fake", "satire", "bias", "conspiracy", "state", "junksci", "hate", "clickbait", "unreliable", "political"]:
            fake.append(row[id])
    return reliable, fake


reliable, fake = label(csv_data, "type", "id")


# Filters rows based on id's
reliable_df = csv_data[csv_data["id"].isin(reliable)]
fake_df = csv_data[csv_data["id"].isin(fake)]

# Save to csv file
reliable_df.to_csv("reliable_articles_sample.csv", index=False)
fake_df.to_csv("fake_articles_sample.csv", index=False)
print("CSV-files saved")

print("Id's of reliable articles:", reliable)
print("Id's of anything other than reliable articles:", fake)
        
        
        