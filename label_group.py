import code_pre_processor
csv_data = code_pre_processor.read_csv_file("data/news_sample.csv")

reliable = []
fake = []

def label(csv_data, type, id):
    for index, row in csv_data.iterrows():
        if row[type] == "reliable":
            reliable.append(row[id])
        elif row[type] in ["fake", "satire", "bias", "conspiracy", "state", "junksci", "hate", "clickbait", "unreliable", "political"]:
            fake.append(row[id])
    return reliable, fake
   
reliable, fake = label(csv_data, "type", "id")
print("Id's of reliable articles:", reliable)
print("Id's of anything other than reliable articles:", fake)
        
        
        