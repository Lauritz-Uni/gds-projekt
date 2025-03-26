import pandas as pd 

#df = pd.read_table("data/test.tsv", sep="\t")

# Save as CSV to the output folder
#df.to_csv("output/test_converted.csv", index=False)

#Combining the liar dataset which got split by preproccessing it
df1 = pd.read_csv("data/liar_preprocessed_test.csv")
df2 = pd.read_csv("data/liar_preprocessed_val.csv")
df3 = pd.read_csv("data/liar_prerocessed_train.csv")
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

df_combined.to_csv("output/liar_test_combined.csv", index=False)