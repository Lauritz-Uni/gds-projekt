from sklearn.feature_extraction.text import TfidfVectorizer
import code_pre_processor as cpp
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from top10k import get_top_words
from f1_score_model import categorize_reliable_or_fake

#using the 10.000 list
csv_data = cpp.read_csv_file('data/news_sample_processed.csv')
top_words = [get_top_words(csv_data)[:10000]]

#convert preprocessed text into TF-IDF weights/features, using the top 10.000 words
vectorizer = TfidfVectorizer(vocabulary=top_words) #No need for: stop_words='english' because data has been preprocessed already..
tfidf_train = vectorizer.fit_transform("training_data") #fitting model on the training data split part AND transforming it to TF-IDF
tfidf_test = vectorizer.transform("testing_data") #just transforming the testing data split part into the same feature space (TF-IDF)
tfidf_val = vectorizer.transform("validation_data") #transforming the validation data split part into the same feature space (TF-IDF)

#applying the categorize_reliable_or_fake function to the type column of the csv_data to extract all labels
csv_data["binary_type"] = csv_data["type"].apply(categorize_reliable_or_fake)

training_labels = csv_data_train["binary_type"].values
test_labels = csv_data_test["binary_type"].values
val_labels = csv_data_val["binary_type"].values


#training the naive bayes classifier model!
model = MultinomialNB() #Naive bayes classifier, multinomial version means based on word frequency counts hence TF-IDF scores
model.fit(tfidf_train, "training_labels") #where tfidk_train is the transformed training set, and "training labels"