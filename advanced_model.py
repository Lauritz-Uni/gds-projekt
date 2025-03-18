from sklearn.feature_extraction.text import TfidfVectorizer
import deprecated_code_pre_processor as cpp
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import get_top_words, categorize_reliable_or_fake

#using the 10.000 list
csv_data_train = cpp.read_csv_file('data/995,000_rows_processed_train.csv')
csv_data_test = read_csv_file('data/995,000_rows_processed_test.csv')
csv_data_val = read_csv_file('data/995,000_rows_processed_val.csv')
train_words = [get_top_words(csv_data_train, content+'-tokens_stemmed')]
test_words = [csv_data_test['content-tokens_stemmed']]
val_words = [csv_data_val['content-tokens_stemmed']]

#convert preprocessed text into TF-IDF weights/features, using the top 10.000 words
vectorizer = TfidfVectorizer(vocabulary=train_words) #No need for: stop_words='english' because data has been preprocessed already..
tfidf_train = vectorizer.fit_transform("train_words") #fitting model on the training data split part AND transforming it to TF-IDF
tfidf_test = vectorizer.transform("test_words") #just transforming the testing data split part into the same feature space (TF-IDF)
tfidf_val = vectorizer.transform("val_words") #transforming the validation data split part into the same feature space (TF-IDF)

#applying the categorize_reliable_or_fake function to the type column of the csv_data to extract all labels
csv_data_train["binary_type"] = csv_data_train["type"].apply(categorize_reliable_or_fake)

training_labels = csv_data_train["binary_type"].values
test_labels = csv_data_test["binary_type"].values
val_labels = csv_data_val["binary_type"].values


#training the naive bayes classifier model!
model = ComplementNB() #Naive bayes classifier, multinomial version means based on word frequency counts hence TF-IDF scores
model.fit(tfidf_train, "training_labels") #where tfidk_train is the transformed training set, and "training labels"


#testing model accuracy
test_pred = model.predict(tfidf_test)
train_pred = model.predict(tfidf_train)


print(f"Model Accuracy for training data: {accuracy_score("training_labels", train_pred)*100} %\n")
print(f"Model Accuracy for test data: {accuracy_score("test_labels", test_pred)*100} % \n\n")
print(f"Model Report: \n\n {classification_report("test_labels", test_pred)}")