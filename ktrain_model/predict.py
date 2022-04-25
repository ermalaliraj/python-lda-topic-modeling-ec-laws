import pickle

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data_dir = "../data"
year = "2016"
fileModel = './model/ktrain_model_EU_REG_year-' + year + '.pkl'
fileTopics = './model/ktrain_model_EU_REG_year-' + year + '_topics.pkl'
fileTopicsToDocs = './model/ktrain_model_EU_REG_year-' + year + '_topics_to_docs.pkl'


# Cleaning Text
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    stopwordList = set(stopwords.words("english"))
    words = text.lower().split(" ")
    cleaned_text = ""
    for word in words:
        if word in stopwordList: continue
        cleaned_text += lemmatizer.lemmatize(word) + " "
    return cleaned_text


print("Loading Model and Topics...")
model = pickle.load(open(fileModel, "rb"))
topic_to_document = pickle.load(open(fileTopicsToDocs, "rb"))
topics = pickle.load(open(fileTopics, "rb"))

model.print_topics(show_counts=True)

print("\ndocuments are spread as follow:")
for topic_doc in range(len(topic_to_document)):
    print("Topic", topic_doc, "is found in", len(topic_to_document[topic_doc]), "documents")

# Generating Predictions
while True:
    unseen_phrase = input("\nEnter text: ")
    print("unseen phrase: ", unseen_phrase)
    if unseen_phrase == 'exit':
        print("Good bye.")
        break

    pred = np.array(model.predict([lemmatization(unseen_phrase)])[0])
    for ind in range(len(pred)):
        if pred[ind] >= 0.25:  # 0.25 is threshold value for similarity.
            print("Topic {} - {}".format(ind, topics[ind]))
            print("{} Similar Documents -> {}".format(len(topic_to_document), topic_to_document[ind]))
