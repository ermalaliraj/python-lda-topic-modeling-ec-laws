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
def clean(text):
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

print("documents are spread as follow:")
for topic_doc in range(len(topic_to_document)):
    print("Topic", topic_doc, "is found in", len(topic_to_document[topic_doc]), "documents")

# Generating Predictions
text = input("\nEnter text: \n")

pred = np.argmax(model.predict([clean(text)]))
print("Inserted text is similar to Topic", pred, " - \"", topics[pred], "\"")
print(len(topic_to_document[pred]), "documents containing Topic", pred, ": ", topic_to_document[pred])
