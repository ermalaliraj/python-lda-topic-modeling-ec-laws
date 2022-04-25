import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta
from timeit import default_timer as timer

import ktrain
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

np.set_printoptions(edgeitems=30, linewidth=1000)
nltk.download('stopwords')
nltk.download('wordnet')

data_dir = "../data"
year = "2016"
fileModel = './model/ktrain_model_EU_REG_year-' + year + '.pkl'
fileTopics = './model/ktrain_model_EU_REG_year-' + year + '_topics.pkl'
fileTopicsToDocs = './model/ktrain_model_EU_REG_year-' + year + '_topics_to_docs.pkl'


def to_string_utf8(document):
    return document.decode('utf-8')


def get_file_content(filepath):
    tree = ET.parse(filepath)
    document = ET.tostring(tree.getroot(), encoding='utf-8', method='text')
    document = to_string_utf8(document)
    document = re.sub('[ \t\n]+', ' ', document)
    return document


# Cleaning Text
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    stopwordsList = set(stopwords.words("english"))
    words = text.lower().split(" ")
    cleaned_text = ""
    for word in words:
        if word in stopwordsList: continue
        cleaned_text += lemmatizer.lemmatize(word) + " "
    return cleaned_text


def save_content_to_file(content, filePath, name=""):
    file = open(filePath, "wb")
    pickle.dump(content, file)
    print(name, "SAVED in ", filePath)


# load all documents in the np array as a couple [fileName, content]
start = timer()
documents = []
for doc in os.listdir(data_dir):
    # if doc.endswith(".xml"):
    if doc.startswith("reg_" + year) and doc.endswith(".xml"):
        try:
            documents.append([doc, lemmatization(get_file_content(os.path.join(data_dir, doc)))])
        except:
            pass
documents = np.array(documents)
endLoad = timer()
print("Total number of documents (already lemmatized):", len(documents), "loaded in", timedelta(seconds=endLoad - start), "seconds")

# Building the model
model = ktrain.text.get_topic_model(documents[:, 1])
model.build(documents[:, 1], threshold=0.25)
endBuildModel = timer()
print("\nKtrain Model built in ", timedelta(seconds=endBuildModel - endLoad), "seconds")

model.print_topics(show_counts=True)
topics = model.get_topics()
topic_to_document = defaultdict(list)
for doc in documents:
    pred = model.predict([doc[1]])[0]
    found = False
    for i in range(len(pred)):
        if pred[i] >= 0.25:  # 0.25 is threshold value of similarity. Less than this is talking for different topic
            topic_to_document[i].append(doc[0])
            found = True

    # if not found:
    #     print("No Topic found for document ", doc[0], "(similarity threshold 0.25)")

print(len(documents), "\ndocuments are spread as follow:")
for topic_doc in range(len(topic_to_document)):
    print("Topic", topic_doc, "is found in", len(topic_to_document[topic_doc]), "documents")

# Saving Model
save_content_to_file(model, fileModel, "MODEL")
save_content_to_file(topics, fileTopics, "TOPICS")
save_content_to_file(topic_to_document, fileTopicsToDocs, "MAPPER TOPIC-DOCUMENTS")
