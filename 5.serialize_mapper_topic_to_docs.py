"""
Build a map Topic-Documents and serialize it as a list:
    [
        [topic0, [doc1, doc2, doc3]]
        [topic1, [doc1]]
        ...
    ]

1) For each document, use the content to call the LDA model and predict the related Topic.
2) Append the document filename to the predicted topic element in the list.
"""

import pickle
from collections import defaultdict

num_topics = 20
data_dir = "./data"
year = "2016"
# year = "ALL"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileTopicToDocuments = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_topic_to_docs.pkl'
fileDocumentsArr = './model/EU_REG_year-' + year + '_documentsArr.pkl'


def deserializeFile(file_name):
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


def predictTopic(lda_model, unseen_phrase, fileName=""):
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(unseen_phrase)
    predicted_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    # print("Predicted with highest score {} - Topic {}".format(predicted_topics[0][1], predicted_topics[0][0]))
    print("File {} predicted with highest score {} - Topic {}".format(fileName, predicted_topics[0][1], predicted_topics[0][0]))
    return predicted_topics[0][0]

lda_model = deserializeFile(fileModel)
documents = deserializeFile(fileDocumentsArr)
print("Loaded Regulations model and", len(documents), "documents.\n")

# Calculate the topic probabilities for each document in the training data.
# The heights probability, is treated as the predicted Topic for the specific document.
topic_to_documents = defaultdict(list)
for doc in documents:
    prediction = predictTopic(lda_model, doc[1], doc[0])
    topic_to_documents[prediction].append(doc[0])

outputFile = open(fileTopicToDocuments, 'wb')
pickle.dump(topic_to_documents, outputFile)
print("Mapper topic_to_documents saved in: ", outputFile.name)
