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


lda_model = deserializeFile(fileModel)
documents = deserializeFile(fileDocumentsArr)
print("Loaded Regulations model and", len(documents), "documents.\n")

# Calculate the topic probabilities for each document in the training data.
# The heights probability, is treated as the predicted Topic for the specific document.
id2word = lda_model.id2word
topic_to_documents = defaultdict(list)
for doc in documents:
    doc_bow = id2word.doc2bow(doc[1])
    document_topics = sorted(lda_model.get_document_topics(doc_bow), key=lambda tup: -1 * tup[1])   #almost same accuracy as lda_model[doc_bow]
    predictedTopic = document_topics[0][0]
    topic_to_documents[predictedTopic].append(doc[0])

outputFile = open(fileTopicToDocuments, 'wb')
pickle.dump(topic_to_documents, outputFile)
print("Mapper topic_to_documents saved in: ", outputFile.name)
