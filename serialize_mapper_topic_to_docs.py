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
    print("\nLoading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


lda_model = deserializeFile(fileModel)
documents = deserializeFile(fileDocumentsArr)
print("Loaded Regulations model and documents.\n")

topic_to_documents = defaultdict(list)
for doc in documents:
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(doc[1])
    predicted_top_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    predictedTopic = predicted_top_topics[0][0]
    topic_to_documents[predictedTopic].append(doc[0])

totDocsDistributed = 0
print(len(documents), "documents are spread as follow:")
for topic_doc in range(len(topic_to_documents)):
    nrDocsForTopic = len(topic_to_documents[topic_doc])
    print("Topic", topic_doc, "is found in", nrDocsForTopic, "documents")
    totDocsDistributed += nrDocsForTopic
print("Total nr of documents distributed in", len(topic_to_documents), "topics is", totDocsDistributed)
print(len(documents) - totDocsDistributed, "documents do not belong to any topic! Strange!")

outputFile = open(fileTopicToDocuments, 'wb')
pickle.dump(topic_to_documents, outputFile)
print("Mapper topic_to_documents saved in: ", outputFile.name)
