"""
What you need is the following:
    1. Your LDA model, trained on the input documents
    2. Create a matrix with size n x k (n = number of documents, k = number of topics). We will call this matrix M
    3. In this matrix M, you put the topic probabilities for each document in the training data.
        Hereto, you iterate over all your documents and apply:
            - https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.get_document_topics
            - https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore  (if scalability is an issue)
    4. Now, for a NEW document, use the predict function to get the topic probabilities. Store this as a vector. We call it V
    5. Then, iterate over all your rows in M, and compute the Euclidean distance between the row and V.
    Use Numpy’s linalg.norm function. Example: https://www.kite.com/python/answers/how-to-find-euclidean-distance-in-python
    6. Store the Euclidean distance computed for every row in M, and return those rows (documents) (e.g. top 10) with the smallest Euclidean distance.
"""

import pickle
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=1000)

num_topics = 20
data_dir = "./data"
year = "2016"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileBowCampus = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_bowcampus.pkl'


def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


def getBowForPhrase(lda_model, unseen_document):
    unseen_document = unseen_document.split()
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(unseen_document)
    return unseen_bow


lda_model = deserializeFile(fileModel)
bow_corpus = deserializeFile(fileBowCampus)
print("Loaded Regulations model.\n")

documents_topics = np.zeros((len(bow_corpus), 20))
count = 1
docNr = 0
for bow_doc in bow_corpus:
    doc_topics = lda_model.get_document_topics(bow_doc)
    # print("document:", count, "-", document_topics)
    for doc_topic in doc_topics:
        documents_topics[docNr][doc_topic[0]] = doc_topic[1]
    count = count + 1
    docNr = docNr + 1


print("documents_topics: \n", documents_topics)


unseen_document = 'In order to ensure a consistent level of protection for natural persons throughout the Union and to prevent divergences hampering the free movement ' \
                  'of personal data within the internal market, a Regulation is necessary to provide legal certainty and transparency for economic operators, including micro, ' \
                  'small and medium-sized enterprises, and to provide natural persons in all Member States with the same level of legally enforceable rights and obligations and ' \
                  'responsibilities for controllers and processors, to ensure consistent monitoring of the processing of personal data, and equivalent sanctions in all Member States ' \
                  'as well as effective cooperation between the supervisory authorities of different Member States. The proper functioning of the internal market requires that the' \
                  ' free movement of personal data within the Union is not restricted or prohibited for reasons connected with the protection of natural persons with regard to the ' \
                  'processing of personal data. To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for ' \
                  'organisations with fewer than 250 employees with regard to record-keeping. In addition, the Union institutions and bodies, and Member States and their supervisory ' \
                  'authorities, are encouraged to take account of the specific needs of micro, small and medium-sized enterprises in the application of this Regulation. ' \
                  'The notion of micro, small and medium-sized enterprises should draw from Article 2 of the Annex to Commission Recommendation 2003/361/EC '

unseen_bow = getBowForPhrase(lda_model, unseen_document)
unseen_document_topics = lda_model.get_document_topics(unseen_bow)
print("unseen_document_topics:", unseen_document_topics)

unseen_documents_topics = np.zeros((1, 20))
for doc_topics in unseen_document_topics:
    unseen_documents_topics[0][doc_topics[0]] = doc_topics[1]
print("unseen_documents_topics:", unseen_documents_topics)

a = np.linalg.norm(unseen_documents_topics)
print("\na:", a)


res = []
countDoc = 0;
for doc_topics in documents_topics:
    # print("document:", count, "-", document_topics)
    euclidean_distance = np.linalg.norm(doc_topics - unseen_documents_topics)
    print("\nDocument", countDoc, ", doc_topics:", doc_topics)
    print("euclidean_distance:", euclidean_distance)
    # a = np.linalg.norm(unseen_documents_topics)
    res.append(["doc" + str(countDoc), a])
    countDoc = countDoc +1


print("\nres:", res)