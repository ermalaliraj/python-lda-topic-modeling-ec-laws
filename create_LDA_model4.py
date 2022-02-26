import pickle
from datetime import timedelta
from timeit import default_timer as timer

import gensim
import gensim.corpora as corpora

num_topics = 50
data_dir = "./data"
year = "2016"
fileModel = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '.pkl'
fileCampus = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_campus.pkl'
fileDocumentsArr = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_documentsArr_str.pkl'


def deserializeFile(file_name):
    print("\nLoading ", file_name)
    with open(file_name, 'rb') as f:
        corpus = pickle.load(f)
    return corpus


start = timer()
documents = deserializeFile(fileDocumentsArr)
endLoad = timer()
print("Total number of documents:", len(documents), "loaded in", timedelta(seconds=endLoad - start), "seconds. (lemmatized)")
print("doc[0] - ", documents[0][0: 150], "...")
print("doc[1] - ", documents[1][0: 150], "...")
print("doc[2] - ", documents[2][0: 150], "...")
print("doc[3] - ", documents[3][0: 150], "...")
print("doc[4] - ", documents[4][0: 150], "...")

corpus_words = documents[:, 1]
corpus_words = [x.split() for x in documents[:, 1]]
# df['doc'] = df['documentContent'].apply(lambda x: nlp(x))

id2word = corpora.Dictionary(corpus_words)
print("\nid2word size for full corpus: ", len(id2word))

bow_corpus = []
for document in corpus_words:
    bow_doc = id2word.doc2bow(document)
    bow_corpus.append(bow_doc)
# bow_corpus = [id2word.doc2bow(doc) for doc in corpus_words]

print("bow_corpus[0][0:20]: ", bow_corpus[0][0:20])

outputFile = open(fileCampus, 'wb')
pickle.dump(bow_corpus, outputFile)
outputFile.close()
print("Corpus data saved in: ", outputFile.name)

startBuildModel = timer()
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,  #
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")
endBuildModel = timer()
print("\nModel built in ", timedelta(seconds=endBuildModel - startBuildModel), "seconds")

bow_corpus = []
for bow_doc in bow_corpus:
    topics = lda_model.get_document_topics(document)

# //design a distance measure document to topic
# euclidian distance of these vectors, and give that as a feedback
# document to document

# 1) train LDA model over my 100 regulations
#  doc to topic metrixs (vectors) out of my lda
#     for every single document, form 100 regulations i get my topic probability (using get_document:topics)
#       we have a matrix with 100 vectors of probabilities##
# go to my uiinseen document
# and let LDA predict the topic probability
# iterate all over 100 regulations and compute euclidian dinstance with my vector


with open(fileModel, 'wb') as f:
    pickle.dump(lda_model, f)
print("Model saved in: ", f.name)

topics = lda_model.show_topics(num_words=10, num_topics=-1)
print("\nAll topics in the model", *topics, sep="\n")
