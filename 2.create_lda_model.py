"""
Build LDA model with the array build in 'create_documents_array.py'

Serialize:
- LDA Model
- bow_corpus
"""
import pickle
from datetime import timedelta
from timeit import default_timer as timer

import gensim
import gensim.corpora as corpora

num_topics = 20
data_dir = "./data"
year = "2016"
# year = "ALL"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileBowCampus = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_bowcampus.pkl'
fileDocumentsArr = './model/EU_REG_year-' + year + '_documentsArr.pkl'


def deserializeFile(file_name):
    print("\nLoading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


start = timer()
documents = deserializeFile(fileDocumentsArr)
documents_words = documents[:, 1]
id2word = corpora.Dictionary(documents_words)
endLoad = timer()
print(len(documents), "documents loaded in", timedelta(seconds=endLoad - start), "seconds. Each document is a list of words inside the list 'documents_words'")
print("Dictionary 'id2word' built with 'documents_words'.  Size ", len(id2word))
print("\ndocuments_words[0][:20]:", documents_words[0][:20])
print("documents_words[1][:20]:", documents_words[1][:20])
print("documents_words[2][:20]:", documents_words[2][:20])
print("documents_words[3][:20]:", documents_words[3][:20])
print("documents_words[4][:20]:", documents_words[4][:20])

bow_corpus = [id2word.doc2bow(document) for document in documents_words]
print("\n'doc2bow' using 'id2word' built for each document and saved in the list 'bow_corpus'. Length ", len(bow_corpus))
print("bow_corpus[0][0:20]: ", bow_corpus[0][0:20])

outputFile = open(fileBowCampus, 'wb')
pickle.dump(bow_corpus, outputFile)
outputFile.close()
print("\n'bow_corpus' saved in: ", outputFile.name)

startBuildModel = timer()
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")
endBuildModel = timer()
print("\nModel built in ", timedelta(seconds=endBuildModel - startBuildModel), "seconds")

outputFile = open(fileModel, 'wb')
pickle.dump(lda_model, outputFile)
print("Model saved in: ", outputFile.name)

