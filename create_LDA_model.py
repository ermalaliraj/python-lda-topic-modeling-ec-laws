import os
import pickle
import re
import xml.etree.ElementTree as ET
from datetime import timedelta
from timeit import default_timer as timer

import gensim
import gensim.corpora as corpora
import spacy

num_topics = 50
data_dir = "./data"
year = "2016"
fileModelName = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '.pkl'
fileCampusName = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_campus.pkl'


def to_string_utf8(document):
    return document.decode('utf-8')


def get_file_content(filepath):
    tree = ET.parse(filepath)
    document = ET.tostring(tree.getroot(), encoding='utf-8', method='text')
    document = to_string_utf8(document)
    document = re.sub('[ \t\n]+', ' ', document)
    return document


def lemmatization(documents, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    all_texts = []
    for document in documents:
        doc = nlp(document)
        new_text = []
        for token in doc:
            if token.is_stop is False and len(token.text) > 2 and token.pos_ in allowed_postags:
                new_text.append(token.lemma_.lower())
        # all_texts.append(" ".join(new_text)) # as text
        all_texts.append(new_text)
    return all_texts


start = timer()
docs = []
for filename in os.listdir(data_dir):
    # if filename.startswith("reg_" + year) and filename.endswith(".xml"):
    if filename.endswith(".xml"):
        # if filename.startswith("reg_" + year) and filename.endswith(".xml"):
        try:
            docs.append(get_file_content(os.path.join(data_dir, filename)))
        except:
            pass
            # print(os.path.join(data_dir, x))
endLoad = timer()

print("Total number of documents:", len(docs), "loaded in", timedelta(seconds=endLoad - start), "seconds")
print("doc[0] - ", docs[0][0: 150], "...")
print("doc[1] - ", docs[1][0: 150], "...")
print("doc[2] - ", docs[2][0: 150], "...")
print("doc[3] - ", docs[3][0: 150], "...")
print("doc[4] - ", docs[4][0: 150], "...")

documents_words = lemmatization(docs)
endClean = timer()
print("\nCleaned in", timedelta(seconds=endClean - endLoad), "seconds")
print("documents_words[0] cleaned - ", documents_words[0][0: 150], "...")
print("documents_words[1] cleaned - ", documents_words[1][0: 150], "...")
print("documents_words[2] cleaned - ", documents_words[2][0: 150], "...")
print("documents_words[3] cleaned - ", documents_words[3][0: 150], "...")
print("documents_words[4] cleaned - ", documents_words[4][0: 150], "...")

id2word = corpora.Dictionary(documents_words)
print("\nid2word size for full corpus: ", len(id2word))

bow_corpus = []
for document in documents_words:
    bow_doc = id2word.doc2bow(document)
    bow_corpus.append(bow_doc)
# bow_corpus = [id2word.doc2bow(doc) for doc in documents_words]
print("\ncorpus size: ", len(bow_corpus))
print("corpus[0][0:20]: ", bow_corpus[0][0:20])

outputFile = open(fileCampusName, 'wb')
pickle.dump(bow_corpus, outputFile)
outputFile.close()
print("Corpus data saved in: ", outputFile.name)

lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,  #
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")
endBuildModel = timer()
print("\nModel built in ", timedelta(seconds=endBuildModel - endClean), "seconds")



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



with open(fileModelName, 'wb') as f:
    pickle.dump(lda_model, f)
print("Model saved in: ", f.name)

topics = lda_model.show_topics(num_words=10, num_topics=-1)
print("\nAll topics in the model", *topics, sep="\n")
