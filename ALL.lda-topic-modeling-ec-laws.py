"""
All the logic in a single file.

1. Load all documents from filesystem, cleans the xml tags, lemmatize the content, and create the array 'documents' as follows:
    [
        [fileName1, fileContent1],
        [fileName1, fileContent2],
        ...
    ]

2. Build the LDA model:
    documents_words = documents[:, 1]
    id2word = corpora.Dictionary(documents_words)
    bow_corpus = [id2word.doc2bow(document) for document in documents_words]
    lda_model = LdaModel(corpus=bow_corpus, id2word=id2word)

3. Calculate the topic probabilities for each document in the training data. The heights probability, is treated as the predicted Topic for the specific document.
After iterating all the documents we build the mapper 'topic_to_documents' as follow:
    [
        [topic0, [docName1, docName2, docName3]]
        [topic1, [docName4]]
        ...
    ]

4. Make a prediction with an unseen phrase.
The phrase is taken from GDPR regulation and the Topic is correctly found together with the filename 'reg_2016_679_akn_nr119seq0001.xml'
"""

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta
from timeit import default_timer as timer

import gensim
import gensim.corpora as corpora
import numpy as np
import spacy

data_dir = "./data"
year = "2016"
# year = "ALL"
num_topics = 20
fileDocumentsArr = './model/EU_REG_year-' + year + '_documentsArr.pkl'


def to_string_utf8(document):
    return document.decode('utf-8')


def get_file_content(filepath):
    tree = ET.parse(filepath)
    document = ET.tostring(tree.getroot(), encoding='utf-8', method='text')
    document = to_string_utf8(document)
    document = re.sub('[ \t\n]+', ' ', document)
    return document


def lemmatization(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    doc = nlp(text)
    new_text = []
    for token in doc:
        if token.is_stop is False and len(token.text) > 2 and token.pos_ in allowed_postags:
            new_text.append(token.lemma_.lower())
    return new_text


start = timer()
docs = []
for file in os.listdir(data_dir):
    filterYear = True
    if year != "ALL":
        filterYear = file.startswith("reg_" + year)
    if filterYear and file.endswith(".xml"):
        try:
            docs.append([file, lemmatization(get_file_content(os.path.join(data_dir, file)))])
        except:
            pass
documents = np.array(docs)
endLoad = timer()
print("Total number of documents:", len(documents), "loaded in", timedelta(seconds=endLoad - start), "seconds. (lemmatized)")

documents_words = documents[:, 1]  # only the content
print("\ndocuments_words[0][:20]:", documents_words[0][:20])
print("documents_words[1][:20]:", documents_words[1][:20])
print("documents_words[2][:20]:", documents_words[2][:20])
print("documents_words[3][:20]:", documents_words[3][:20])
print("documents_words[4][:20]:", documents_words[4][:20])

id2word = corpora.Dictionary(documents_words)
print("\nDictionary 'id2word' built with 'documents_words'.  Size ", len(id2word))

bow_corpus = [id2word.doc2bow(document)
              for document in documents_words]
print("\n'doc2bow' using 'id2word' built for each document and saved in the list 'bow_corpus'. Length ", len(bow_corpus))
print("bow_corpus[0][0:20]: ", bow_corpus[0][0:20])

startBuildModel = timer()

documents_words = documents[:, 1]  # only the content
id2word = corpora.Dictionary(documents_words)
bow_corpus = [id2word.doc2bow(document)
              for document in documents_words]
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

topics = lda_model.show_topics(num_topics=-1)
print("All topics in the model", *topics, sep="\n")


def predictTopic(lda_model, unseen_phrase, fileName=""):
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(unseen_phrase)
    predicted_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    print("File {} predicted with highest score {} - Topic {}".format(fileName, predicted_topics[0][1], predicted_topics[0][0]))
    return predicted_topics[0][0]

print("\nPredicting ALL files:")
topic_to_documents = defaultdict(list)
for doc in documents:
    prediction = predictTopic(lda_model, doc[1], doc[0])
    topic_to_documents[prediction].append(doc[0])

print("\nTopic distribution over the documents")
totDocsDistributed = 0
for topic_doc in range(len(topic_to_documents)):
    nrDocsForTopic = len(topic_to_documents[topic_doc])
    print("Topic", topic_doc, "is found in", nrDocsForTopic, "documents")
    totDocsDistributed += nrDocsForTopic
print("Total nr of documents distributed in", len(topic_to_documents), "topics is", totDocsDistributed)

print("\n***** Prediction 1 *****")
# GDPR phrase
unseen_phrase = 'In order to ensure a consistent level of protection for natural persons throughout the Union and to prevent divergences hampering the free movement ' \
                  'of personal data within the internal market, a Regulation is necessary to provide legal certainty and transparency for economic operators, including micro, ' \
                  'small and medium-sized enterprises, and to provide natural persons in all Member States with the same level of legally enforceable rights and obligations and ' \
                  'responsibilities for controllers and processors, to ensure consistent monitoring of the processing of personal data, and equivalent sanctions in all Member States ' \
                  'as well as effective cooperation between the supervisory authorities of different Member States. The proper functioning of the internal market requires that the' \
                  ' free movement of personal data within the Union is not restricted or prohibited for reasons connected with the protection of natural persons with regard to the ' \
                  'processing of personal data. To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for ' \
                  'organisations with fewer than 250 employees with regard to record-keeping. In addition, the Union institutions and bodies, and Member States and their supervisory ' \
                  'authorities, are encouraged to take account of the specific needs of micro, small and medium-sized enterprises in the application of this Regulation. ' \
                  'The notion of micro, small and medium-sized enterprises should draw from Article 2 of the Annex to Commission Recommendation 2003/361/EC '

predictedTopic = predictTopic(lda_model, unseen_phrase.split())
print("unseen phrase: ", unseen_phrase)
print("Predicted Topic", predictedTopic, "-", lda_model.print_topic(predictedTopic))
print("List of documents containing Topic", predictedTopic, "-", topic_to_documents[predictedTopic])

end = timer()
print("\n\nApplication executed in :", timedelta(seconds=end - start), "seconds")
