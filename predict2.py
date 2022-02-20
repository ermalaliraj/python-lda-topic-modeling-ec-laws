import pickle


num_topics = 20
data_dir = "./data"
year = "2016"
fileModelName = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '.pkl'
fileCampusName = './model/lda_model_EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_campus.pkl'

def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        corpus = pickle.load(f)
    return corpus


lda_model = deserializeFile(fileModelName)
bow_corpus = deserializeFile(fileCampusName)
print("Loaded Regulations model and corpus data.\n")
topics = lda_model.show_topics(num_topics=-1)
print("All topics in the model", *topics, sep="\n")
print("\n")

id2word = lda_model.id2word
# less than 15 documents (absolute number) or
# more than 0.5 documents (fraction of total corpus size, not absolute number).
# after the above two steps, keep only the first 100000 most frequent tokens.
# id2word.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# from gensim import  models
#
# tfidf = models.TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
# for doc in corpus_tfidf:
#     print("doc", *doc, sep="\n")
#     break
#
# lda_model_tfidf = models.LdaModel(corpus_tfidf, num_topics=10, id2word=id2word, passes=2)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))


unseen_document = 'How to protect data when processing information '
bow_vector = id2word.doc2bow(unseen_document.split())
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10)))





