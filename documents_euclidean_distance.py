import pickle

num_topics =20
data_dir = "./data"
year = "2016"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileBowCampus = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_bowcampus.pkl'


def deserializeFile(file_name):
    print("\nLoading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


lda_model = deserializeFile(fileModel)
bow_corpus = deserializeFile(fileBowCampus)
print("Loaded Regulations model.\n")

topics = lda_model.get_document_topics(bow_corpus)
print("bow_corpus: ", topics)
for bow_doc in bow_corpus:
    topics = lda_model.get_document_topics(bow_doc)
    print("bow_doc: ", bow_doc)


# //design a distance measure document to topic
# euclidian distance of these vectors, and give that as a feedback
# document to document


# Jochen Steps
# 1) train LDA model over my 100 regulations
#  doc to topic metrix (vectors) out of my lda
#     for every single document, form 100 regulations i get my topic probability (using get_document_topics)
#       we have a matrix with 100 vectors of probabilities##
# go to my unseen document
# and let LDA predict the topic probability
# iterate all over 100 regulations and compute euclidian dinstance with my vector

