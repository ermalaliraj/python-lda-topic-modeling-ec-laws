import pickle

from collections import defaultdict

num_topics = 20
data_dir = "./data"
year = "2016"
# year = "ALL"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileCampus = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_bowcampus.pkl'
fileDocumentsArr = './model/EU_REG_year-' + year + '_documentsArr.pkl'

def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


# def listClosestTopics(lda_model, unseen_document):
#     id2word = lda_model.id2word
#     unseen_document = unseen_document.split()
#     unseen_bow = id2word.doc2bow(unseen_document)
#     predicted_top_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
#     for index, score in predicted_top_topics:
#         print("Score: {}\t Topic:{} - {}".format(score, index, lda_model.print_topic(index, 20)))


def predict(lda_model, unseen_document):
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(unseen_document)
    predicted_top_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    # print("Predicted as most probable with Score: {}\n\t Topic:{} - {}".format(predicted_top_topics[0][1],
    #                                                                            predicted_top_topics[0][0],
    #                                                                            lda_model.print_topic(predicted_top_topics[0][0], 20)))
    return predicted_top_topics[0][0]


lda_model = deserializeFile(fileModel)
documents = deserializeFile(fileDocumentsArr)
print("Loaded Regulations model and documents.\n")


topic_to_document = defaultdict(list)
for doc in documents:
    pred = predict(lda_model, doc[1])
    topic_to_document[pred].append(doc[0])

totDocsDistributed = 0
print(len(documents), "documents are spread as follow:")
for topic_doc in range(len(topic_to_document)):
    nrDocsForTopic = len(topic_to_document[topic_doc])
    print("Topic", topic_doc, "is found in", nrDocsForTopic, "documents")
    totDocsDistributed += nrDocsForTopic
print("Total nr of documents distributed in ", len(topic_to_document), "topics is", totDocsDistributed)
print(len(documents) - totDocsDistributed, "documents do not belong to any topic!")




















#
# print("\n***** Prediction 1 *****")
# # GDPR phrase
# unseen_document = 'In order to ensure a consistent level of protection for natural persons throughout the Union and to prevent divergences hampering the free movement ' \
#                   'of personal data within the internal market, a Regulation is necessary to provide legal certainty and transparency for economic operators, including micro, ' \
#                   'small and medium-sized enterprises, and to provide natural persons in all Member States with the same level of legally enforceable rights and obligations and ' \
#                   'responsibilities for controllers and processors, to ensure consistent monitoring of the processing of personal data, and equivalent sanctions in all Member States ' \
#                   'as well as effective cooperation between the supervisory authorities of different Member States. The proper functioning of the internal market requires that the' \
#                   ' free movement of personal data within the Union is not restricted or prohibited for reasons connected with the protection of natural persons with regard to the ' \
#                   'processing of personal data. To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for ' \
#                   'organisations with fewer than 250 employees with regard to record-keeping. In addition, the Union institutions and bodies, and Member States and their supervisory ' \
#                   'authorities, are encouraged to take account of the specific needs of micro, small and medium-sized enterprises in the application of this Regulation. ' \
#                   'The notion of micro, small and medium-sized enterprises should draw from Article 2 of the Annex to Commission Recommendation 2003/361/EC '
#
# # listClosestTopics(lda_model, unseen_document)
# print()
# predict(lda_model, unseen_document)
