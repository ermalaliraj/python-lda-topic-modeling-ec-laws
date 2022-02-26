"""
Test the model.
1) Use LDA model to predict the closest topic to the unseen phrase.

See 'serialize_mapper_topic_to_docs.py' to list the documents containing the predicted topic.
"""

import pickle

num_topics = 20
data_dir = "./data"
year = "2016"
# year = "ALL"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileBowCampus = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_bowcampus.pkl'


def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


def listClosestTopics(lda_model, unseen_document):
    id2word = lda_model.id2word
    unseen_document = unseen_document.split()
    unseen_bow = id2word.doc2bow(unseen_document)
    predicted_top_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    for index, score in predicted_top_topics:
        print("Score: {}\t Topic:{} - {}".format(score, index, lda_model.print_topic(index, 20)))


def predict(lda_model, unseen_document):
    print("unseen phrase: ", unseen_document)
    id2word = lda_model.id2word
    unseen_document = unseen_document.split()
    unseen_bow = id2word.doc2bow(unseen_document)
    predicted_top_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    print("Predicted as most probable with Score: {}\n\t Topic:{} - {}".format(predicted_top_topics[0][1],
                                                                               predicted_top_topics[0][0],
                                                                               lda_model.print_topic(predicted_top_topics[0][0], 20)))


lda_model = deserializeFile(fileModel)
print("Loaded Regulations model.\n")
topics = lda_model.show_topics(num_topics=-1)
print("All topics in the model", *topics, sep="\n")

print("\n***** Prediction 1 *****")
# GDPR phrase
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

listClosestTopics(lda_model, unseen_document)
print()
predict(lda_model, unseen_document)

print("\n***** Prediction 2 *****")
unseen_document = 'clinical research health'
predict(lda_model, unseen_document)

print("\n***** Prediction 3 *****")
unseen_document = 'Hybrid electric vehicles (HEVs), plug-in hybrid electric vehicles (PHEVs), and all-electric vehicles (EVs) typically produce lower tailpipe emissions than ' \
                  'conventional vehicles do. When measuring well-to-wheel emissions, the electricity source is important: for PHEVs and EVs, part or all of the power provided by ' \
                  'the battery comes from off-board sources of electricity. There are emissions associated with the majority of electricity production'
predict(lda_model, unseen_document)

print("\n***** Prediction 4 *****")
unseen_document = 'new startup entrepreneur industry loan funds'
predict(lda_model, unseen_document)