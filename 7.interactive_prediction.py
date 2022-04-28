"""
Test the model.
1) Use LDA model to predict the closest topic to the unseen phrase.
2) Fetch from 'topic_to_documents' (built in 'create_mapper_topic_to_docs.py') the related documents to predicted topic.
"""

import pickle

num_topics = 20
data_dir = "./data"
year = "2016"
# year = "ALL"
fileModel = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_model.pkl'
fileTopicToDocuments = './model/EU_REG_year-' + year + '_nrtopics' + str(num_topics) + '_topic_to_docs.pkl'


def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


def predictTopic(lda_model, unseen_phrase):
    unseen_phrase = unseen_phrase.split()
    id2word = lda_model.id2word
    unseen_bow = id2word.doc2bow(unseen_phrase)
    predicted_topics = sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1])
    return predicted_topics[0][0]


lda_model = deserializeFile(fileModel)
topic_to_documents = deserializeFile(fileTopicToDocuments)

topics = lda_model.show_topics(num_topics=-1)
print("All topics in the model", *topics, sep="\n")

while True:
    unseen_phrase = input("\nEnter text: ")
    if unseen_phrase == 'exit':
        print("Good bye.")
        break

    predictedTopic = predictTopic(lda_model, unseen_phrase)
    print("unseen phrase: ", unseen_phrase)
    print("Predicted Topic {} contained in {} documents".format(predictedTopic, len(topic_to_documents)))
    print("Topic {}:".format(lda_model.print_topic(predictedTopic)))
    print("Documents: {} ".format(topic_to_documents[predictedTopic]))
