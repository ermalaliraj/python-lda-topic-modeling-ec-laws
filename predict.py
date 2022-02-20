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
campus = deserializeFile(fileCampusName)
print("Loaded Regulations model and corpus data.\n")
topics = lda_model.show_topics(num_topics=-1)
print("All topics in the model", *topics, sep="\n")
print("\n")

for i, doc in enumerate(campus[:20]):
    doc_topics = lda_model[doc]
    print("doc_topics", doc_topics)
    estimate = max(doc_topics, key=lambda x: x[1])
    # print("document: ", doc)
    # print("real topic: ", topics[i])
    # print("document: ", doc)
    print("estimate topic: ", lda_model.get)
