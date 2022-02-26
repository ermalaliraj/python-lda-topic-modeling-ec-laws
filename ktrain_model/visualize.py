import pickle

data_dir = "../data"
year = "2016"
fileModel = './model/ktrain_model_EU_REG_year-' + year + '.pkl'
fileTopics = './model/ktrain_model_EU_REG_year-' + year + '_topics.pkl'
fileTopicsToDocs = './model/ktrain_model_EU_REG_year-' + year + '_topics_to_docs.pkl'
fileTopicsVisualization = './model/ktrain_model_EU_REG_year-' + year + '_visualization.html'


def deserializeFile(file_name):
    print("Loading ", file_name)
    with open(file_name, 'rb') as f:
        file_content = pickle.load(f)
    return file_content


model = deserializeFile(fileModel)
topics = model.get_doctopics()
model.visualize_documents(doc_topics=topics, filepath=fileTopicsVisualization)
