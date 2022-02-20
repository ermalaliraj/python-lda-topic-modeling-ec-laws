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

unseen_document = 'In order to ensure a consistent level of protection for natural persons throughout the Union and to prevent divergences hampering the free movement of personal data within the internal market, a Regulation is necessary to provide legal certainty and transparency for economic operators, including micro, small and medium-sized enterprises, and to provide natural persons in all Member States with the same level of legally enforceable rights and obligations and responsibilities for controllers and processors, to ensure consistent monitoring of the processing of personal data, and equivalent sanctions in all Member States as well as effective cooperation between the supervisory authorities of different Member States. The proper functioning of the internal market requires that the free movement of personal data within the Union is not restricted or prohibited for reasons connected with the protection of natural persons with regard to the processing of personal data. To take account of the specific situation of micro, small and medium-sized enterprises, this Regulation includes a derogation for organisations with fewer than 250 employees with regard to record-keeping. In addition, the Union institutions and bodies, and Member States and their supervisory authorities, are encouraged to take account of the specific needs of micro, small and medium-sized enterprises in the application of this Regulation. The notion of micro, small and medium-sized enterprises should draw from Article 2 of the Annex to Commission Recommendation 2003/361/EC'
unseen_document = unseen_document.split()
unseen_bow = id2word.doc2bow(unseen_document)

for index, score in sorted(lda_model[unseen_bow], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic:{} - {}".format(score, index, lda_model.print_topic(index, 20)))
