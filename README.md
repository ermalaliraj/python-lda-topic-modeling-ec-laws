# Machine Learning - LDA Topic Modelling for EC Regulations/Laws

Building a `machine learning` algorithm for predicting to which Topic a Document belongs to.
`LDA (Linear Discriminant Analysis)` technique will be used for classifying documents into topics and prediction. <br/>
Documents are downloaded from the Official Journal of the European Commission using this [project](https://github.com/ermalaliraj/eur-lex-official-journal-sparql).

### Project Structure
The project is split in the following files: 
1. `serialize_documents_array.py`
    - Loads all documents, reads fileContent, cleans xml, lemmatize, and creates the array `documents` with rows [fileName, fileContent] for each document.
    - OUTPUT: serialize the array `documents` in the filesystem. 
    - EXAMPLES: different serialization files are created for regulations of year 2016, 2017, 2018, 2019 2020 and ALL. <br/>
    For regulations of the year 2016 use the file `./model/EU_REG_year-2016_documentsArr.pkl`. 
2. `serialize_model.py`
    - Loads the array `documents` from the filesystem and calculates the followings:
        - `documents_words = documents[:, 1]`
        - `id2word = corpora.Dictionary(documents_words)`
        - `bow_corpus = [id2word.doc2bow(document) for document in documents_words]`
        - `lda_model = LdaModel(corpus=bow_corpus, id2word=id2word)`
    - OUTPUT: serializes in filesystem:
        - `bow_corpus`, Bag of Words for all documents.
        - `lda_model`, the model.
    - EXAMPLES: different models and the related bow_corpus for regulations of 2016, 2017, 2018, 2019 2020 and ALL are created in the folder `./model`. 
3. `visualize.py`
    - Visualises topics distribution of the model. 
    - OUTPUT: Html file with interactive visualisation. Ex `./model/EU_REG_year-2016_nrtopics20_visualize.html`
4. `predict.py`
    - Tests the model using an unseen phrase and print the predicted Topic closer to the phrase. <br/>
    See [`predict.out`](./out/predict.out) with all predictions. Or full [`logs`](./out/console.out)
    - OUTPUT: Print in the console the result of 4 predictions.
5. `serialize_mapper_topic_to_docs.py`
    - Builds a mapper Topic-Documents expressed through a list as follows. 
        1) For each document, use its content to predict the related topic.
        2) Appends the document filename to the topic value in the list:
           ``` 
            [   
                [topic0, [docName1, docName2, docName3]]
                [topic1, [docName1]]
                ...
            ] 
            ```
    - OUTPUT: Mapper `EU_REG_year-2016_nrtopics20_topic_to_docs.pkl`
6. `predict_documents.py`
    - Tests the model using an unseen phrase and print the topic closer to the phrase, together with the list of documents containing that topic. 
    - OUTPUT: Print in the console the result of 4 predictions.
    
    
### Visualisation - Topics distribution using gensimvis
<img src="./img/laws_2016_20topics.png" width="100%" height="auto">

Download in your local and consult the interactive [visualisation file](./model/EU_REG_year-2016_nrtopics20_visualize.html)


### Part-of-speech (POS) tagging for GDPR regulation 
<img src="./img/gdpr_pos.png" width="80%" height="auto">


### Links
- [Topic model using ktrain library](./ktrain_model)
 