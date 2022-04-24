# Machine Learning - LDA Topic Modelling for EC Regulations/Laws

Building a `machine learning` algorithm for predicting to which Topic a Document belongs to.
`LDA (Linear Discriminant Analysis)` technique will be used for classifying documents into topics and prediction. <br/>
Documents are downloaded from the Official Journal of the European Commission using this [project](https://github.com/ermalaliraj/eur-lex-official-journal-sparql).

### Algorithm

1. Load all documents from filesystem, cleans the xml tags, lemmatize the content, and create the array `documents` as follows:
    ```
        [
            [fileName1, fileContent1],
            [fileName1, fileContent2],
            ...
        ]
    ```
2. Build the LDA model:
    ```
    documents_words = documents[:, 1]
    id2word = corpora.Dictionary(documents_words)
    bow_corpus = [id2word.doc2bow(document) for document in documents_words]
    lda_model = LdaModel(corpus=bow_corpus, id2word=id2word)
    ```
3. Calculate the topic probabilities for each document in the training data. The heights probability, is treated as the predicted Topic for the specific document.
After iterating all the documents we build the mapper `topic_to_documents` as follow:
    ```
    [
        [topic0, [docName1, docName2, docName3]]
        [topic1, [docName4]]
        ...
    ]
    ```

4. Make a prediction with an unseen phrase.
The phrase is taken from GDPR regulation and the `Topic 4` is correctly found together with the filename `reg_2016_679_akn_nr119seq0001.xml`

The file containing all the steps in sequence is `ALL.lda-topic-modeling-ec-laws.py`

### Project Structure

For optimising the time we split the computation in 5 different files. 
Ex, we don't need to wait 3 minutes each time we rebuild the model for loading all xml files in a `np array`. 
Instead we can reuse the np array already saved in the filesystem using `spacy` library.

The project is split in the following files: 

##### Load file in NP array
`serialize_documents_array.py`
    - Loads all documents, reads fileContent, cleans xml, lemmatize, and creates the array `documents` with rows [fileName, fileContent] for each document.
    - OUTPUT: serialize the array `documents` in the filesystem. 
    - EXAMPLES: different serialization files are created for regulations of year 2016, 2017, 2018, 2019 2020 and ALL. <br/>
    For regulations of the year 2016 use the file `./model/EU_REG_year-2016_documentsArr.pkl`. 

##### LDA Model
`serialize_model.py`
    - Loads the array `documents` from the filesystem and calculates the followings:
        - `documents_words = documents[:, 1]`
        - `id2word = corpora.Dictionary(documents_words)`
        - `bow_corpus = [id2word.doc2bow(document) for document in documents_words]`
        - `lda_model = LdaModel(corpus=bow_corpus, id2word=id2word)`
    - OUTPUT: serializes in filesystem:
        - `bow_corpus`, Bag of Words for all documents.
        - `lda_model`, the model.
    - EXAMPLES: different models and the  related bow_corpus for regulations of 2016, 2017, 2018, 2019 2020 and ALL are created in the folder `./model`. 

##### Topics distribution visualisation
`3.visualize.py`
    - Visualises topics distribution of the model. 
    - OUTPUT: Html file with interactive visualisation. Ex `./model/EU_REG_year-2016_nrtopics20_visualize.html`
    
##### Topic Prediction
`4.predict_topic.py`
    - Tests the model using an unseen phrase and print the predicted Topic closer to the phrase. <br/>
    See [`predict.out`](./out/predict.out) with all predictions. Or full [`logs`](./out/console.out)
    - OUTPUT: Print in the console the result of 4 predictions.

##### Mapper Topic-to-Document
`5.serialize_mapper_topic_to_docs.py`
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

##### Document prediction
`6.predict_documents.py`
    - Tests the model using an unseen phrase and print the topic closer to the phrase, together with the list of documents containing that topic. 
    - OUTPUT: Print in the console the result of 4 predictions.
    
    
##### Visualisation - Topics distribution using gensimvis
<img src="./img/laws_2016_20topics.png" width="100%" height="auto">

Download in your local and consult the interactive [visualisation file](./model/EU_REG_year-2016_nrtopics20_visualize.html)


### Part-of-speech (POS) tagging for GDPR regulation 
<img src="./img/gdpr_pos.png" width="80%" height="auto">


### Links
- [Get regulations from EUR-lex - Dataset](https://github.com/ermalaliraj/eur-lex-official-journal-sparql)
- [Topic Modelling by Félix Revert](https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc)
- [Topic Modelling by Susan Li](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
- [LDAModel](https://radimrehurek.com/gensim/models/ldamodel.html)
- [Dictionary and BoW](https://radimrehurek.com/gensim/corpora/dictionary.html)
- [Topic model using ktrain library](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
- [My Topic modelling using ktrain](./ktrain_model)
- [My BigData And AI Portfolio](https://github.com/ermalaliraj/bigdata_and_ai)
