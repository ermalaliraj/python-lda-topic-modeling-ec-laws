"""
Used for time optimisation.
Serializes all documents in a np array as:
 [
    [file1, fileContentAsListOfWords1],
    [file2, fileContentAsListOfWords2],
    ...
 ]
Deserialize "fileDocumentsArr" file for building your model.
"""

import os
import pickle
import re
import xml.etree.ElementTree as ET
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import spacy

data_dir = "./data"
year = "2016"
# year = "ALL"
fileDocumentsArr = './model/EU_REG_year-' + year + '_documentsArr.pkl'


def to_string_utf8(document):
    return document.decode('utf-8')


def get_file_content(filepath):
    tree = ET.parse(filepath)
    document = ET.tostring(tree.getroot(), encoding='utf-8', method='text')
    document = to_string_utf8(document)
    document = re.sub('[ \t\n]+', ' ', document)
    return document


def lemmatization(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    doc = nlp(text)
    new_text = []
    for token in doc:
        if token.is_stop is False and len(token.text) > 2 and token.pos_ in allowed_postags:
            new_text.append(token.lemma_.lower())
    return new_text


start = timer()
docs = []
for file in os.listdir(data_dir):
    filterYear = True
    if year != "ALL":
        filterYear = file.startswith("reg_" + year)
    if filterYear and file.endswith(".xml"):
        try:
            docs.append([file, lemmatization(get_file_content(os.path.join(data_dir, file)))])
        except:
            pass
documents = np.array(docs)
endLoad = timer()
print("Total number of documents:", len(documents), "loaded in", timedelta(seconds=endLoad - start), "seconds. (lemmatized)")

file = open(fileDocumentsArr, "wb")
pickle.dump(documents, file)
print("fileDocumentsArr saved in: ", file.name)
