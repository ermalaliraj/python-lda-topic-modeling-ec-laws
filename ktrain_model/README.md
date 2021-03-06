# Topic Modeling using KTrain

Cluster the documents by topics they talk about.

# Run
To run the model follow the steps:

1. Clone the repository:
```
git clone https://github.com/ermalaliraj/python-lda-topic-modeling-ec-laws.git
cd python-lda-topic-modeling-ec-laws/ktrain
```   
2. Install Requirements:
```
pip install -r requirements.txt
```
3. Train The Model:
```
python train_model.py
```
4. Run the model:
```
python predict.py
``` 
5. Enter your text. The model would predict the topic and other documents related to it.


# Visualizations
Generate visualization file with documents distributions.
```
python visualize.py
``` 

<img src="./img/ktrain_docs_distributions.png" width="80%" height="auto">
 
 
# Conclusions about Ktrain
    
    - PROS:
        - Easy to implement
    - CONS:
        - A document is not expressed as a distribution of Topics, instead as a single Topic.
    
 
### Links 
- [ktrain - doc](https://amaiya.github.io/ktrain/text/eda.html)
- [ktrain - python library](https://pythonrepo.com/repo/amaiya-ktrain-python-deep-learning)