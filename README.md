# Visual-Question-Answering

> Team Avengers:
> * Renjie Zhu ([@renjiezhu](https://github.com/renjiezhu))
> * Daoyu Li ([@Nwoodle](https://github.com/Nwoodle))
> * Zi He ([@Terahezi](https://github.com/Terahezi))
> * Houjian Yu ([@hjyu2019](https://github.com/hjyu2019))

This is an project about visual question answering by Tean Avengers. 

## Requirements

1. [Coco Dataset](https://visualqa.org/download.html)  
    1. Training Set: (images, questions, annotations)
    2. Validation Set: (images, questions, annotations)
    3. Test Set: (images, questions, annotations)  
    We used the dataset at DSMLP.
    

2. [Pytorch](https://pytorch.org/get-started/locally/)

    Please follow the link for instructions for installing pytorch on your machine.

3. numpy, matplotlib

    Please install them using your preferred way of installing python packages if they don't 
already exist on the machine.

    ```pip install numpy```
    
    ```conda install numpy```
    
4. [Model files](https://drive.google.com/file/d/1XJpnBiQ6ZNIZ3ODosAvCqfsOtOvFvLo-/view?usp=sharing)

    These files contain the trained model and they are too large to be included in a github repository.
    Please download, unzip and put the three files into folder data (refer to contents below).

    Included files include:
    1. ```eval_dset.pkl```
    2. ```model.pkl```
    3. ```model.pth```
   
5. [GloVe](http://nlp.stanford.edu/data/glove.6B.zip)

    Download and extract all ```*.txt``` in ```lstm_cnn``` folder.

## Contents

- ```DEMO.ipynb``` - **Demo for our model**
- ```TRAIN.ipynb``` - **trainning process of our model**
- ```utils.py``` - Provide utilities of the project
- ```train.py``` - Module used to train the model
- ```language_model.py``` - Module implementing GRU language model
- ```fc.py``` - Module for realizing fully-connected layers
- ```dataset.py``` - Module for processing data
- ```classifier.py``` - Module for implementing forward propagation of the classifier
- ```base_model.py``` - Module for creating a baseline model
- ```attention.py``` - Module for introducing attention mechanism into the model 
- lstm_cnn *for comparison*
  - ```v_preprocess.py``` - parse train data (questions and answers) to a json file **Run first for preprocessing vocabulary**
  - ```v_preprocess_val.py``` - parse validation data to a json file (answer is a list instead of a single word) **Run first for preprocessing vocabulary**
  - ```train_demo.ipynb``` - **training demo of lstm_cnn**
  - ```config.py``` - some global variables for paths and etc
  - ```dataloader.py``` - implemented dataloader that corresponds pictures with their questions and answers
  - ```glove.py``` - implemented a method to turn words into glove embedding
  - ```model.py``` - implemented the model used
  - ```nntools.py``` - provided NN tools (slightly modified)
  - ```glove.6B.{dim}d.txt``` - required for ```glove.py```, change dimension as required: 50, 100 (recommended), 200, 300
  - ```train_qna.json``` - pretrained training set json file for quicker loading
  - ```vocab.json``` - indexing json file for all the vocabulary in the training and validation set  
- data
  - ```eval_dset.pkl``` (not included in the repo)
  - ```model.pkl``` (not included in the repo)
  - ```model.pth``` (not included in the repo)
  - ```val36_imgid2idx.pkl```

- ```/datasets/ee285f-public/VQA2017``` (Located at DSMLP)
  - train2015 (not included in the repo)
  - val2014 (not included in the repo)
  - test2015 (not included in the repo)
  - ```v2_OpenEnded_mscoco_train2014_questions.json``` (not included in the repo)
  - ```v2_OpenEnded_mscoco_val2014_questions.json``` (not included in the repo)