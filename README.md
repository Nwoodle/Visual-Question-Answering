# Visual-Question-Answering

> Team Avengers:
> * Renjie Zhu ([@renjiezhu](https://github.com/renjiezhu))
> * Daoyu Li ([@Nwoodle](https://github.com/Nwoodle))
> * Zi He ([@Terahezi](https://github.com/Terahezi))
> * Houjian Yu ()

This is an project about visual question answering by Tean Avengers. 

## Requirements

1. [Coco Dataset](https://visualqa.org/download.html)
    1. Training Set: (images, questions, annotations)
    2. Validation Set: (images, questions, annotations)
    3. Test Set: (images, questions, annotations)

2. [Pytorch](https://pytorch.org/get-started/locally/)

    Please follow the link for instructions for installing pytorch on your machine.

3. numpy, matplotlib

    Please install them using your preferred way of installing python packages if they don't 
already exist on the machine.

    ```pip install numpy```
    
    ```conda install numpy```
    
## Contents

- ```demo.ipynb``` - demo for our model
- ```train.ipynb``` 
- lstm_cnn
  - ```config.py``` - some global variables for paths and etc
  - ```dataloader.py``` - implemented dataloader that corresponds pictures with their questions and answers
  - ```glove.py``` - implemented a method to turn words into glove embedding
  - ```model.py``` - implemented the model used
  - ```nntools``` - provided NN tools (slightly modified)
  - ```v_preprocess.py``` - parse train data (questions and answers) to a json file
  - ```v_preprocess_val.py``` - parse validation data to a json file (answer is a list instead of a single word)
  - ```glove.6B.{dim}d.txt``` - required for ```glove.py```, change dimension as required: 50, 100 (recommended), 200, 300
  - ```train_qna.json``` - pretrained training set json file for quicker loading
  - ```val_qna_multi.json``` - pretrained validation set json file for quicker loading
  - ```vocab.json``` - indexing json file for all the vocabulary in the training and validation set
- attention
  - ...
- data
  - train2015
  - val2014
  - test2015
  - ```v2_OpenEnded_mscoco_train2014_questions.json```
  - ```v2_OpenEnded_mscoco_val2014_questions.json```
