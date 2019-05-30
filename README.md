Description
===========
This is project A--Visual Question Answering developed by team Avengers composed of Renjie Zhu([@renjiezhu](https://github.com/renjiezhu)), Daoyu Li([@Nwoodle](https://github.com/Nwoodle)), Zi He and Houjian Yu
Requirements
============
Install package packages as follow:

Code organization
=================
VQA baseline_bottom_up_attention.ipynb -- Run the baseline of VQA 2017 winner and create a model file
utils.py -- Provide utilities of the project
train.py -- Module used to train the model
language_model.py -- Module implementing GRU language model
fc.py -- Module for realizing fully-connected layers
dataset.py -- Module for processing data
classifier.py -- Module for implementing forward propagation of the classifier
base_model.py -- Module for creating a baseline model
attention.py -- Module for introducing attention mechanism into the model 
