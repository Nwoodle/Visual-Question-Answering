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

Usage
=================

The current version only uses 1/4 of the whole dataset from VQA due to limitation of resources.

Just open VQA baseline_bottom_up_attention.ipynb and it will helps you go through the entire process of training the model. It is originally built on DSMLP platform of UCSD. To accomodate a different environment, please go to the first line and change the root_dir where you put the raw data from VQA (please download via this link: https://visualqa.org/download.html).
