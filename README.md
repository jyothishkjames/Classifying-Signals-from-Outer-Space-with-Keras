# Classifying-Signals-from-Outer-Space-with-Keras


Getting Started
---------------

This repository implements a deep learning model to classify different radio signals from
outer space. The different radio signals include "squiggle", "narrowband", "noise" and
"narrowbanddrd".


Prerequisites
-------------

    pandas
    numpy
    matplotlib
    tensorflow
    livelossplot


Dataset
-------

Here, we use the SETI Dataset. The dataset is a csv file which encodes information of
images (spectrogram of audio signal). 

How to train the model?
-----------------------

Go to the folder data and run the following commands to pre-process the data

    python process_data.py
    
Go to the folder model and run the following command to train the model    

    python train_classifier.py


Summary
-------

The deep learning model attains a test accuracy of **74.37 %** on classification.
