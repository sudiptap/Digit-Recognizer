# Digit-Recognizer
Kaggle challenge for recognizing handwritten digits
-----------------------------------------------------
This is an ensemble of convolutional neural networks to identify handwritten digits. The training and test datasets are available at https://www.kaggle.com/c/digit-recognizer/data. The model builds an ensemble of convolutional neurals nets using keras deep learning library. Prediction is given based on majority vote. Please note that this is a work in Progress. 

Steps to run:
------------
1. Download the training and test datasets from the following weblink : https://www.kaggle.com/c/digit-recognizer/data
2. Place all the training and test files in the same directory as ensemble_conv_net.py for smooth execution. Change the input and output paths in python code accordingly.
3. execute ensemble_conv_net.py. Output csv file will be generated in local directory and it has two columns: TestId and Label.

Improvements:
-------------
Since it is an active challenge in Kaggle right now I am changing the code almost everyday. Some of the changes to come are the following:
1. Add a json file for user to provide all input configuration for the convolutional nets.
2. Ipython notebook version of the code and some heavy commenting on the existing python file.
3. Introducing batch normalization.




