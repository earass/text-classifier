This text classifier is build using a simple artificial neural network. The code is written in Python 3 using the Keras package with TensorFlow backend.

The Keras embedding layer learned from the pre-learned GloVe embeddings. The GloVe embeddings can be downloaded from the following link: https://nlp.stanford.edu/projects/glove/

The model classifies texts (works best with news headlines) into three categories i.e. Environment, Sports, Technology

The dataset was compiled from different sources and was split into 80/20 ratio for training/testing

After evaluating on the test set, 94.3% accuracy was achieved

A web app is made in Python Flask framework to get the predictions

