'''
Created on Oct 12, 2018

@author: earass
'''
import pandas as pd
import numpy as np
from numpy import asarray, zeros
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Embedding, Dropout, Dense


# importing the dataset as pandas dataframe
df = pd.read_excel('my_dataset.xlsx')

# some data cleaning
df['text'] = df['text'].apply(lambda x:x.strip('" "'))
df.drop_duplicates(subset=['text'], inplace=True)
df['text'].replace('', np.nan, inplace=True)
df['category'].replace('', np.nan, inplace=True)
df.dropna(inplace=True)
print(df.shape)

# assigning numeric labels for each unique category of text
# categories are : Environment, Sports, Technology
unique_categories = df['category'].unique().tolist()
for c in unique_categories:
    df.loc[df['category'] == c, 'label'] = unique_categories.index(c)

texts = df['text']
#labels matrix
labels = to_categorical(df['label'])

# tokenizing
tkzr = Tokenizer()
tkzr.fit_on_texts(texts)
words_dict = tkzr.word_index
vocab_size = len(words_dict) + 1 # number of unique words

# saving vocabulary 
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tkzr, handle, protocol=pickle.HIGHEST_PROTOCOL)

# integer encoding the texts, assigning integers to each text in texts
encoded_texts = tkzr.texts_to_sequences(texts)

# padding each sequence to the same length
max_len = 60
padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

# splitting dataset into train set (0.8) and test set (0.2)
X_train, X_test, y_train, y_test = train_test_split(padded_texts, labels, random_state=0, test_size=0.2)

# loading the pre-trained Glove embeddings
embeddings_index = dict()
f = open('glove.6B/glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    if words_dict.get(word): # filtering it only for the unique words in our training data
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()

# creating a weight matrix for words in training texts
embedding_matrix = zeros((vocab_size, 100))
for word, i in words_dict.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# defining model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# fiting the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# evaluating the model on test set
accuracy = model.evaluate(X_test, y_test, verbose=1)
print(accuracy[1]*100)

# saving the model
model.save('cls_model.h5')
print ('model saved')