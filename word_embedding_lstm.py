from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense,InputLayer,Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import os
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
from keras.initializers import Constant

nltk.download('punkt')
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories = categories)
twenty_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42,categories = categories)
X = twenty_train.data+twenty_test.data
Y = np.concatenate((twenty_train.target,twenty_test.target),axis=0)
del twenty_train
del twenty_test



def preprocessing_dataset(dataset):
    preprocessed_docs = list()
    for data in dataset:
        tokens = word_tokenize(data)
        #convert to lowercase
        tokens = [word.lower() for word in tokens]
        #remove punctuation from each word
        #table = str.maketrans('','', string.punctuations)
        #filter out stopwords

        # remove tokens that are not alphabet
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        preprocessed_docs.append(tokens)
    return preprocessed_docs

def read_wordembedding(file_name):
    embedding_index = {}
    f = open(os.path.join('',file_name), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embedding_index[word]=coefs
    f.close()
    return embedding_index

docs = preprocessing_dataset(X)
del X

tokenizer_object = Tokenizer()
tokenizer_object.fit_on_texts(docs)
max_length = max([len(doc) for doc in docs])
max_length_index = np.argmax([len(doc) for doc in docs])
sequences = tokenizer_object.texts_to_sequences(docs)

#pad sequence
word_index = tokenizer_object.word_index
doc_pad = pad_sequences(sequences, maxlen=max_length)
word_embedding_file_name = 'word2vector2.txt'
embedding_index = read_wordembedding(word_embedding_file_name)
num_words = len(word_index)+1


def average_embedding(word_index,embedding_index):
    embedding_matrix = np.zeros((num_words, 300))
    weighted_average_emb = {}
    weighted_average_emb[0] = 0
    embedding_word = embedding_index.keys()
    len1 = len(embedding_index)
    len2 = len(word_index)

    for word, i in word_index.items():
        #if i > num_words:
         #   continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            average = np.average(embedding_vector.astype(np.float))
            embedding_matrix[i] = embedding_vector
            weighted_average_emb[i] = average
        else :
            weighted_average_emb[i] = 0
    return weighted_average_emb, embedding_matrix

avg_emb,emb_mat = average_embedding(word_index,embedding_index)



num_category = 4


def base_model():
    model = Sequential()
    embedding_layer = Embedding(num_words,300, embeddings_initializer=Constant(emb_mat), input_length=max_length,trainable=False)
    model.add(embedding_layer)

    #model.add(Embedding(len(word_index),128,input_length=max_length))
    model.add(LSTM(100))
    model.add(Dense(num_category, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.summary()
    return model

mod =base_model()

train_x =doc_pad[:2000]
test_x = doc_pad[2001:]
train_y = Y[:2000]
encoded_train_y = np_utils.to_categorical(train_y)
test_y = Y[2001:]
encoded_test_y = np_utils.to_categorical(test_y)

mod.fit(train_x,encoded_train_y,epochs=1, batch_size=10)
accuracy =mod.evaluate(test_x,encoded_test_y)
print(accuracy)