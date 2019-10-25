from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import os
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
import string
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.initializers import Constant
from keras.utils import np_utils
from keras import backend as K


import numpy as np
import nltk

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


word_emb_weight = list()
for doc in doc_pad:
    l =[avg_emb[value] for value in doc]
    word_emb_weight.append(l)

print('finish')





def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#define model
def get_model(number_feature):
    model = Sequential()
    # embedding_layer = Embedding(num_words,300, embeddings_initializer=Constant(embedding_matrix), input_length=max_length,trainable=False)
    #model.add(embedding_layer)
    model.add(Dense(100, input_dim=number_feature, activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()
    return model

model = get_model(max_length)
train_x = np.array(word_emb_weight[:2000])
test_x = np.array(word_emb_weight[2001:])
train_y = Y[:2000]
encoded_train_y = np_utils.to_categorical(train_y)
test_y = Y[2001:]
encoded_test_y = np_utils.to_categorical(test_y)

model.fit(train_x,encoded_train_y,epochs=150, batch_size=10)
loss,accuracy,f1,pre,rec =model.evaluate(test_x,encoded_test_y)

print("Loss :",loss)
print("Accuracy : ",accuracy)
print("F1-measure: ",f1)
print("Precision: ",pre)
print("Recall: ",rec)



#sentences = brown.sents()
''''
w2v =Word2Vec(docs, size=300, window=5, min_count=5, negative=5, iter=20, workers=multiprocessing.cpu_count())
word_vector = w2v.wv
vocabulary = list(word_vector.wv.vocab)
del docs

file = 'word2vector2.txt'
w2v.wv.save_word2vec_format(file,binary=False)
del w2v
del word_vector
'''