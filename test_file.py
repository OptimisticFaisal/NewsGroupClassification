from gensim.models import Word2Vec
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import os
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
import string
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import nltk

nltk.download('brown')


sentences = brown.sents()
sen = sentences[3:5]

w2v =Word2Vec(sen, size=10, window=5, min_count=5, negative=5, iter=20, workers=multiprocessing.cpu_count())
word_vector = w2v.wv
vocabulary = list(word_vector.wv.vocab)


tokenizer_object = Tokenizer()
tokenizer_object.fit_on_texts(sen)
max_length = max([len(doc) for doc in sen])
word_index = tokenizer_object.word_index
sequences = tokenizer_object.texts_to_sequences(sen)
sen_pad = pad_sequences(sequences, maxlen=max_length)
voc = word_index.keys()

file = 'test_word2vector.txt'
w2v.wv.save_word2vec_format(file,binary=False)