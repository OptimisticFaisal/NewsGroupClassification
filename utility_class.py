from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



class Utility:

    def get_data(self,categories):
        nltk.download('punkt')
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

        twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
        twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, categories=categories)
        X = twenty_train.data + twenty_test.data
        Y = np.concatenate((twenty_train.target, twenty_test.target), axis=0)
        return X,Y

    def preprocessing_dataset(self, dataset):
        preprocessed_docs = list()
        for data in dataset:
            tokens = word_tokenize(data)
            # convert to lowercase
            tokens = [word.lower() for word in tokens]
            # remove punctuation from each word
            # table = str.maketrans('','', string.punctuations)
            # filter out stopwords

            # remove tokens that are not alphabet
            tokens = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            preprocessed_docs.append(tokens)
        return preprocessed_docs



    def pad_input(self, docs):
        tokenizer_object = Tokenizer()
        tokenizer_object.fit_on_texts(docs)
        max_length = max([len(doc) for doc in docs])
        max_length_index = np.argmax([len(doc) for doc in docs])
        total_doc_with_length = [index for index, doc in enumerate(docs) if len(doc) > 500]
        sequences = tokenizer_object.texts_to_sequences(docs)

        # pad sequence
        word_index = tokenizer_object.word_index
        doc_pad = pad_sequences(sequences, maxlen=max_length)
        word_embedding_file_name = 'word2vector2.txt'

        num_words = len(word_index) + 1
        return doc_pad, word_index, max_length

    def read_wordembedding(self, file_name):
        embedding_index = {}
        f = open(os.path.join('', file_name), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embedding_index[word] = coefs
        f.close()
        return embedding_index

    def average_embedding(self,word_index, word_embedding_file_name):
        embedding_index = self.read_wordembedding(word_embedding_file_name)
        num_words = len(word_index)+1
        embedding_matrix = np.zeros((num_words, 300))
        weighted_average_emb = {}
        weighted_average_emb[0] = 0
        embedding_word = embedding_index.keys()
        len1 = len(embedding_index)
        len2 = len(word_index)

        for word, i in word_index.items():
            # if i > num_words:
            #   continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                average = np.average(embedding_vector.astype(np.float))
                embedding_matrix[i] = embedding_vector
                weighted_average_emb[i] = average
            else:
                weighted_average_emb[i] = 0
        return weighted_average_emb, embedding_matrix

    def create_embedding_matrix(filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        with open(filepath) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

        return embedding_matrix

