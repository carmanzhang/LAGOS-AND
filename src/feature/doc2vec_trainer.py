import inspect
import logging
import os

from mytookit.data_reader import DBReader
from sklearn.model_selection import train_test_split

from myconfig import cached_dir

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
project_dir_path = os.path.dirname(os.path.abspath(base_file_path))
data_path = project_dir_path

import inspect
import logging
import os
import random

import numpy as np
from gensim.models import doc2vec

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')


class doc2VecModel():

    def __init__(self):
        super().__init__()

    def initialize_model(self, corpus):
        logging.info("Building Doc2Vec vocabulary")
        self.corpus = corpus
        self.model = doc2vec.Doc2Vec(
            epochs=8,
            min_count=1,
            # Ignores all words with
            # total frequency lower than this
            window=10,
            # The maximum distance between the current
            #  and predicted word within a sentence
            vector_size=200,  # Dimensionality of the
            #  generated feature vectors
            workers=12,  # Number of worker threads to
            #  train the model
            alpha=0.025,  # The initial learning rate
            min_alpha=0.00025,
            # Learning rate will linearly drop to
            # min_alpha as training progresses
            dm=1)
        # dm defines the training algorithm.
        #  If dm=1 means 'distributed memory' (PV-DM)
        # and dm =0 means 'distributed bag of words' (PV-DBOW)
        self.model.build_vocab(self.corpus)

    def train_model(self):
        logging.info("Training Doc2Vec model")
        for epoch in range(2):
            logging.info('Training iteration #{0}'.format(epoch))
            self.model.train(
                self.corpus, total_examples=self.model.corpus_count,
                epochs=self.model.epochs)
            # shuffle the corpus
            random.shuffle(self.corpus)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha

    def get_vectors(self, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = self.model.docvecs[prefix]
        return vectors

    def save_model(self, model_path):
        logging.info("Doc2Vec model saved at: " + model_path)
        self.model.save(model_path)

    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each
         document/paragraph to have a label associated with it.
        We do this by using the LabeledSentence method.
        The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the review.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
        return labeled


def prepare_all_data(ds):
    x_train, x_test, y_train, y_test = train_test_split(ds.review, ds.sentiment, random_state=0, test_size=0.1)
    x_train = doc2VecModel.label_sentences(x_train, 'Train')
    x_test = doc2VecModel.label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data


if __name__ == "__main__":
    ds = DBReader.tcp_model_cached_read(os.path.join(cached_dir, 'doc2vec_train_corpus.pkl'),
                                        """select content from and_ds.doc2vec_train_corpus""",
                                        cached=False)
    print(ds.shape)
    print(ds.head())
    ds_content = list(ds['content'])
    ds_content = [item.split() for item in ds_content]
    print('training samples size:', len(ds_content))
    print('first 3 training samples:', ds_content[:3])
    corpus = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(ds_content)]
    # print(corpus[0:4])
    d2v = doc2VecModel()
    d2v.initialize_model(corpus)
    d2v.train_model()
    d2v.save_model(os.path.join(cached_dir, 'doc2vec_model'))
